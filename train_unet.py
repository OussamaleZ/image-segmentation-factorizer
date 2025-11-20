import os
os.environ["MONAI_SUBMODULE_AUTOIMPORT"] = "0"   # MONAI import speed-up

import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.utils import set_determinism

from data_loader import data_laoder
from nnunet3D import NNUNet3D


# -------------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/brats"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--roi-size", type=int, nargs=3, default=(128,128,128))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-rate", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=100000)  # factorizer
    parser.add_argument("--val-interval", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cache", action="store_true")
    return parser.parse_args()


# -------------------------------------------------------------------------
# Factorizer Single-scale Loss: L = Dice + CE
# -------------------------------------------------------------------------
def factorizer_loss_single(logits, labels, eps=1e-5):
    """
    logits: (B, C, D, H, W)
    labels: (B, 1, D, H, W)
    """
    labels_ce = labels.long().squeeze(1)
    ce = F.cross_entropy(logits, labels_ce, reduction="mean")

    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    labels_oh = F.one_hot(labels_ce, num_classes).permute(0,4,1,2,3).float()

    dims = (0,2,3,4)

    intersection = torch.sum(labels_oh * probs, dim=dims)
    g2 = torch.sum(labels_oh**2, dim=dims)
    p2 = torch.sum(probs**2, dim=dims)

    dice = 1 - (2*intersection + eps) / (g2 + p2 + eps)
    dice = dice.mean()

    return dice + ce


# -------------------------------------------------------------------------
# Multi-scale Deep Supervision (Factorizer Eq. 15)
# -------------------------------------------------------------------------
def factorizer_deepsup_loss(outputs, labels, lambdas=(1.0, 0.5, 0.25)):
    if isinstance(outputs, Tensor):
        outputs = [outputs]

    total = 0
    for w, out in zip(lambdas, outputs):
        if w == 0:
            continue

        target_size = out.shape[2:]
        lab_resized = F.interpolate(labels.float(), size=target_size, mode="nearest")

        total += w * factorizer_loss_single(out, lab_resized)

    return total


# -------------------------------------------------------------------------
# BraTS metrics
# -------------------------------------------------------------------------
def get_brats_masks(pred, gt):
    p = pred.cpu().numpy()
    g = gt.cpu().numpy()

    return {
        "ET": ((p==3).astype(np.uint8), (g==3).astype(np.uint8)),
        "TC": (np.isin(p,[1,3]).astype(np.uint8), np.isin(g,[1,3]).astype(np.uint8)),
        "WT": (np.isin(p,[1,2,3]).astype(np.uint8), np.isin(g,[1,2,3]).astype(np.uint8)),
    }

def compute_brats_metrics_single(pred, gt, device):
    masks = get_brats_masks(pred, gt)
    out = {}

    for region,(pm,gm) in masks.items():
        p = torch.tensor(pm)[None,None].float().to(device)
        g = torch.tensor(gm)[None,None].float().to(device)

        if g.sum()==0 and p.sum()==0:
            dice=1.0
            hd95=0.0
        elif g.sum()==0 or p.sum()==0:
            # One is empty, other is not -> Dice = 0, HD95 = NaN or skip
            dice=0.0
            hd95=float('nan')
        else:
            dice = compute_dice(p,g).item()
            try:
                hd95 = compute_hausdorff_distance(p,g,percentile=95).item()
            except:
                hd95 = float('nan')

        out[f"Dice_{region}"] = dice
        out[f"HD95_{region}"] = hd95

    out["Dice_Avg"] = np.nanmean([out["Dice_ET"], out["Dice_TC"], out["Dice_WT"]])
    out["HD95_Avg"] = np.nanmean([out["HD95_ET"], out["HD95_TC"], out["HD95_WT"]])
    return out

# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------
def run_validation(model,val_loader,roi_size,device):
    model.eval()
    all_metrics=[]

    with torch.no_grad(), autocast():
        for batch in tqdm(val_loader, desc="Validating"):
            x=batch["image"].to(device)
            y=batch["label"].to(device)

            out = sliding_window_inference(x, roi_size, 1, model)
            if isinstance(out,(list,tuple)):
                out=out[0]

            pred = torch.softmax(out,1).argmax(1)
            gt = y.squeeze(1).long()
            print(f"Predicted classes: {torch.unique(pred)}")
            print(f"GT classes: {torch.unique(gt)}")
            print(f"Pred shape: {pred.shape}, GT shape: {gt.shape}")
            for pv,gv in zip(decollate_batch(pred), decollate_batch(gt)):
                all_metrics.append(compute_brats_metrics_single(pv,gv,device))

    return {k:float(np.nanmean([m[k] for m in all_metrics])) for k in all_metrics[0]}


# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------
def train():
    args = parse_args()
    print("🚀 Starting Factorizer-style nnU-Net Training")
    set_determinism(args.seed)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi_size=tuple(args.roi_size)

    loader=data_laoder(
        data_dir=str(args.data_dir),
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        roi_size=roi_size,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        seed=args.seed,
        use_cache=args.use_cache
    )

    train_loader,val_loader=loader.get_loaders()

    model=NNUNet3D(in_channels=4,out_channels=4).to(device)

    # Factorizer optimizer
    optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Warmup + Cosine Annealing (Factorizer)
    warmup_steps=2000
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.max_steps-warmup_steps)

    scaler=GradScaler()

    global_step=0
    model.train()
    ckpt_dir = Path("./checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    best_dice = -1  # track best Dice_Avg

    for epoch in range(9999):  # infinite loop → break at max_steps
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            if global_step >= args.max_steps:
                break

            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(x)
                loss = factorizer_deepsup_loss(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # ---- LR warmup ----
            if global_step < warmup_steps:
                lr = args.lr * float(global_step) / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = lr
            else:
                scheduler.step()

            if global_step % 50 == 0:
                print(f"Step {global_step} | Loss={loss.item():.4f}")

            # ---- VALIDATION ----
            if global_step > 0 and global_step % args.val_interval == 0:
                print("\n🔍 Running validation…")
                metrics = run_validation(model, val_loader, roi_size, device)
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}")

                # ---- SAVE "last.pt" checkpoint (overwrite each time) ----
                last_path = ckpt_dir / "last.pt"
                torch.save(model.state_dict(), last_path)
                print(f"💾 Saved last checkpoint: {last_path}")

                # ---- SAVE best checkpoint based on Dice_Avg ----
                if metrics["Dice_Avg"] > best_dice:
                    best_dice = metrics["Dice_Avg"]
                    best_path = ckpt_dir / "best.pt"
                    torch.save(model.state_dict(), best_path)
                    print(f"🏆 New best model saved: {best_path} (Dice_Avg={best_dice:.4f})")

                model.train()

            # ---- SAVE checkpoint every 5000 steps (overwrite same file) ----
            if global_step > 0 and global_step % 5000 == 0:
                step_ckpt = ckpt_dir / "step_ckpt.pt"
                torch.save(model.state_dict(), step_ckpt)
                print(f"📝 Periodic checkpoint saved at step {global_step}: {step_ckpt}")

            global_step += 1

        if global_step >= args.max_steps:
            break

    # Final save
    final_path = ckpt_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"✔ Final model saved: {final_path}")



if __name__=="__main__":
    train()
