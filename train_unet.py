import os
os.environ["MONAI_SUBMODULE_AUTOIMPORT"] = "0"

import argparse
from pathlib import Path
from typing import Tuple

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

from sklearn.model_selection import KFold

from data_loader import data_laoder
from nnunet3D import NNUNet3D


# -------------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Factorizer-style nnU-Net (5-fold CV)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/brats"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-frac", type=float, default=0.2)  # kept but unused in CV
    parser.add_argument("--roi-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-rate", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--val-interval", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--data-frac", type=float, default=1.0,
                        help="fraction of dataset to use (e.g. 0.2 for 20%)")
    return parser.parse_args()


# -------------------------------------------------------------------------
# CV folds
# -------------------------------------------------------------------------
def create_folds(datalist, n_splits=5, seed=42):
    """
    Create K folds from the full datalist.
    Returns a list of (train_files, val_files) pairs.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    indices = np.arange(len(datalist))

    for train_idx, val_idx in kf.split(indices):
        train_files = [datalist[i] for i in train_idx]
        val_files = [datalist[i] for i in val_idx]
        folds.append((train_files, val_files))

    return folds


# -------------------------------------------------------------------------
# Loss
# -------------------------------------------------------------------------
def factorizer_loss_single(logits: Tensor, labels: Tensor, eps: float = 1e-5) -> Tensor:
    """
    logits: (B, C, D, H, W)
    labels: (B, 1, D, H, W)
    """
    labels_ce = labels.long().squeeze(1)
    ce = F.cross_entropy(logits, labels_ce, reduction="mean")

    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    labels_oh = F.one_hot(labels_ce, num_classes).permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)
    intersection = torch.sum(labels_oh * probs, dim=dims)
    g2 = torch.sum(labels_oh ** 2, dim=dims)
    p2 = torch.sum(probs ** 2, dim=dims)

    dice = 1 - (2 * intersection + eps) / (g2 + p2 + eps)
    dice = dice.mean()
    return dice + ce


def factorizer_deepsup_loss(outputs, labels, lambdas=(1.0, 0.5, 0.25)):
    if isinstance(outputs, Tensor):
        outputs = [outputs]

    total = 0.0
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
        "ET": ((p == 3).astype(np.uint8), (g == 3).astype(np.uint8)),
        "TC": (np.isin(p, [1, 3]).astype(np.uint8), np.isin(g, [1, 3]).astype(np.uint8)),
        "WT": (np.isin(p, [1, 2, 3]).astype(np.uint8), np.isin(g, [1, 2, 3]).astype(np.uint8)),
    }


def compute_brats_metrics_single(pred, gt, device):
    masks = get_brats_masks(pred, gt)
    out = {}
    for region, (pm, gm) in masks.items():
        p = torch.tensor(pm)[None, None].float().to(device)
        g = torch.tensor(gm)[None, None].float().to(device)

        if g.sum() == 0 and p.sum() == 0:
            dice = 1.0
            hd95 = 0.0
        elif g.sum() == 0 or p.sum() == 0:
            dice = 0.0
            hd95 = float("nan")
        else:
            dice = compute_dice(p, g).item()
            try:
                hd95 = compute_hausdorff_distance(p, g, percentile=95).item()
            except Exception:
                hd95 = float("nan")

        out[f"Dice_{region}"] = dice
        out[f"HD95_{region}"] = hd95

    out["Dice_Avg"] = np.nanmean([out["Dice_ET"], out["Dice_TC"], out["Dice_WT"]])
    out["HD95_Avg"] = np.nanmean([out["HD95_ET"], out["HD95_TC"], out["HD95_WT"]])
    return out


# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------
def run_validation(model, val_loader, roi_size, device):
    model.eval()
    all_metrics = []

    with torch.no_grad(), autocast():
        for batch in tqdm(val_loader, desc="Validating"):
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            out = sliding_window_inference(x, roi_size, 1, model)
            if isinstance(out, (list, tuple)):
                out = out[0]

            pred = torch.softmax(out, 1).argmax(1)
            gt = y.squeeze(1).long()

            for pv, gv in zip(decollate_batch(pred), decollate_batch(gt)):
                all_metrics.append(compute_brats_metrics_single(pv, gv, device))

    metrics = {}
    keys = all_metrics[0].keys()
    for k in keys:
        vals = np.array([m[k] for m in all_metrics], dtype=float)
        if "HD95" in k:
            metrics[k] = float(np.nanmean(vals))
        else:
            metrics[k] = float(np.mean(vals))
    return metrics


# -------------------------------------------------------------------------
# Training with 5-fold cross-validation
# -------------------------------------------------------------------------
def main():
    args = parse_args()
    print("🚀 Starting Factorizer-style nnU-Net training (5-fold CV)")
    set_determinism(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi_size: Tuple[int, int, int] = tuple(args.roi_size)

    # Data loader (we'll bypass val_frac and do folds ourselves)
    loader = data_laoder(
        data_dir=str(args.data_dir),
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        roi_size=roi_size,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        seed=args.seed,
        use_cache=args.use_cache,
        data_frac=args.data_frac,
    )

    # Load full datalist once
    full_datalist = loader._load_datalist()

    # Create 5 folds
    folds = create_folds(full_datalist, n_splits=5, seed=args.seed)

    # Root checkpoints directory
    ckpt_root = Path("./checkpoints_cv")
    ckpt_root.mkdir(exist_ok=True)
    warmup_steps = 2000

    for fold_id, (train_files, val_files) in enumerate(folds):
        print(f"\n==============================")
        print(f"   🚀 Starting Fold {fold_id+1}/5")
        print(f"==============================")

        # Build train/val loaders for this fold
        train_loader = loader._build_custom_loader(train_files, is_train=True)
        val_loader = loader._build_custom_loader(val_files, is_train=False)

        # Recreate model, optimizer, scheduler for each fold
        model = NNUNet3D(in_channels=4, out_channels=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_steps - warmup_steps
        )
        scaler = GradScaler()

        # Fold checkpoint directory
        fold_ckpt = ckpt_root / f"fold_{fold_id+1}"
        fold_ckpt.mkdir(exist_ok=True)

        best_dice = -1.0
        global_step = 0

        # Training loop for this fold
        model.train()
        for epoch in range(9999):
            for batch in tqdm(train_loader, desc=f"Fold {fold_id+1} | Epoch {epoch}", leave=False):
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

                # LR warmup + cosine
                if global_step < warmup_steps:
                    lr = args.lr * float(global_step) / warmup_steps
                    for g in optimizer.param_groups:
                        g["lr"] = lr
                else:
                    scheduler.step()

                if global_step % 50 == 0:
                    print(f"Fold {fold_id+1} | Step {global_step} | Loss={loss.item():.4f}")

                # Validation
                if global_step > 0 and global_step % args.val_interval == 0:
                    print(f"\n🔍 [Fold {fold_id+1}] Running validation…")
                    metrics = run_validation(model, val_loader, roi_size, device)
                    for k, v in metrics.items():
                        print(f"[Fold {fold_id+1}] {k}: {v:.4f}")

                    # Save last checkpoint (per fold)
                    last_path = fold_ckpt / "last.pt"
                    torch.save(model.state_dict(), last_path)
                    print(f"💾 [Fold {fold_id+1}] Saved last checkpoint: {last_path}")

                    # Save best based on Dice_Avg (per fold)
                    if metrics["Dice_Avg"] > best_dice:
                        best_dice = metrics["Dice_Avg"]
                        best_path = fold_ckpt / "best.pt"
                        torch.save(model.state_dict(), best_path)
                        print(f"🏆 [Fold {fold_id+1}] New best model: {best_path} (Dice_Avg={best_dice:.4f})")

                    model.train()

                # Optional periodic checkpoint (per fold)
                if global_step > 0 and global_step % 5000 == 0:
                    step_ckpt = fold_ckpt / f"step_{global_step}.pt"
                    torch.save(model.state_dict(), step_ckpt)
                    print(f"📝 [Fold {fold_id+1}] Periodic checkpoint at step {global_step}: {step_ckpt}")

                global_step += 1

            if global_step >= args.max_steps:
                break

    # Final "global" save after all folds
    final_path = ckpt_root / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"✔ Final model saved after all folds: {final_path}")


if __name__ == "__main__":
    main()
