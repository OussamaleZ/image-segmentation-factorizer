
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np

from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.utils import set_determinism

from data_loader import data_laoder
from nnunet3D import NNUNet3D
from train_unet import compute_brats_metrics_single   # reuse your function


def parse_args():
    parser = argparse.ArgumentParser("Evaluate checkpoint")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint, e.g. checkpoints/best.pt")
    parser.add_argument("--data-dir", type=str, default="data/brats")
    parser.add_argument("--roi-size", type=int, nargs=3, default=(128,128,128))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-frac", type=float, default=0.2)
    return parser.parse_args()


def run_validation(model, val_loader, roi_size, device):
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating checkpoint"):
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            out = sliding_window_inference(x, roi_size, 1, model)
            if isinstance(out, (list, tuple)):
                out = out[0]

            pred = torch.softmax(out, 1).argmax(1)
            gt = y.squeeze(1).long()

            for pv, gv in zip(decollate_batch(pred), decollate_batch(gt)):
                all_metrics.append(compute_brats_metrics_single(pv, gv, device))

    # aggregate results using nanmean for HD95
    results = {}
    keys = all_metrics[0].keys()
    for k in keys:
        vals = np.array([m[k] for m in all_metrics], dtype=float)
        if "HD95" in k:
            results[k] = float(np.nanmean(vals))
        else:
            results[k] = float(np.mean(vals))

    return results


def main():
    args = parse_args()
    set_determinism(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi_size = tuple(args.roi_size)

    print(f"📦 Loading checkpoint: {args.ckpt}")
    print("📂 Loading dataset...")

    loader = data_laoder(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        roi_size=roi_size,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        seed=args.seed,
        use_cache=False
    )

    _, val_loader = loader.get_loaders()

    # build model
    model = NNUNet3D(in_channels=4, out_channels=4).to(device)

    # load checkpoint
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    print("✔ Checkpoint loaded.")

    # evaluate
    metrics = run_validation(model, val_loader, roi_size, device)
    print("\n📊 Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
