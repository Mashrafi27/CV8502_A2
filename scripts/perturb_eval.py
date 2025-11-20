#!/usr/bin/env python3
"""
Evaluate a trained model on clean vs mild perturbations (contrast ±10%, Gaussian noise ~PSNR>30dB).
Usage:
  python scripts/perturb_eval.py --csv <file> --img-root <root> --labels Effusion \
    --weights outputs/baseline/best.pt --split test --outdir outputs/perturb_baseline
Outputs:
  outdir/clean_metrics.json
  outdir/perturbed_metrics.json
"""
import argparse
import json
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torch.utils.data import DataLoader

from main import (
    MultiLabelImageDataset,
    build_model,
    metrics_from_logits,
    seed_everything,
    get_device,
)


def mild_transform(image_size: int):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT),
            A.CenterCrop(image_size, image_size),
            A.GaussNoise(var_limit=(5.0, 5.0), p=1.0),  # small noise ~ high PSNR
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def clean_transform(image_size: int):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT),
            A.CenterCrop(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def run_eval(csv, img_root, labels, weights, split, group_col, image_size, batch_size, workers, outdir):
    device = get_device()
    model = build_model(num_classes=len(labels), drop_rate=0.2, pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    def eval_with_tf(tf, tag):
        ds = MultiLabelImageDataset(csv, img_root, labels, split=split, transform=tf, group_col=group_col)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
        logits_all = []
        labels_all = []
        for batch in loader:
            images = batch["image"].to(device)
            labels_np = batch["labels"].numpy()
            with torch.no_grad():
                logits = model(images)
            logits_all.append(logits.cpu().numpy())
            labels_all.append(labels_np)
        import numpy as np

        logits_all = np.concatenate(logits_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        metrics = metrics_from_logits(logits_all, labels_all, labels, threshold=0.5)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        with open(Path(outdir) / f"{tag}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics["macro"]

    clean_macro = eval_with_tf(clean_transform(image_size), "clean")
    pert_macro = eval_with_tf(mild_transform(image_size), "perturbed")
    print("Clean macro:", json.dumps(clean_macro, indent=2))
    print("Perturbed macro:", json.dumps(pert_macro, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Evaluate model on mild perturbations.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    seed_everything(args.seed)
    run_eval(
        csv=args.csv,
        img_root=args.img_root,
        labels=args.labels,
        weights=args.weights,
        split=args.split,
        group_col=args.group_col,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
