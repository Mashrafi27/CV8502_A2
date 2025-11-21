#!/usr/bin/env python3
"""
Create a copy of model weights with a randomized classifier head.
Useful for Grad-CAM/attribution sanity checks: explanations should degrade when the head is random.

Usage:
  python scripts/randomize_head.py --weights outputs/baseline/best.pt --out outputs/baseline/random_head.pt
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser(description="Randomize classifier head and save new weights.")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    state = torch.load(args.weights, map_location="cpu")
    # Infer classifier shape from keys
    cls_w = None
    cls_b = None
    for k in state:
        if k.endswith("classifier.weight"):
            cls_w = state[k]
        if k.endswith("classifier.bias"):
            cls_b = state[k]
    if cls_w is None or cls_b is None:
        raise ValueError("Could not find classifier weights/bias in state dict.")

    out_features = cls_w.shape[0]
    in_features = cls_w.shape[1]
    head = nn.Linear(in_features, out_features)
    nn.init.kaiming_uniform_(head.weight, a=5 ** 0.5)
    nn.init.zeros_(head.bias)

    for k in state:
        if k.endswith("classifier.weight"):
            state[k] = head.weight
        if k.endswith("classifier.bias"):
            state[k] = head.bias
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    print(f"Wrote randomized head weights to {args.out}")


if __name__ == "__main__":
    main()
