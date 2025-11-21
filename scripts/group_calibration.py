#!/usr/bin/env python3
"""
Compute overall and per-group calibration (ECE) and thresholds for a predictions CSV.
Assumes columns: label column (e.g., 'Effusion'), probability column (e.g., 'prob_Effusion'),
and a group column (e.g., 'sex').

Usage:
  python scripts/group_calibration.py \
    --preds outputs/baseline_eval/preds_test.csv \
    --label Effusion \
    --prob prob_Effusion \
    --group sex \
    --target-spec 0.95
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    y_true = y_true.astype(int)
    conf = np.where(probs >= 0.5, probs, 1 - probs)
    preds = (probs >= 0.5).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idxs = np.digitize(conf, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idxs == b
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == y_true[mask]).mean()
        c = conf[mask].mean()
        ece += (mask.sum() / len(conf)) * abs(acc - c)
    return float(ece)


def tpr_at_specificity(y_true: np.ndarray, y_prob: np.ndarray, target_spec: float):
    fpr, tpr, thresh = roc_curve(y_true, y_prob)
    spec = 1 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0:
        return float("nan"), None
    best = idx[-1]
    return float(tpr[best]), float(thresh[best])


def main():
    ap = argparse.ArgumentParser(description="Group-wise calibration and thresholds.")
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--label", required=True, help="Ground-truth column")
    ap.add_argument("--prob", required=True, help="Probability column for positive class")
    ap.add_argument("--group", required=True, help="Group column")
    ap.add_argument("--target-spec", type=float, default=0.95)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    for col in (args.label, args.prob, args.group):
        if col not in df.columns:
            raise ValueError(f"Column {col} missing in predictions CSV")

    y_true = df[args.label].values
    y_prob = df[args.prob].values
    ece_all = ece_score(y_true, y_prob, n_bins=args.bins)
    tpr95_all, thr95_all = tpr_at_specificity(y_true, y_prob, args.target_spec)

    by_group = []
    for g, gdf in df.groupby(args.group):
        yt = gdf[args.label].values
        yp = gdf[args.prob].values
        ece_g = ece_score(yt, yp, n_bins=args.bins)
        tpr95_g, thr95_g = tpr_at_specificity(yt, yp, args.target_spec)
        by_group.append({
            "group": g,
            "count": int(len(gdf)),
            "ece": ece_g,
            "tpr_at_spec": tpr95_g,
            "threshold_at_spec": thr95_g,
        })

    out = args.out or (args.preds.parent / "group_calibration_summary.json")
    summary = {
        "preds_file": str(args.preds),
        "label": args.label,
        "prob": args.prob,
        "group": args.group,
        "target_spec": args.target_spec,
        "overall": {"ece": ece_all, "tpr_at_spec": tpr95_all, "threshold_at_spec": thr95_all},
        "groups": by_group,
    }
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
