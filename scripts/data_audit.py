#!/usr/bin/env python3
"""
Compute label prevalence and subgroup counts with 95% Wilson CIs.
Usage:
  python scripts/data_audit.py --csv <file> --labels Effusion --group-col sex --out data_audit
Outputs:
  data_audit/label_prevalence.csv
  data_audit/group_counts.csv (if group-col provided)
"""
import argparse
from pathlib import Path
import pandas as pd
import math


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    low = (center - margin) / denom
    high = (center + margin) / denom
    return low, high


def main():
    ap = argparse.ArgumentParser(description="Data audit: prevalence and subgroup counts.")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--labels", nargs="+", required=True, help="Label columns (binary).")
    ap.add_argument("--group-col", default=None, help="Optional subgroup column.")
    ap.add_argument("--out", type=Path, default=Path("data_audit"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)

    # Label prevalence
    rows = []
    for lbl in args.labels:
        if lbl not in df.columns:
            raise ValueError(f"Label column '{lbl}' not found.")
        k = int(df[lbl].sum())
        n = int(len(df))
        low, high = wilson_ci(k, n)
        rows.append(
            {
                "label": lbl,
                "count_pos": k,
                "count_total": n,
                "prevalence": k / n if n else float("nan"),
                "ci_low_95": low,
                "ci_high_95": high,
            }
        )
    pd.DataFrame(rows).to_csv(args.out / "label_prevalence.csv", index=False)

    # Subgroup counts + prevalence
    if args.group_col:
        if args.group_col not in df.columns:
            raise ValueError(f"Group column '{args.group_col}' not found.")
        g_rows = []
        for g, gdf in df.groupby(args.group_col):
            for lbl in args.labels:
                k = int(gdf[lbl].sum())
                n = int(len(gdf))
                low, high = wilson_ci(k, n)
                g_rows.append(
                    {
                        "group": g,
                        "label": lbl,
                        "count_pos": k,
                        "count_total": n,
                        "prevalence": k / n if n else float("nan"),
                        "ci_low_95": low,
                        "ci_high_95": high,
                    }
                )
        pd.DataFrame(g_rows).to_csv(args.out / "group_counts.csv", index=False)

    print(f"Wrote audit to {args.out}")


if __name__ == "__main__":
    main()
