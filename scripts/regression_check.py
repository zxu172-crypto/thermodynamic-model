"""Local regression checks for HCNP candidate outputs.

This script is intentionally data-agnostic: pass local score, label, and assay
tables at runtime. It is not used by the public CLI and does not hard-code FLAb
or local filesystem paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from abthermo_aggregation.scoring import DEFAULT_HCNP_THRESHOLD, THRESHOLD_EPSILON


TARGET_ASSAYS = {
    "jain2017biophysical_ACSINS.csv",
    "jain2017biophysical_SGACSINS.csv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate HCNP candidate scores against local labels/assays.")
    parser.add_argument("--scores", type=Path, required=True, help="Candidate output CSV.")
    parser.add_argument("--labels", type=Path, required=True, help="CSV with antibody_id and label columns.")
    parser.add_argument(
        "--flab-numeric",
        type=Path,
        default=None,
        help="Optional mapped numeric FLAb CSV with antibody_id, assay_file, and risk_oriented_value.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_HCNP_THRESHOLD)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scores = pd.read_csv(args.scores)
    labels = pd.read_csv(args.labels)
    merged = scores.merge(labels[["antibody_id", "label"]], on="antibody_id", how="inner")
    if merged.empty:
        raise SystemExit("No score rows matched the label table by antibody_id.")

    metrics = confusion_metrics(merged["label"], merged["hcnp_risk_score"] >= args.threshold - THRESHOLD_EPSILON)
    print("HCNP candidate confusion")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    if args.flab_numeric is not None:
        assay_values = pd.read_csv(args.flab_numeric)
        assay_values = assay_values[assay_values["assay_file"].isin(TARGET_ASSAYS)].copy()
        correlations = correlate_assays(scores, assay_values)
        print("\nAC-SINS / SGAC-SINS Pearson-first correlations")
        for row in correlations:
            print(
                f"{row['assay_file']}: n={row['n']}, "
                f"primary_Pearson={row['pearson']:.3f}, "
                f"secondary_Spearman={row['spearman']:.3f}"
            )
    return 0


def confusion_metrics(labels: pd.Series, predicted_positive: pd.Series) -> dict[str, float | int]:
    y = labels.astype(int)
    pred = predicted_positive.astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    n = len(y)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
    }


def correlate_assays(scores: pd.DataFrame, assay_values: pd.DataFrame) -> list[dict[str, float | int | str]]:
    rows = []
    for assay_file, subset in assay_values.groupby("assay_file"):
        merged = scores[["antibody_id", "hcnp_risk_score"]].merge(
            subset[["antibody_id", "risk_oriented_value"]],
            on="antibody_id",
            how="inner",
        )
        merged = merged.dropna()
        if len(merged) < 3:
            continue
        pearson = pearsonr(merged["hcnp_risk_score"], merged["risk_oriented_value"]).statistic
        spearman = spearmanr(merged["hcnp_risk_score"], merged["risk_oriented_value"]).statistic
        rows.append(
            {
                "assay_file": assay_file,
                "n": int(len(merged)),
                "pearson": float(pearson),
                "spearman": float(spearman),
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
