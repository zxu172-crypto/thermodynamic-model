"""Build a locked-HCNP feature manifest from local structure-descriptor CSVs.

The script is generic and path-driven. It does not embed labels or local paths
unless the input CSVs themselves contain those fields.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


STAT_KEYS = [
    "cmc_roughness_mass",
    "cmc_charge_field_mass",
    "cmc_hydrophobic_field_mass",
    "cmc_exposure_field_mass",
    "cmc_delta_g_bulk",
    "cmc_gamma_eff",
    "best_self_interface_delta_sasa",
    "CMC_DeltaMu_eff",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an HCNP feature manifest from descriptor CSVs.")
    parser.add_argument("--surface-predictions", type=Path, required=True)
    parser.add_argument("--cnt-predictions", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=None, help="Optional panel CSV for antibody_id/pdb_id mapping.")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    surface = pd.read_csv(args.surface_predictions)
    cnt = pd.read_csv(args.cnt_predictions)
    panel = pd.read_csv(args.panel) if args.panel is not None else pd.DataFrame()

    merged = surface.merge(cnt, on="pdb_id", how="outer", suffixes=("", "_cnt"))
    if not panel.empty:
        keep = [col for col in ["pdb_id", "antibody_id"] if col in panel.columns]
        merged = merged.merge(panel[keep].drop_duplicates("pdb_id"), on="pdb_id", how="left", suffixes=("", "_panel"))
    if "antibody_id" not in merged.columns:
        merged["antibody_id"] = merged["pdb_id"]
    merged["antibody_id"] = merged["antibody_id"].fillna(merged["pdb_id"])

    stats = panel_stats(merged, STAT_KEYS)
    rows = []
    for row in merged.to_dict(orient="records"):
        pdb_id = str(row.get("pdb_id", "")).upper()
        rows.append(
            {
                "pdb_id": pdb_id,
                "antibody_id": row.get("antibody_id") or pdb_id,
                "old_mu_kT": to_float(row.get("CMC_DeltaMu_eff"), 0.25),
                "old_risk": to_float(row.get("CMC_risk"), 0.30),
                "patch_hydrophobic": to_float(row.get("structure_top5_hydrophobic_patch"), 0.8),
                "patch_hyd_pos": to_float(row.get("structure_max_hyd_pos_patch"), 1.0),
                "interface_reu": to_float(row.get("best_self_interface_dG_REU"), 0.0),
                "interface_delta_sasa": to_float(row.get("best_self_interface_delta_sasa"), 0.0),
                "cnt_bulk": to_float(row.get("cmc_delta_g_bulk"), 0.02),
                "cnt_gamma": to_float(row.get("cmc_gamma_eff"), 0.02),
                "cnt_barrier_kT": to_float(row.get("cmc_barrier_kT"), math.nan),
                "cnt_risk_score": to_float(row.get("cmc_risk_score"), 0.30),
                "roughness_z": zscore(row.get("cmc_roughness_mass"), "cmc_roughness_mass", stats),
                "charge_z": zscore(row.get("cmc_charge_field_mass"), "cmc_charge_field_mass", stats),
                "hydrophobic_field_z": zscore(row.get("cmc_hydrophobic_field_mass"), "cmc_hydrophobic_field_mass", stats),
                "exposure_z": zscore(row.get("cmc_exposure_field_mass"), "cmc_exposure_field_mass", stats),
                "cnt_bulk_z": zscore(row.get("cmc_delta_g_bulk"), "cmc_delta_g_bulk", stats),
                "cnt_gamma_z": zscore(row.get("cmc_gamma_eff"), "cmc_gamma_eff", stats),
                "interface_delta_sasa_z": zscore(
                    row.get("best_self_interface_delta_sasa"),
                    "best_self_interface_delta_sasa",
                    stats,
                ),
                "old_mu_z": zscore(row.get("CMC_DeltaMu_eff"), "CMC_DeltaMu_eff", stats),
            }
        )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    return 0


def panel_stats(rows: pd.DataFrame, keys: list[str]) -> dict[str, dict[str, float]]:
    stats = {}
    for key in keys:
        values = pd.to_numeric(rows.get(key), errors="coerce").dropna()
        if values.empty:
            stats[key] = {"mean": 0.0, "sd": 1.0}
        else:
            sd = float(values.std(ddof=1))
            stats[key] = {"mean": float(values.mean()), "sd": sd if sd > 0 else 1.0}
    return stats


def zscore(value, key: str, stats: dict[str, dict[str, float]]) -> float:
    value = to_float(value, math.nan)
    if math.isnan(value):
        return 0.0
    stat = stats.get(key, {"mean": 0.0, "sd": 1.0})
    return (value - stat["mean"]) / stat["sd"]


def to_float(value, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


if __name__ == "__main__":
    raise SystemExit(main())
