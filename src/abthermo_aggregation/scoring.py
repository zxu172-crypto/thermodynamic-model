"""Thermodynamic model scoring functions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


R_KCAL_PER_MOL_K = 0.0019872043
DEFAULT_THERMODYNAMIC_MODEL_THRESHOLD = 0.8616561363652143
THRESHOLD_EPSILON = 1e-12


def zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - numeric.mean()) / std


def add_batch_scores(
    descriptor_df: pd.DataFrame,
    temperature_K: float = 313.0,
    risk_threshold: float = DEFAULT_THERMODYNAMIC_MODEL_THRESHOLD,
) -> pd.DataFrame:
    """Add thermodynamic model scores to a batch of surface descriptors.

    Scores are designed for ranking batches. Single-structure runs still produce
    raw descriptors and a score, but z-score terms are necessarily zero.
    """

    out = descriptor_df.copy()
    for column in [
        "hydrophobic_curvature_density",
        "hydrophobic_surface",
        "absolute_charge_surface",
        "hidden_cdr_hydrophobicity",
        "buried_hydrophobicity_all",
        "hydrophobic_curvedness_mean",
    ]:
        out[f"{column}_z"] = zscore(out[column])

    hcm_z = out["hydrophobic_curvature_density_z"]
    hyd_surface_z = out["hydrophobic_surface_z"]
    charge_z = out["absolute_charge_surface_z"]
    hidden_z = out["hidden_cdr_hydrophobicity_z"]
    buried_z = out["buried_hydrophobicity_all_z"]
    curvedness_z = out["hydrophobic_curvedness_mean_z"]

    out["DeltaMu_proxy"] = (
        0.55
        + 0.24 * hcm_z
        + 0.20 * hidden_z
        + 0.16 * hyd_surface_z
        + 0.10 * buried_z
        - 0.10 * charge_z
    ).clip(lower=0.05)

    out["gamma_proxy"] = (
        2.05
        - 0.22 * hcm_z
        - 0.14 * hyd_surface_z
        - 0.10 * curvedness_z
        + 0.12 * charge_z
    ).clip(lower=0.35)

    out["dewetting_drive"] = 0.18 * hcm_z + 0.12 * hyd_surface_z - 0.05 * charge_z
    out["hidden_hydrophobic_drive"] = 0.22 * hidden_z + 0.12 * buried_z
    out["DeltaG_assoc_proxy"] = -out["DeltaMu_proxy"]

    barriers = [
        nucleation_barrier_kT(row.DeltaMu_proxy, row.gamma_proxy, temperature_K)
        for row in out.itertuples(index=False)
    ]
    out["n_star"] = [item["n_star"] for item in barriers]
    out["DeltaG_star_kcal_mol"] = [item["DeltaG_star_kcal_mol"] for item in barriers]
    out["DeltaG_star_kT"] = [item["DeltaG_star_kT"] for item in barriers]
    out["thermodynamic_risk_score"] = 1.0 / (1.0 + np.exp((out["DeltaG_star_kT"] - 10.0) / 4.0))
    out["risk_rank"] = out["thermodynamic_risk_score"].map(_risk_rank)
    out["thermo_call"] = (out["thermodynamic_risk_score"] >= risk_threshold - THRESHOLD_EPSILON).astype(int)

    return out


def nucleation_barrier_kT(
    delta_mu_proxy: float,
    gamma_proxy: float,
    temperature_K: float,
    max_n: int = 500,
) -> dict[str, float | int]:
    kBT = R_KCAL_PER_MOL_K * temperature_K
    best_n = 1
    best_g = -1e99
    for n in range(1, max_n + 1):
        g = gamma_proxy * (n ** (2.0 / 3.0)) - delta_mu_proxy * n
        if g > best_g:
            best_g = g
            best_n = n
    barrier = max(best_g, 0.0)
    return {
        "n_star": best_n,
        "DeltaG_star_kcal_mol": barrier,
        "DeltaG_star_kT": barrier / kBT,
    }


def _risk_rank(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"
