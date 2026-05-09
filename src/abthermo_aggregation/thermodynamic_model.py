"""Frozen thermodynamic model design-proxy scoring path.

This module implements the frozen thermodynamic model feature-to-score contract.
It expects a table of structure-derived descriptors, not labels. The descriptors
can come from any preprocessing pipeline that produces the same columns.
"""

from __future__ import annotations

import math

import pandas as pd

from .scoring import DEFAULT_THERMODYNAMIC_MODEL_THRESHOLD, R_KCAL_PER_MOL_K, THRESHOLD_EPSILON


FEATURE_COLUMNS = [
    "old_mu_kT",
    "old_risk",
    "patch_hydrophobic",
    "patch_hyd_pos",
    "interface_reu",
    "interface_delta_sasa",
    "cnt_bulk",
    "cnt_gamma",
    "cnt_barrier_kT",
    "cnt_risk_score",
    "roughness_z",
    "charge_z",
    "hydrophobic_field_z",
    "exposure_z",
    "cnt_bulk_z",
    "cnt_gamma_z",
    "interface_delta_sasa_z",
    "old_mu_z",
]


def add_thermodynamic_model_scores(
    feature_df: pd.DataFrame,
    temperature_K: float = 313.0,
    risk_threshold: float = DEFAULT_THERMODYNAMIC_MODEL_THRESHOLD,
    include_electrostatic_regularizer: bool = True,
) -> pd.DataFrame:
    """Score a descriptor table without using labels."""

    missing = [column for column in FEATURE_COLUMNS if column not in feature_df.columns]
    if missing:
        raise ValueError(f"Feature manifest is missing required thermodynamic model columns: {missing}")

    rows = []
    for row in feature_df.to_dict(orient="records"):
        scored = thermodynamic_model_proxy(
            row,
            temperature_K=temperature_K,
            include_electrostatic_regularizer=include_electrostatic_regularizer,
        )
        rows.append({**row, **scored})

    out = pd.DataFrame(rows)
    out["thermo_call"] = (out["thermodynamic_risk_score"] >= risk_threshold - THRESHOLD_EPSILON).astype(int)
    return out


def thermodynamic_model_proxy(
    features: dict,
    temperature_K: float = 313.0,
    ionic_strength_mM: float = 150.0,
    surface_context: str = "bulk",
    include_electrostatic_regularizer: bool = True,
) -> dict[str, float | int | str]:
    """Map structure descriptors into the thermodynamic model nucleation proxy."""

    kBT = R_KCAL_PER_MOL_K * temperature_K
    patch = _to_float(features["patch_hydrophobic"], 0.0)
    hydpos = _to_float(features["patch_hyd_pos"], 0.0)
    dsasa = max(_to_float(features["interface_delta_sasa"], 0.0), 0.0)
    interface_reu = _to_float(features["interface_reu"], 0.0)
    favorable_interface_reu = max(0.0, -interface_reu)
    positive_interface_stress = min(math.log1p(max(interface_reu, 0.0)) / 9.0, 1.5)
    rough_z = _to_float(features["roughness_z"], 0.0)
    charge_z = _to_float(features["charge_z"], 0.0)
    hyd_z = _to_float(features["hydrophobic_field_z"], 0.0)
    exposure_z = _to_float(features["exposure_z"], 0.0)
    bulk_z = _to_float(features["cnt_bulk_z"], 0.0)
    gamma_z = _to_float(features["cnt_gamma_z"], 0.0)
    dsasa_z = _to_float(features["interface_delta_sasa_z"], 0.0)
    old_mu_z = _to_float(features["old_mu_z"], 0.0)
    old_mu_kT = max(_to_float(features["old_mu_kT"], 0.0), 0.0)
    cnt_bulk = max(_to_float(features["cnt_bulk"], 0.0), 0.0)

    local_patch_correction_proxy = 0.12 * hydpos + 0.10 * max(hyd_z, 0.0) - 0.06 * max(charge_z, 0.0)

    surface_curvature_roughness_z = max(0.0, rough_z)
    hydrophobic_curvature_coupling = surface_curvature_roughness_z * (1.0 + 0.25 * max(hyd_z, 0.0))
    curvature_gamma_discount = min(
        0.60,
        0.08 * surface_curvature_roughness_z + 0.02 * hydrophobic_curvature_coupling,
    )

    latent_aggregation_drive = (
        0.75 * rough_z
        + 0.70 * bulk_z
        + 0.25 * gamma_z
        + 0.35 * dsasa_z
        + 0.65 * positive_interface_stress
        - 0.35 * old_mu_z
        + 0.15 * hyd_z
    )
    if surface_context == "air_water":
        latent_aggregation_drive += 0.75
    elif surface_context == "solid":
        latent_aggregation_drive += 0.50

    delta_g_assoc_proxy = -(
        kBT * max(0.0, 0.45 + 0.25 * latent_aggregation_drive)
        + 0.10 * patch
        + 0.04 * hydpos
        + 0.010 * favorable_interface_reu
        + 0.00010 * dsasa
        + local_patch_correction_proxy
    )
    delta_g_dewetting = -(
        0.12 * patch
        + 0.00012 * dsasa
        + 0.06 * max(hyd_z, 0.0)
        + 0.05 * hydrophobic_curvature_coupling
    )
    delta_g_unfold_proxy = max(
        0.50,
        5.50 - 0.70 * rough_z - 0.25 * exposure_z - 0.15 * patch,
    )

    ionic_screen = math.sqrt(max(float(ionic_strength_mM), 1.0) / 150.0)
    # Empirical regularizer retained for frozen thermodynamic model compatibility.
    # It partially overlaps with the charge contribution in gamma_proxy, so it
    # should not be presented as an independent first-principles electrostatic
    # free-energy term.
    electrostatic_penalty = max(0.05, 0.75 + 0.20 * charge_z) / ionic_screen
    electrostatic_term = electrostatic_penalty if include_electrostatic_regularizer else 0.0

    delta_mu_proxy = max(
        0.05,
        0.45 + 0.28 * latent_aggregation_drive + 2.0 * cnt_bulk + 0.02 * old_mu_kT,
    )
    gamma_proxy = max(
        0.35,
        2.15
        - 0.30 * latent_aggregation_drive
        - curvature_gamma_discount
        + 0.12 * charge_z
        + 0.03 * (float(ionic_strength_mM) / 150.0 - 1.0),
    )

    nuc = nucleation_profile(delta_mu_proxy, gamma_proxy, electrostatic_term, temperature_K)
    risk_score = 1.0 / (1.0 + math.exp((nuc["DeltaG_star_kT"] - 10.0) / 4.0))

    return {
        "DeltaG_assoc_proxy": delta_g_assoc_proxy,
        "DeltaG_unfold_proxy": delta_g_unfold_proxy,
        "DeltaG_dewetting": delta_g_dewetting,
        "local_patch_correction_proxy": local_patch_correction_proxy,
        "latent_aggregation_drive": latent_aggregation_drive,
        "surface_curvature_roughness_z": surface_curvature_roughness_z,
        "hydrophobic_curvature_coupling": hydrophobic_curvature_coupling,
        "curvature_gamma_discount": curvature_gamma_discount,
        "gamma_proxy": gamma_proxy,
        "DeltaMu_proxy": delta_mu_proxy,
        "electrostatic_regularizer_kcal": electrostatic_penalty,
        "n_star": nuc["n_star"],
        "DeltaG_star_kcal_mol": nuc["DeltaG_star_kcal_mol"],
        "DeltaG_star_kT": nuc["DeltaG_star_kT"],
        "k_nuc_relative": nuc["k_nuc_relative"],
        "thermodynamic_risk_score": risk_score,
        "aggregation_pathway": classify_pathway(delta_g_unfold_proxy, favorable_interface_reu, dsasa, electrostatic_penalty),
        "risk_rank": risk_rank(risk_score),
    }


def nucleation_profile(
    delta_mu_proxy: float,
    gamma_proxy: float,
    electrostatic_regularizer: float,
    temperature_K: float,
    max_n: int = 500,
) -> dict[str, float | int]:
    kBT = R_KCAL_PER_MOL_K * temperature_K
    best_n = 1
    best_g = -1e99
    for n in range(1, max_n + 1):
        g = gamma_proxy * (n ** (2.0 / 3.0)) - delta_mu_proxy * n + electrostatic_regularizer * (n ** (1.0 / 3.0))
        if g > best_g:
            best_g = g
            best_n = n
    barrier = max(best_g, 0.0)
    barrier_kT = barrier / kBT
    return {
        "n_star": best_n,
        "DeltaG_star_kcal_mol": barrier,
        "DeltaG_star_kT": barrier_kT,
        "k_nuc_relative": math.exp(-min(barrier_kT, 700.0)),
    }


def classify_pathway(
    delta_g_unfold_proxy: float,
    favorable_interface_reu: float,
    interface_delta_sasa: float,
    electrostatic_regularizer: float,
) -> str:
    if favorable_interface_reu > 20.0 and interface_delta_sasa > 800.0:
        return "native antibody self-association"
    if delta_g_unfold_proxy < 3.5:
        return "partial-unfolding-assisted aggregation"
    if electrostatic_regularizer > 1.2:
        return "pH/ionic-strength-sensitive colloidal association"
    return "hydrophobic-patch/dewetting-driven association"


def risk_rank(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def _to_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result
