# thermodynamic model

GitHub-ready candidate for **Method 01: PyRosetta
Structure-Thermodynamic Nucleation Workflow**, restricted to the standalone
**Hydrophobic-Curvature Nucleation Proxy (HCNP)** branch.

HCNP maps structure-derived antibody descriptors into effective
`DeltaMu_proxy` and `gamma_proxy`, then into a CNT-like nucleation barrier and
`hcnp_risk_score`. The public claim is deliberately limited: this is a static
surface-thermodynamic screening workflow, not a state-of-the-art predictor and
not a production simulation workflow.

## Two Run Modes

### 1. Locked Method 01 Mode

This is the benchmark-equivalent path. It scores a descriptor manifest generated
from the same kind of PyRosetta/CNT structure features used by the internal
notebook.

```bash
python scripts/build_feature_manifest.py \
  --surface-predictions surface_predictions.csv \
  --cnt-predictions cnt_predictions.csv \
  --panel optional_panel_without_labels.csv \
  --out hcnp_feature_manifest.csv

abthermo-screen \
  --feature-manifest hcnp_feature_manifest.csv \
  --out hcnp_scores.csv
```

This mode does not use labels during scoring.

### 2. Lightweight PDB Mode

This mode accepts PDB files directly and computes an open, approximate surface
descriptor set from coordinates. It is useful for exploratory screening, but it
is **not** the frozen reliable-30 benchmark-equivalent score path.

```bash
abthermo-screen \
  --pdb ./pdbs/example_fab.pdb \
  --heavy-chain H \
  --light-chain L \
  --out aggregation_scores.csv
```

For a directory:

```bash
abthermo-screen \
  --pdb-dir ./pdbs \
  --heavy-chain H \
  --light-chain L \
  --out aggregation_scores.csv
```

For a PDB manifest:

```bash
abthermo-screen \
  --manifest examples/manifest_template.csv \
  --out aggregation_scores.csv
```

Manifest columns:

```text
pdb_path,antibody_id,pdb_id,heavy_chain,light_chain
```

Only `pdb_path` is required. Explicit heavy/light chain IDs are recommended for
antibody-antigen structures.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.10+ is recommended.

## Thermodynamic Core

The feature-to-score path is:

```text
structure descriptors
-> hydrophobic / curvature / hidden-hydrophobic terms
-> DeltaMu_proxy and gamma_proxy
-> CNT-like barrier
-> hcnp_risk_score
```

Hydrophobic curvature enters as:

\[
C_{h\kappa}^{\mathrm{proxy}}
=\max(0,z_{\mathrm{rough}})
\left[1+0.25\max(0,z_h)\right].
\]

The validated Method 01 barrier keeps the original small electrostatic
regularizer:

\[
\Delta G(n)=
\gamma_{\mathrm{proxy}}n^{2/3}
-n|\Delta\mu_{\mathrm{proxy}}|
+\lambda_{\mathrm{elec}}n^{1/3}.
\]

The last term is **not** claimed as a clean first-principles CNT term. It is an
empirical electrostatic regularizer retained from the frozen Method 01 score.
Because charge already contributes to `gamma_proxy`, this term is partly
redundant from a strict thermodynamic standpoint. I tested the textbook-clean
version without the last term; it did **not** pass the frozen binary gate, so the
regularized version remains the accepted Method 01 score.

## Output Schema

Key output columns:

- `antibody_id`
- `pdb_id`
- `DeltaG_assoc_proxy`
- `DeltaMu_proxy`
- `gamma_proxy`
- `n_star`
- `DeltaG_star_kT`
- `hcnp_risk_score`
- `risk_rank`
- `thermo_call`

Locked mode also reports:

- `DeltaG_unfold_proxy`
- `DeltaG_dewetting`
- `local_patch_correction_proxy`
- `latent_aggregation_drive`
- `surface_curvature_roughness_z`
- `hydrophobic_curvature_coupling`
- `curvature_gamma_discount`
- `electrostatic_regularizer_kcal`

## Benchmark Summary

Frozen internal Method 01 reference on the reliable-30 public-structure panel:

```text
accuracy = 0.867
TP = 5
TN = 21
FP = 2
FN = 2
balanced_accuracy = 0.814
```

The copied locked scorer reproduces this result using the same threshold
(`hcnp_risk_score >= 0.861656`, with a small floating-point tolerance).

Pearson-first correlation check against mapped FLAb numeric assays under the
same local validation script:

```text
AC-SINS:   Pearson = 0.335, Spearman = 0.255, n = 30
SGAC-SINS: Pearson = 0.452, Spearman = 0.498, n = 30
```

For continuous FLAb assay values, Pearson correlation is treated as the primary
validation metric because it tests whether changes in `hcnp_risk_score` track
the assay magnitude. Spearman correlation is still reported as a secondary
rank-robustness check.

The lightweight PDB-only approximation did not pass the reliable-30 acceptance
gate (`accuracy = 0.667`, `FP = 4`, `FN = 6`), so it is kept only as an
exploratory open fallback.

## Limitations

HCNP is a screening/ranking proxy. It is sensitive to structure quality, chain
selection, missing residues, glycosylation handling, and batch composition. Use
it to prioritize antibodies for deeper biophysical analysis, not as a final CMC
decision rule.
