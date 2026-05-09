# thermodynamic model

This repository contains a thermodynamic model for screening antibody
aggregation risk from structure-derived descriptors.

The model is a physically motivated proxy, not a first-principles simulation.
It maps antibody surface information into an effective association drive,
an effective interfacial penalty, a CNT-like nucleation barrier, and a bounded
risk score.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.10+ is recommended.

## Python Usage

```python
import pandas as pd

from abthermo_aggregation.pdb_surface import compute_surface_descriptors
from abthermo_aggregation.scoring import add_batch_scores

descriptor = compute_surface_descriptors(
    "example_fab.pdb",
    heavy_chain="H",
    light_chain="L",
)

scores = add_batch_scores(pd.DataFrame([descriptor]))
print(scores[["thermodynamic_risk_score", "DeltaG_star_kT", "risk_rank"]])
```

For a table of precomputed descriptor features:

```python
import pandas as pd

from abthermo_aggregation.thermodynamic_model import add_thermodynamic_model_scores

features = pd.read_csv("feature_table.csv")
scores = add_thermodynamic_model_scores(features)
```

## Model Form

The thermodynamic model uses a CNT-like barrier:

\[
\Delta G(n)=
\gamma_{\mathrm{proxy}} n^{2/3}
-n|\Delta\mu_{\mathrm{proxy}}|
+\lambda_{\mathrm{elec}} n^{1/3}.
\]

Here \(n\) is the cluster size, \(\Delta\mu_{\mathrm{proxy}}\) is an effective
association drive, and \(\gamma_{\mathrm{proxy}}\) is an effective interfacial
penalty.  The electrostatic term is a fixed proxy correction, not a separate
first-principles free-energy law.

The reported score is a monotonic transform of the barrier:

\[
S=
\frac{1}{1+\exp[(\Delta G^*/k_BT-10)/4]}.
\]

This score should be interpreted as a ranking coordinate: lower barriers give
higher predicted aggregation risk.

## Inputs

The direct structure path uses:

- `pdb_path`
- `heavy_chain`
- `light_chain`

The descriptor-table path expects columns describing hydrophobic exposure,
charge, interface terms, roughness, and CNT-style intermediate quantities.

## Important Limits

The thermodynamic model is a screening tool.  It does not replace experimental
developability assays, molecular dynamics, or formulation-specific CMC studies.
The numerical weights are fixed project-calibrated proxy weights, not measured
thermodynamic constants.
