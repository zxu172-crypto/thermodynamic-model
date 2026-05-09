# thermodynamic model

This repository provides one user-facing workflow:

```text
notebooks/thermodynamic_model_prediction.ipynb
```

Open that notebook, put in a PDB file or a folder of PDB files, and run the
cells. The notebook returns:

- `thermodynamic_risk_score`: primary ranking score; higher means higher
  predicted aggregation risk
- `risk_rank`: low, medium, or high
- `thermo_call`: thresholded screening call; `1` means predicted high risk,
  `0` means predicted lower risk
- `DeltaMu_proxy`, `gamma_proxy`, and `DeltaG_star_kT`: the thermodynamic
  quantities behind the score

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
jupyter notebook notebooks/thermodynamic_model_prediction.ipynb
```

The notebook ships with one example PDB:

```text
examples/pdbs/5VH3.pdb
```

To use your own antibody, edit only the first input cell:

```python
INPUT_MODE = "pdb_file"
INPUT_PATH = "/path/to/your_antibody.pdb"
HEAVY_CHAIN = "H"
LIGHT_CHAIN = "L"
```

For a folder:

```python
INPUT_MODE = "pdb_directory"
INPUT_PATH = "/path/to/pdb_folder"
```

For a manifest:

```python
INPUT_MODE = "manifest"
INPUT_PATH = "/path/to/manifest.csv"
```

Manifest columns:

```text
pdb_path,antibody_id,pdb_id,heavy_chain,light_chain
```

Only `pdb_path` is strictly required, but chain IDs are recommended.

## Model Concept

The model treats antibody aggregation as a nucleation-like thermodynamic
screening problem. Structure-derived surface features are mapped into:

\[
\Delta\mu_{\mathrm{proxy}}
\]

and:

\[
\gamma_{\mathrm{proxy}}.
\]

The notebook then evaluates:

\[
\Delta G(n)=\gamma_{\mathrm{proxy}}n^{2/3}-n|\Delta\mu_{\mathrm{proxy}}|
\]

and converts the nucleation barrier into a bounded risk score. The score should
be used primarily for ranking antibodies. The pass/non-pass call is a secondary
thresholded screening decision.

## Reference Result

The internal reliable-30 benchmark for the locked thermodynamic model was:

```text
accuracy = 0.867
TP = 5
TN = 21
FP = 2
FN = 2
balanced accuracy = 0.814
```

This benchmark is included as context, not as a claim that the model is a final
CMC decision rule.

## Limitations

This is a fast screening model, not molecular dynamics, quantum chemistry, or a
replacement for experimental developability assays. Results depend on PDB
quality, chain selection, missing loops, bound antigen, and formulation context.
