# Qsaria V1 on corrected MoleculeACE

This follow-up uses ABL1 binding affinity (`CHEMBL1862_Ki`) and farnesoid X receptor functional potency
(`CHEMBL2047_EC50`). The corrected release revised cliff-pair annotations and
the train/test assignments after the original study. Both datasets are for human
targets and come from MoleculeACE v3.0.0. [Release notes](https://github.com/molML/MoleculeACE/releases/tag/v3.0.0) · [Original article](https://doi.org/10.1021/acs.jcim.2c01073) · [Split correction](https://doi.org/10.1021/acs.jcim.3c01576)

The protocol uses the supplied `split`, `cliff_mol`, and `y` columns. For
these files, `y` equals −log₁₀ of the concentration in nM. The same additive
shift to the observed and predicted values leaves RMSE unchanged. The Qsaria
standardizer precedes binary Morgan fingerprints of radius 2 and 2,048 bits.
The Qsaria LightGBM backend and scikit-learn Random Forest each train on all
usable official training rows with 80 trees, one CPU worker, and seeds 42, 1701,
and 2026. The naïve reference predicts the official training mean. Model
settings are fixed in advance and training takes no test labels or cliff flags
as input. [Frozen settings](qsaria-v1-moleculeace.toml) · [Runner](../scripts/benchmark_qsaria_v1_moleculeace.py)

RMSE covers all scored official test rows. RMSE_cliff covers the scored test
rows marked `cliff_mol=1` by MoleculeACE. A sensitivity analysis reports both
metrics after excluding test rows whose Qsaria standardized identity occurs in
training. The report includes official and scored row counts, invalid-structure
counts, raw exact overlaps and the Qsaria identity audit. The latter ignores
stereochemistry and also reports similarity using sparse-count Morgan
Tanimoto at 0.95 and 0.99. [Results](results/qsaria-v1/moleculeace.json)

The MoleculeACE authors release their software under MIT. Their benchmark
data derive from ChEMBL v29, whose database is available under CC BY-SA 3.0.
The repository tracks aggregate results and downloads benchmark tables
directly from MoleculeACE. [Pinned MoleculeACE license](https://raw.githubusercontent.com/molML/MoleculeACE/230e060c35c8abf999e13c3c8122c229950585ce/LICENCE) · [ChEMBL licensing](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions)

## Replay

Run from the Qsaria repository root with its locked `.venv`. The runner
checks both data digests and the Qsaria source files against the pinned Git
revision. Choose a fresh local directory for each run.

```bash
ace_commit=230e060c35c8abf999e13c3c8122c229950585ce
ace_data=.files/publication/datasets/moleculeace-v3
mkdir -p "$ace_data" .files/publication/{mpl,tmp,xdg-cache}
for endpoint in CHEMBL1862_Ki CHEMBL2047_EC50; do
  if [ ! -f "$ace_data/$endpoint.csv" ]; then
    curl --fail --location --silent --show-error \
      "https://raw.githubusercontent.com/molML/MoleculeACE/$ace_commit/MoleculeACE/Data/benchmark_data/$endpoint.csv" \
      --output "$ace_data/$endpoint.csv"
  fi
done

export MPLCONFIGDIR="$PWD/.files/publication/mpl"
export TMPDIR="$PWD/.files/publication/tmp"
export XDG_CACHE_HOME="$PWD/.files/publication/xdg-cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# On macOS, use the libomp bundled with the locked PyTorch wheel if needed.
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/torch/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

ace_output="$(mktemp -d "$PWD/.files/publication/moleculeace-replay.XXXXXX")/run"
.venv/bin/python -m scripts.benchmark_qsaria_v1_moleculeace \
  --config benchmarks/qsaria-v1-moleculeace.toml \
  --datasets "$ace_data" \
  --output "$ace_output"
```

The published and replayed reports can be checked independently of wall
times. This exact replay check uses the same package versions and runtime.

```bash
.venv/bin/python - "$ace_output/moleculeace.json" <<'PY'
import json
import sys
from pathlib import Path

original = Path("benchmarks/results/qsaria-v1/moleculeace.json")
replay = Path(sys.argv[1])
reference, repeated = (json.loads(path.read_text()) for path in (original, replay))
assert reference["manifest"] == repeated["manifest"]
assert reference["datasets"].keys() == repeated["datasets"].keys()
for endpoint, row in reference["datasets"].items():
    for field in row.keys() - {"fit_predict_seconds_by_seed"}:
        assert row[field] == repeated["datasets"][endpoint][field], (endpoint, field)
print("MoleculeACE predictions, counts, audits and metrics reproduced.")
PY
```
