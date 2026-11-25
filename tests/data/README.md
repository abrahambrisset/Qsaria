# Reference test data

## WARNING - SYNTHETIC PIPELINE FIXTURES, NOT EXPERIMENTAL DATA

**The target columns in `reference_regression.csv` and `reference_classification.csv`
are COMPUTED RDKit DESCRIPTORS. They are NOT measured properties. Nothing in these
files was observed in a laboratory.**

- `clogp` is `rdkit.Chem.Crippen.MolLogP` - the Wildman-Crippen atom-contribution
  estimate of logP, not a measured partition coefficient.
- `tpsa_high` is `1` when `rdkit.Chem.Descriptors.TPSA` is strictly above
  **60.0 A^2** and `0` otherwise. TPSA is Ertl's additive polar-surface-area sum,
  not a measurement either.

The structures are real and valid; the labels are a deterministic function of those
structures. **A model trained here is only re-learning an RDKit formula.** A high R^2
means the featuriser can reproduce an additive atom-contribution sum - it says nothing
about predicting any real-world endpoint.

**These files exist solely to exercise the QSAR pipeline end to end in CI and in the
eval harness** (featurisation, splitting, training, metrics, reporting), because the
split builder rejects datasets smaller than 10 rows and nothing else committed to this
repository is large enough to train on.

**They MUST NEVER be used to support a scientific claim about model quality, method
comparison, prospective performance, or chemical series behaviour**, and they must
never be presented to a user as experimental data.

## Contents

| File | Rows | Target | Type |
| --- | --- | --- | --- |
| `reference_regression.csv` | 400 | `clogp` | continuous |
| `reference_classification.csv` | 400 | `tpsa_high` | binary, 164 / 236 (41% positive) |

Both files share `compound_id`, `smiles` (RDKit canonical) and `scaffold`
(Bemis-Murcko, for scaffold-split tests). 162 distinct scaffolds, no duplicate
canonical SMILES.

## Regenerating

```bash
PYTHONPATH=src python -m cs_copilot.tools.benchmarks.reference
```

Generation is fully deterministic - no RNG, no seed, structures enumerated
combinatorially from fixed fragment lists in
`src/cs_copilot/tools/benchmarks/reference.py` and ordered by a SHA-256 digest of the
canonical SMILES. Regenerating must produce a byte-identical file;
`tests/unit/test_benchmark_reference.py` asserts exactly that.

The one external coupling is the RDKit version, since `MolLogP` and `TPSA` are RDKit
implementations. If the reproducibility test fails after a dependency bump, regenerate
and commit the diff.
