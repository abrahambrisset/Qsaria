# Prediction Architecture

## Goal

QSAR prediction and training are backend-neutral at the agent boundary. Agents
talk to public toolkits that own one responsibility each; backend-specific
toolkits and backend adapters sit behind those public facades.

## Layering

```text
Agents
  |
  |-- DatasetCurationToolkit
  |-- QSARTrainingToolkit
  |     |-- training_orchestration.py
  |     |-- backend_factory.py
  |     |-- tabular_representations.py
  |     |-- MolecularFeatureToolkit
  |     |-- ActivityCliffToolkit
  |     |-- ChempropToolkit  -> ChempropBackend
  |     |-- LightGBMToolkit  -> LightGBMBackend
  |     |-- TabICLToolkit    -> TabICLBackend
  |-- ModelRegistryToolkit
  |-- PredictionInferenceToolkit
  |-- BenchmarkToolkit
  |-- EnsembleToolkit
  |     |-- EnsembleBackend
  |-- qsar_report_agent
```

`ChempropToolkit`, `LightGBMToolkit`, and `TabICLToolkit` are backend-internal
training toolkits. They are not the registry, inference, or agent-facing
training facade.

## Public Toolkits

- `QSARTrainingToolkit`: single agent-facing entry point for training workflows,
  including `train_qsar_model`, `train_chemprop_model`, `train_lightgbm_model`,
  and `train_tabicl_model`. Its Python-only `prepare_training_dataset` method is
  retained for explicit diagnostic exports but is not registered as an agent tool.
- `ModelRegistryToolkit`: catalog and session registry operations, including
  model registration, persistence, model summaries, catalog recommendations,
  and backend capability descriptions.
- `PredictionInferenceToolkit`: batch and direct inference for registered or
  catalog models.
- `BenchmarkToolkit`: explicit benchmark campaigns only. It compares candidates
  by delegating actual training to `QSARTrainingToolkit` and persistence to
  `ModelRegistryToolkit`.
- `EnsembleToolkit`: ensemble creation, summary, and evaluation workflows.

## Tabular QSAR Representations

Tabular representations are declared once in `tabular_representations.py` and
consumed by training, benchmark, and backend capability reporting.

Modern automatic pack:

- `morgan_only`
- `rdkit_all`
- `morgan_count_only`

Explicit advanced representations:

- `morgan_binary_count_rdkit_all`
- `morgan_rdkit_all`

`MolecularFeatureToolkit` owns feature generation for all tabular backends.
Features are cached by dataset fingerprint, SMILES column, representation, and
feature parameters. Combined representations reuse the cached RDKit all, Morgan
binary, and Morgan count feature tables rather than regenerating them per split
or candidate.

## Training Contracts

- `standard_qsar` runs random, scaffold and cluster holdouts with a target ratio
  of 80/10/10. Each grouped holdout needs at least three groups and keeps each
  group intact. Every classification fit must retain all classes. Use
  `single_random_holdout` for one explicit random 80/10/10 split.
- Random classification holdouts and repeated cross-validation use the labels
  for stratification. Very small classes are prioritized in training. A test
  or validation set can still lack a rare class when the counts are too low.
- For LightGBM, the standard contract uses `rdkit_all` and eligible post-selection
  outlier analysis. Set `tuning.enabled=true` to request Optuna/TPE and optionally
  set `tuning.n_trials` to choose the trial budget.
- Advanced validation uses explicit `validation_strategy` objects for holdout,
  repeated holdout, cross-validation, full-train, scaffold, and cluster flows.
- `local_light`, `local_standard`, `heavy_validation`, and `benchmark` are
  compute profiles. They control resources and never select validation.
- Benchmark is a workflow kind, not a protocol. It defaults to
  `standard_qsar` and propagates one explicit strategy to every candidate.

The molecular overlap audit compares each labelled holdout with its fit rows.
It counts exact standardized identities and nonidentical molecules with a
radius-two count-Morgan Tanimoto similarity of at least 0.95 or 0.99. It also
reports the median and maximum nearest-neighbour similarity and how many rows
could be audited. The registry keeps the fit-structure reference so external
evaluations can repeat the audit, including for each ensemble component. Older
models may have only a source dataset that includes holdout rows. The report
identifies that broader reference or marks the audit unavailable.

## Backend Construction

`backend_factory.py` centralizes default backend creation. Registry and
inference facades receive the same backend mapping so catalog records,
registered session models, and prediction calls agree on backend identity.

## Session State Contract

The dedicated prediction state lives under:

```python
session_state["prediction_models"] = {
    "catalog_recommendations": {
        "selected_model": {...},
        "alternatives": [...],
        "selection_summary": "...",
    },
    "registered": {
        "<model_id>": {
            "model_id": "...",
            "backend_name": "lightgbm",
            "model_path": "...",
            "status": "validated",
            "known_metrics": {...},
            "task": {
                "task_type": "regression",
                "smiles_columns": ["smiles"],
                "target_columns": ["pEC50"],
            },
            "tags": {...},
        }
    },
    "last_prediction": {...},
    "prediction_history": [...],
    "training_runs": [...],
}
```

## Persistent Model Catalog

The persistent catalog lives in:

```text
src/cs_copilot/tools/prediction/model_catalog.json
```

Each model entry can include:

- runtime identity: `model_id`, `display_name`, `backend_name`, `model_path`
- governance: `version`, `status`, `owner`, `source`
- scientific fit: `domain_summary`, `recommended_for`, `not_recommended_for`
- quality signals: `known_metrics`, `training_data_summary`
- operational hints: `inference_profile`, `selection_hints`
- user-facing caveats: `strengths`, `limitations`

## Canonical Data Contracts

### Input

- Preferred molecule column: `smiles`
- Optional identifier column: `compound_id`
- Optional split column for training flows: `split`
- Optional task targets: one or many endpoint columns

### Output

- Input columns preserved whenever possible
- Backend-specific prediction columns plus a canonical `prediction` when
  available
- Metadata tracked separately in session state:
  - `model_id`
  - `backend_name`
  - `task_type`
  - `preds_path`
  - `return_uncertainty`

## Non-Goals

- No registry or inference routing through `ChempropToolkit`.
- No implicit benchmark launch for ordinary `standard_qsar` or custom
  `validation_strategy` training requests.
- No hidden training workflow guessed from user text without explicit dataset,
  target, task, and protocol metadata.
