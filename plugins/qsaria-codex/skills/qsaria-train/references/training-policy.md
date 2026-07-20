# Training policy

## Input contract

Require the curated dataset path, `smiles_column_curated` (normally `smiles`), target column list, and task type. Do not reuse an original SMILES column on a curated file.

The Qsaria MCP profile intentionally does not expose `prepare_training_dataset`. Pass an approved curated dataset directly to training; do not add a rename or export step inside the agent.

## Public entry points

- Use the unified training facade or its explicit Chemprop, LightGBM, or TabICL backend facade.
- Backend tools prepare their own features and caches. Do not precompute Morgan or RDKit data by hand.
- The SMILES column is identity/provenance, never a numeric tabular feature.
- Describe declared hyperparameters before setting custom values. Pass only supported parameters.
- Inspect local compute and respect the selected compute profile independently of validation strategy.

## Standard LightGBM workflow

Treat an unqualified request for a standard Qsaria LightGBM model as one exact
scientific contract:

- use `representation_name="rdkit_all"`;
- use `validation_protocol="standard_qsar"` and omit `validation_strategy`;
- preserve the toolkit's default hyperparameter tuning, currently 50 requested trials;
- preserve the toolkit's default outlier analysis whenever the task is eligible.

`standard_qsar` names only the fixed random 80/10/10 validation protocol. The
standard LightGBM workflow is the complete contract above. `training_profile`
describes compute policy and does not prove that tuning or outlier analysis ran.

Words about presentation or interaction such as `simple`, `simple prompt`, or
`quick explanation` never disable scientific stages. Pass
`hyperparameter_tuning.enabled=false` or `outlier_analysis.enabled=false` only
when the user explicitly requests that scientific change. Label such a run
`baseline` or `simplified`, never `standard`.

Prefer omitting the tuning and outlier objects for the standard workflow so the
current Qsaria toolkit remains the source of truth. After execution, verify the
effective representation, validation protocol, requested tuning trials, and
outlier-analysis status from structured results. Stop before persistence when
those facts contradict the delegated contract.

## Benchmark and activity cliffs

Benchmark only after an explicit request for a benchmark, head-to-head comparison, multi-backend evaluation, or representation leaderboard. Use a `benchmark_*` mode and set the explicit-request guard. A normal standard or robust training request is not a benchmark.

Activity Cliff analysis is attached to model evidence. Preserve its exact indexes, split labels, artifacts, applicability-domain context, and warnings. Never invent validation Activity Cliff metrics when prediction artifacts were not exported.

## Persistence

- Training and Benchmark may return a candidate manifest, concrete model artifact, recommended registry payload, and persistence plan. Preserve those values exactly; never reconstruct compacted feature lists or payload fields.
- When the training result itself confirms persistence, use its canonical model identifiers and persistence evidence as returned.
- Otherwise, do not call Registry tools. Put the exact manifest path or payload reference in the structured handoff and recommend a bounded Registry mission.
- A saved training checkpoint is not automatically a catalogued model. Never claim catalog persistence without explicit persistence evidence.

## Training handoff facts

Include `requested_workflow`, `effective_workflow`, and `contract_conformance`
in the handoff facts. For the standard LightGBM workflow, derive the effective
facts from the structured training result and its reporting handoff; never infer
them from `training_profile`. Preserve `reporting_handoff.report_facts` and
`reporting_handoff.report_tables` in the handoff facts when returned.
