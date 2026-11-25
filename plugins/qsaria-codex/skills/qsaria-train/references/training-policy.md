# Training policy

## Input contract

Require the curated dataset path, `smiles_column_curated` (normally `smiles`), target column list, and task type. Do not reuse an original SMILES column on a curated file.

The Qsaria MCP profile intentionally does not expose `prepare_training_dataset`. Pass an approved curated dataset directly to training; do not add a rename or export step inside the agent.

## Public entry points

- Use the unified training facade or its explicit Chemprop, LightGBM, or TabICL backend facade.
- Backend tools prepare their own features and caches. Do not precompute Morgan or RDKit data by hand.
- The SMILES column is identity/provenance, never a numeric tabular feature.
- Describe declared hyperparameters before setting custom values. Put fixed values in `request.backend` and tuning controls in `request.tuning`; unknown keys are invalid.
- Put custom tuning intervals only in the backend-matched typed `request.tuning.search_space`; each field must also be selected in `request.tuning.parameters`.
- Inspect local compute and respect the selected compute profile independently of validation strategy.

## Standard LightGBM workflow

Treat an unqualified request for a standard Qsaria LightGBM model as one exact
scientific contract:

- use representation `rdkit_all`;
- use `representation={"kind":"generated","name":"rdkit_all"}`;
- use protocol `standard_qsar`;
- use `validation={"kind":"standard_qsar"}`;
- use the default disabled hyperparameter tuning;
- preserve the toolkit's default outlier analysis whenever the task is eligible.

`standard_qsar` runs random, scaffold and cluster holdouts with a target ratio of
80/10/10. Use `single_random_holdout` when the user explicitly requests a single
random 80/10/10 split. `training_profile` describes compute policy.

Use `tuning.enabled=true` only when the user requests optimization. Preserve an explicit trial count in `tuning.n_trials`. Keep eligible outlier analysis by default. After execution, verify the effective representation, validation protocol, tuning status and outlier-analysis status from structured results. Stop before persistence when
those facts contradict the delegated contract.

## Benchmark and activity cliffs

Benchmark only after an explicit request for a benchmark, head-to-head comparison, multi-backend evaluation, or representation leaderboard. Build one strict `QsariaBenchmarkRequest`; use `validation={"kind":"standard_qsar"}` by default. When the user requests an advanced strategy, put the same canonical validation object on the whole benchmark. For benchmarks, set `tuning_enabled=true` only when the user asks to tune candidates. Route a standard single-model request to Training.

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
