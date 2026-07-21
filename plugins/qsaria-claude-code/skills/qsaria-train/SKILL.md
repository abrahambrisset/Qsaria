---
name: qsaria-train
description: Train, validate, benchmark, and assess activity cliffs for Qsaria models from an approved curated dataset in a bounded Claude Code training mission.
user-invocable: false
---

# Train Qsaria Models

Run reproducible training from an explicit curated data contract. Do not find a
different dataset, perform inference, call Registry, or write the final report.

## Inputs and protocol

Require the exact experiment, curated artifact id, curated SMILES column,
target columns, task type, and user constraints. Describe the environment when
compute or backend selection matters and describe hyperparameters before
proposing non-default values.

Preserve every explicit backend, representation, validation, tuning, outlier,
and compute constraint. Never translate cross-validation into repeated
holdout, invent split columns, create feature files, or reconstruct registry
payloads.

Build one strict `request` with `schema_version="2.0"`; fixed backend values
belong under `backend`, validation under `validation`, and tuning under
`tuning`. Unknown fields are invalid. Never send runtime paths, split payloads,
checkpoint/device/offload controls, heartbeat controls, persistence controls,
or a free-form argument map.

Custom tuning intervals belong only in the backend-matched typed
`request.tuning.search_space`, and every customized field must also appear in
`request.tuning.parameters`.

An unqualified standard LightGBM workflow means:

- representation `{"kind":"generated","name":"rdkit_all"}`;
- validation `{"kind":"standard_qsar"}`, the toolkit's fixed random 80/10/10 split;
- toolkit-default hyperparameter tuning with 50 requested trials when eligible;
- eligible outlier analysis;
- baseline and retained outlier-filtered evidence kept distinct.

Words such as `simple` describe presentation, not permission to remove these
stages. Use another canonical typed `validation` object only when the user requests another
split family, ratios, repeated holdout, cross-validation, isolated external
test, or full-train. Full-train is trained but not internally evaluated.

## Execute and verify

Call the public training facade directly with `train_csv` and the typed
`request`. Benchmark only when the user requests
multiple backends or representations. Use Activity Cliff operations only as
part of the delegated training analysis.

Verify metrics, split provenance, model files, applicability-domain evidence,
smoke-test evidence, gates, warnings, effective settings, and terminal status.
Preserve the canonical gates: Dataset, Hardest Split, Robustness Gap, and
Random Stability. Never compute missing metrics manually.

Stop before persistence when the effective workflow contradicts the delegated
contract. Never change backend, split, seed, validation, representation, or
method after failure.

Preserve candidate manifests, recommended registry payloads,
`report_facts`, `report_tables`, and persistence plans exactly. A checkpoint is
not a catalog model. When persistence is not confirmed, recommend a separate
Registry mission using the exact returned manifest or payload.

## Return one public handoff

Build a fresh envelope with `agent="qsaria_training"` and
`execution_mode="project_agent"`. Include requested workflow, effective
workflow, contract conformance, metrics, gates, persistence evidence, artifact
ids, model ids, warnings, blockers, and next action. Copy reporting facts and
tables exactly when returned; never submit a toolkit handoff directly.

After scientific execution begins, call `qsaria_record_handoff` exactly once
immediately before returning and return its normalized envelope. For a strict
pre-execution request-schema rejection, return only the validation diagnostic
and do not record a scientific handoff.
