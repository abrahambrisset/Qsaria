---
name: qsaria-science-train
description: Train, validate, benchmark, and assess activity cliffs for Qsaria models from an approved curated dataset in a bounded Claude Science Training mission.
---

# Train Qsaria Models

Train from the exact curated data contract. Do not substitute a dataset, run
inference, call Registry, or write the final report.

Require the experiment, curated artifact, SMILES column, targets, task type,
and explicit constraints. Build one strict request with
`schema_version="2.0"`; backend, validation, representation, and tuning belong
only in their typed fields. Never send runtime paths, heartbeat controls,
persistence controls, or a free-form argument map.

Put an explicit optimization metric only in `request.tuning.objective`. Use a
canonical metric name or an unambiguous supported spelling; the contract
normalizes case, separators, long forms, and `r²`. Reject bare `auc`, unknown
metrics, incompatible backend/task combinations, and every optimization metric
placed under `request.validation`.

An unqualified standard LightGBM workflow means generated `rdkit_all`,
`standard_qsar`, toolkit-default tuning with 50 requested trials when eligible,
and eligible outlier analysis. Presentation words such as “simple” never
disable those stages.

Use synchronous describe tools for short reads. Launch training, benchmark,
or Activity Cliff writes through `qsaria_training_start_operation` with the
exact underlying tool and arguments. Poll state every 20 seconds; obtain the
result only when terminal. Never launch duplicate work after a timeout or
change backend, representation, split, seed, validation, or method after an
error.

Verify metrics, split provenance, model files, applicability domain, smoke
tests, gates, effective settings, candidate manifests, persistence plans,
`report_facts`, and `report_tables`. A checkpoint is not a catalog model.

Build one fresh handoff with `agent="qsaria_training"` and
`execution_mode="project_agent"`. After execution starts, record it exactly
once and submit the normalized envelope with `host.submit_output`. A strict
pre-execution schema rejection returns only its diagnostic and no scientific
handoff.
