---
name: qsaria-train
description: Train, validate, benchmark, and assess activity cliffs for Qsaria QSAR models from an approved curated dataset. Use only inside the qsaria_training project agent for a bounded training mission delegated by the coordinator.
---

# Train Qsaria Models

Run reproducible training from an explicit curated data contract. Do not search for another dataset, perform inference, or write the final report.

Read [training-policy.md](references/training-policy.md), [validation-strategies.md](references/validation-strategies.md), [terminal-failures.md](references/terminal-failures.md), [scientific-invariants.md](../../contracts/scientific-invariants.md), and [handoff.schema.json](../../contracts/handoff.schema.json) before using tools.

## Workflow

1. Open the experiment and require the exact curated dataset artifact, curated SMILES column, target columns, and task type.
2. Describe the training environment when compute or backend selection matters. Describe hyperparameters before proposing non-default values.
3. Build one strict `request` object with `schema_version="2.0"`. Preserve every explicit backend, representation, validation, tuning, outlier, Activity Cliff, applicability-domain, and compute constraint. Never add an undeclared key.
4. Apply the standard LightGBM contract from the training policy whenever the coordinator delegates that workflow. Never interpret presentation words such as `simple` as permission to remove scientific stages.
5. Call the public Qsaria training facade directly with `train_csv` plus the typed `request`. Do not create split columns, feature files, caches, artifact paths, or registry payloads by hand. Runtime paths and hardware details are server-managed.
6. Benchmark only when the user explicitly requested a multi-backend or multi-representation comparison.
7. Use Activity Cliff operations only as part of the requested training analysis; it is a capability, not another agent.
8. Verify returned metrics, split provenance, model files, applicability-domain evidence, smoke-test evidence, warnings, gates, and effective workflow settings from structured results.
9. Stop before persistence when the effective workflow contradicts the delegated contract. Do not relabel a baseline or simplified run as standard.
10. Preserve every returned candidate manifest, recommended registry payload, reporting handoff, and persistence plan exactly. If Training or Benchmark already confirms persistence, report its canonical identifiers. Otherwise recommend a separate Registry mission; never call Registry tools yourself.

For an ordinary standard LightGBM mission, the essential request shape is:

```json
{
  "schema_version": "2.0",
  "smiles_column": "smiles",
  "target_columns": ["pEC50"],
  "task_type": "regression",
  "representation": {"kind": "generated", "name": "rdkit_all"},
  "validation": {"kind": "standard_qsar"},
  "tuning": {"enabled": true, "n_trials": 50},
  "outlier_analysis": {"enabled": true},
  "backend": {"name": "lightgbm"}
}
```

Omitted nested sections use the published typed defaults. Never send `workflow`, split payloads, feature-cache paths, checkpoint paths, heartbeat controls, persistence controls, or any free-form argument map.

Custom tuning intervals belong only in `request.tuning.search_space`, using the
declared typed integer or float distribution for an officially tunable field.
Do not emulate a distribution with backend fields or undeclared keys.

Put an explicit optimization metric only in `request.tuning.objective`. Use a
canonical metric name or an unambiguous supported spelling; the contract
normalizes case, separators, long forms, and `r²`. Never use bare `auc`, invent
a metric, or put an optimization metric under `request.validation`.

Never convert a failed training call into a different backend, split, seed, validation protocol, or method. Follow the terminal-failure reference.

## Return the handoff

After scientific execution begins, build a fresh public envelope with `schema_version="1.0"`, `execution_mode="project_agent"`, and every required field from the shared schema. Never pass through a toolkit-returned `handoff` or `reporting_handoff` object; copy only its verified scientific facts into the new envelope. Set `agent=qsaria_training`, preserve exact artifact and model identifiers, metrics and gate facts, `requested_workflow`, `effective_workflow`, `contract_conformance`, and the recommended next action. Include the exact candidate manifest or registry payload reference when Registry must persist the result. Copy returned `reporting_handoff.report_facts` and `reporting_handoff.report_tables` into the handoff facts. Call `qsaria_record_handoff` exactly once, immediately before returning its normalized envelope to the coordinator. For a strict pre-execution request-schema rejection, return only the validation diagnostic and do not record a scientific handoff.
