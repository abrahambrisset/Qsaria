---
name: qsaria-registry
description: Inspect the Qsaria model catalog, govern registration and persistence, recommend existing models, and create or evaluate persisted-model ensembles. Use only inside the qsaria_registry project agent for a bounded catalog or model-governance mission.
---

# Govern Qsaria Models

Manage model identity and persistence without training models, curating data, running ordinary predictions, or drafting a final report.

Read [model-lifecycle.md](references/model-lifecycle.md), [scientific-invariants.md](../../contracts/scientific-invariants.md), and [handoff.schema.json](../../contracts/handoff.schema.json) before using tools.

## Workflow

1. Require and open the experiment supplied by the coordinator. Standalone read-only catalog consultations are handled directly by the coordinator and are not delegated.
2. For lookup, list and summarize exact catalog records before recommending a model. Preserve task, target, units, status, validation, applicability domain, backend, and representation constraints.
3. Prefer `robust_validated`, then `validated`. Never present `workflow_demo` or `experimental` as routine-ready unless the user explicitly accepts the limitation.
4. For a newly trained model, use only exact returned model paths and registry payloads. Session registration is not persistence. When the experiment reports `persistence.status=pending`, complete the exact required route and materialize the reusable model under the returned `data/model_assets/internal` catalog root; retain `.files` as session evidence rather than moving or deleting it.
5. Select the persistence route from the model-lifecycle table. Use the batch tool only with a verified non-empty candidate manifest. For one exact recommended payload, call `register_model` and then `persist_registered_model`.
6. Treat a model as persisted only when the persistence operation confirms it and returns its canonical model identifier and storage root.
7. Inspect ensemble candidates before creation. Create an ensemble only from exact catalog identifiers. Evaluate it only when evaluation was explicitly requested.
8. Verify catalog and ensemble artifacts before reporting their identifiers.

## Return the handoff

Build the common envelope with `agent=qsaria_registry`, canonical identifiers, governance status, evidence, blockers, and exact artifacts. Call `qsaria_record_handoff` exactly once, immediately before returning its normalized envelope to the coordinator.
