---
name: qsaria-infer
description: Use an exact persisted Qsaria model id to predict from SMILES or CSV files, evaluate labelled datasets, and preserve applicability-domain evidence. Use only inside the qsaria_inference project agent for a bounded inference or external-evaluation mission.
---

# Run Qsaria Inference

Execute governed prediction or evaluation without training, curating, changing catalog policy, or drafting the final report.

Read [inference-policy.md](references/inference-policy.md), [scientific-invariants.md](../../contracts/scientific-invariants.md), and [handoff.schema.json](../../contracts/handoff.schema.json) before using tools.

## Workflow

1. Open the provided experiment when applicable.
2. Require the exact `model_id` selected by the user or resolved by the coordinator. If it is missing, return `needs_user_input` or request coordinator routing; do not inspect the catalog or choose a model yourself.
3. Use direct SMILES prediction for small explicit lists and CSV prediction for batch inputs.
4. Treat a labelled-dataset request to evaluate, test, score, or validate as external evaluation, not ordinary prediction.
5. Preserve returned prediction rows, units, model identity, applicability-domain statuses, ensemble disagreement, file references, metrics, and warnings exactly.
6. Never fabricate target columns or metrics. If labelled targets are missing, the evaluation is terminal; do not fall back to unlabelled prediction.
7. If any result has `status=blocked_failed_external_evaluation`, stop all further inference immediately. Do not retry prediction and do not export a summary. Return the common handoff as `status=terminal_failure`, preserving the exact toolkit status in structured facts and blockers. A later evidence-only Report mission, controlled by the coordinator, is not further inference.

## Return the handoff

Build the common envelope with `agent=qsaria_inference`, the selected model, prediction or evaluation facts, canonical artifact identifiers, warnings, and blockers. Call `qsaria_record_handoff` exactly once, immediately before returning its normalized envelope to the coordinator.
