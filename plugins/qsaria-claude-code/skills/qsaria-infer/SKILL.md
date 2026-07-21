---
name: qsaria-infer
description: Run governed prediction or labelled external evaluation with one exact persisted Qsaria model id in a bounded Claude Code inference mission.
user-invocable: false
---

# Run Qsaria Inference

Predict or evaluate without training, curation, catalog selection, policy
changes, or final reporting.

## Selection and prediction

Open the supplied experiment when applicable. Require the exact `model_id`
resolved by the coordinator. If absent or ambiguous, return
`needs_user_input`; never inspect the catalog or choose a fallback.

Use direct SMILES prediction for small explicit lists and CSV prediction for
batches. Preserve output rows, units, model identity, artifact ids, ensemble
components, disagreement, warnings, and applicability-domain labels exactly:
`in_domain`, `edge_of_domain`, and `out_of_domain`.

## External evaluation

Treat a labelled request to evaluate, test, score, or validate as external
evaluation. Use the evaluation facade and preserve returned predictions,
metrics, report, and catalog evidence. Missing targets make evaluation
terminal; never fall back to ordinary prediction.

If any result has `status=blocked_failed_external_evaluation`, stop all further
inference immediately. Do not retry prediction and do not export a normal
summary. Preserve that exact toolkit status in facts and blockers while the
public handoff uses `status=terminal_failure`.

## Return one public handoff

Build a fresh complete envelope with `agent="qsaria_inference"`,
`execution_mode="project_agent"`, the exact model id, evaluation or prediction
facts, canonical artifacts, warnings, blockers, and next action. Never submit a
toolkit handoff directly.

Call `qsaria_record_handoff` exactly once immediately before returning and
return its normalized envelope.
