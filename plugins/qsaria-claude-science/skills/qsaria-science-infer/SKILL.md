---
name: qsaria-science-infer
description: Run governed prediction or labelled external evaluation with one exact persisted Qsaria model id in a bounded Claude Science Inference mission.
---

# Run Qsaria Inference

Predict or evaluate without training, curation, catalog selection, policy
changes, or final reporting. Require the exact coordinator-supplied `model_id`;
never choose a fallback.

Resolve an uploaded CSV with `host.artifact_path(version_id)` and preserve the
exact path. Run inference or evaluation through
`qsaria_inference_start_operation`; poll state every 20 seconds and obtain the
result only after a terminal state. Never start duplicate inference after a
client timeout.

Preserve output rows, units, model identity, ensemble components,
disagreement, warnings, artifacts, and applicability-domain labels. A request
to evaluate, validate, test, or score labelled data is external evaluation.
Missing targets are terminal and never fall back to prediction. Preserve
`blocked_failed_external_evaluation` exactly in facts and return public
`terminal_failure`; stop all further inference.

Build one fresh handoff with `agent="qsaria_inference"` and
`execution_mode="project_agent"`. Record it exactly once and submit the exact
normalized envelope using `host.submit_output`.
