# Inference and external-evaluation policy

## Selection

The coordinator resolves catalog policy and supplies one exact `model_id`, using task, target, units, validation, and applicability-domain evidence. Use that identifier unchanged. If it is absent or ambiguous, stop with `needs_user_input` or return control to the coordinator; do not inspect the catalog or choose a fallback model.

## Prediction

- Use direct-SMILES prediction for small explicit lists.
- Use CSV prediction for batches.
- Preserve toolkit preview rows and full-output artifacts.
- Preserve applicability-domain labels exactly: `in_domain`, `edge_of_domain`, `out_of_domain`.
- If translated to reliability, use only `Elevee`, `Moderee`, and `Faible`, respectively.
- Preserve ensemble components, aggregation, disagreement, and limitations when returned.

## Labelled external evaluation

An evaluation request on a labelled file uses the evaluation facade. Missing target columns make metrics impossible and terminate that request. Do not fall back to prediction.

After successful external evaluation, the returned predictions, metrics, report, and append-only catalog entry are sufficient. Do not export a normal prediction summary. If the status is `blocked_failed_external_evaluation`, stop all inference tool use and preserve that exact terminal result inside a common `status=terminal_failure` handoff. This prohibits further scientific execution; it does not prohibit the coordinator from dispatching the experiment's one evidence-only Report mission afterward when required.
