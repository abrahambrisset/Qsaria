# Model lifecycle and governance

## States

- **trained**: a training result exists.
- **session-registered**: the runtime knows the concrete model artifact.
- **catalogued**: persistence confirmed a canonical model id and internal model root.
- **decision-ready**: scientific gates support routine selection.

Canonical catalog statuses are `experimental`, `workflow_demo`, `validated`, and `robust_validated`.

- `validated` requires the applicable dataset, hardest-split, and robustness-gap gates.
- `robust_validated` additionally requires a repeated/fold protocol and the random-stability gate.
- Incomplete gates and full-train models remain `workflow_demo`; full-train metrics are `not_evaluated`.

## Persistence rules

Use the concrete trained model artifact, exact returned payload, and exact candidate manifest. Preserve training, split, plot, applicability-domain, and Activity Cliff artifacts. Existing stronger models affect recommendations, not whether a completed run can be retained as a workflow demo.

New experiments default to durable catalog persistence. A publishable Training result therefore creates a `persistence.status=pending` barrier until Registry completes the required route. Only an explicit user request encoded as `persistence_policy=session_only` skips this barrier. Persistence copies durable assets into `data/model_assets/internal`; it does not delete the experiment evidence under `.files`.

Select the persistence route deterministically:

| Verified Training evidence | Required route |
| --- | --- |
| Non-empty `candidate_manifest_path` | Call `register_and_persist_candidates` with that exact manifest. |
| No manifest and one exact `recommended_registry_payload` | Call `register_model`, then `persist_registered_model`. |
| Neither form is available | Stop with a blocker; never reconstruct a payload. |
| Candidate list is empty | Never call the batch persistence tool. |

A missing candidate manifest is normal for a single candidate. Outlier analysis
may legitimately produce no retained filtered variant, so route from the
returned evidence rather than assuming that every standard workflow is a batch.

## Ensembles

Use canonical catalog ids only. Inspection precedes creation. Creation does not imply evaluation. A new ensemble is `workflow_demo` until its own explicit ensemble-level evaluation exists.
