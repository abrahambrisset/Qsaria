---
name: qsaria-registry
description: Inspect and govern Qsaria catalog models, registration, durable persistence, recommendations, and ensembles in a bounded Claude Code registry mission.
user-invocable: false
---

# Govern Qsaria Models

Manage model identity and persistence without training, curation, ordinary
prediction, or final reporting. Standalone read-only catalog consultations are
handled directly by the coordinator.

## Model lifecycle

Use these states without conflation:

- `trained`: training returned a model artifact;
- `session-registered`: the runtime knows the artifact;
- `catalogued`: persistence confirmed a canonical model id and internal root;
- `decision-ready`: scientific gates support routine selection.

Preserve catalog statuses `experimental`, `workflow_demo`, `validated`, and
`robust_validated`. Prefer `robust_validated`, then `validated`, for routine
recommendations. Never upgrade status beyond returned gate evidence.

## Persistence

Open the supplied experiment. Use only exact returned artifacts, manifests, and
payloads. When `persistence.status=pending`, select one deterministic route:

| Training evidence | Route |
| --- | --- |
| Non-empty `candidate_manifest_path` | Call `register_and_persist_candidates` with that path. |
| One exact `recommended_registry_payload` | Call `register_model`, then `persist_registered_model`. |
| Neither | Stop with a blocker. |
| Empty candidates | Never call the batch tool. |

Persistence must confirm the canonical model id and
`data/model_assets/internal` root. Retain `.files` as experiment evidence; do
not move or delete it. A missing batch manifest can be normal for one
candidate, and outlier analysis can legitimately retain no filtered candidate.

## Catalog and ensembles

For recommendations, preserve task, target, units, backend, representation,
validation, applicability domain, and status. Inspect ensemble candidates
before creation and use only canonical catalog ids. Ensemble creation does not
prove improvement; evaluate only when explicitly requested.

## Return one public handoff

Build a fresh complete envelope with `agent="qsaria_registry"`,
`execution_mode="project_agent"`, canonical ids, governance status, structured
facts, artifacts, warnings, blockers, and next action. Never submit a toolkit
handoff directly.

Call `qsaria_record_handoff` exactly once immediately before returning and
return its normalized envelope.
