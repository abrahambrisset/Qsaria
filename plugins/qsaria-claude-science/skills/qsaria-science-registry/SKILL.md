---
name: qsaria-science-registry
description: Inspect and govern Qsaria catalog models, registration, durable persistence, recommendations, and ensembles in a bounded Claude Science Registry mission.
---

# Govern Qsaria Models

Manage model identity and persistence without training, curation, ordinary
prediction, or final reporting.

Keep `trained`, `session-registered`, `catalogued`, and `decision-ready`
distinct. Preserve returned catalog status and never upgrade scientific gates.
For pending persistence use exactly one returned route: a non-empty candidate
manifest goes to batch registration; one recommended payload goes through
single registration then persistence; missing evidence is a blocker. Never
send an empty candidate list to the batch route.

Use synchronous catalog reads for short consultation. Run persistence,
registration, ensemble creation, or ensemble evaluation through
`qsaria_registry_start_operation`. Poll state every 20 seconds and fetch the
result only when terminal. Preserve canonical model ids and require confirmed
storage below `data/model_assets/internal`. Retain `.files` evidence.

Inspect ensemble candidates before creation. Ensemble creation alone does not
prove improvement; evaluate only when explicitly requested.

Build one fresh handoff with `agent="qsaria_registry"` and
`execution_mode="project_agent"`. Record it exactly once and submit the exact
normalized envelope using `host.submit_output`.
