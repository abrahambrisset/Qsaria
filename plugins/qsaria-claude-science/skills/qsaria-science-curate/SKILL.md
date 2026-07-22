---
name: qsaria-science-curate
description: Inspect, validate, curate, summarize, and document one molecular dataset for a bounded Qsaria Curation mission delegated by the Claude Science coordinator.
---

# Curate Qsaria Data

Prepare one traceable QSAR-ready dataset. Do not train, infer, persist models,
or write the final report.

Open the supplied experiment. Resolve an uploaded file through
`host.artifact_path(version_id)` and pass that exact physical path; never guess
or reconstruct a path. Identify structure and target columns while preserving
user choices. Return `needs_user_input` when ambiguity changes scientific
meaning.

Use synchronous inspection tools for short reads. Run experiment-writing
curation through `qsaria_curation_start_operation`, passing the exact underlying
tool name and arguments without `experiment_id` inside `arguments`. Poll
`qsaria_get_operation_state` every 20 seconds and fetch
`qsaria_get_operation_result` only when terminal. Never start a second
operation while the first remains queued or running.

Structure standardization uses `chembl_structure_v1`. Preserve row counts,
exclusions, standardization failures, normalized SMILES, targets, units,
duplicate handling, warnings, blockers, and all artifact ids. Never relax
validation or remove rows silently.

Build one fresh handoff with `agent="qsaria_curation"` and
`execution_mode="project_agent"`. Call `qsaria_record_handoff` exactly once,
then submit that exact normalized envelope through `host.submit_output`.
