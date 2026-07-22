---
name: qsaria-science-coordinate
description: Coordinate Qsaria experiments from Claude Science, including curation, training, persistence, catalog lookup, inference, ensembles, durable-operation recovery, final reporting, artifact presentation, and independent cited analysis.
---

# Coordinate Qsaria

Act as the sole coordinator. Qsaria specialists report only to you; never act
as a sixth specialist and never let specialists contact one another.

## Start and route

Call `qsaria_bootstrap` once. Require the deterministic profile, compatible
contracts, `claude_science_v1`, typed training requests, and detached operation
support. Never resume or list an old experiment automatically. Open only an
explicit `experiment_id`; create one at the first scientific write.

Use direct MCP reads for catalog and state inspection, skipping Report. For
scientific work delegate one bounded mission at a time with
`host.delegate(task=..., profile="QSARIA_ROLE", model="sonnet",
output_schema=HANDOFF_SCHEMA)`. Include the exact experiment, inputs, user
constraints, language, artifact ids, and model ids. Require
`structured_output`; prose is not a valid handoff.

Routes:

| Intent | Roles | Report |
| --- | --- | --- |
| Catalog or state inspection | Coordinator reads | Skip |
| Curation | Curation | Once when substantive |
| Training | Curation, Training, Registry when persistence is pending | Once |
| Prediction or external evaluation | Inference | Once when substantive |
| Ensemble work | Registry | Once when substantive |

## Verify and recover

Inspect every normalized handoff and referenced artifact. Trust structured
artifacts, structured tool results, structured handoffs, narrative report,
then your interpretation. Pause for `needs_user_input`. Retry only one explicit
`retryable_error`, with unchanged scientific intent. Never retry Report or a
terminal scientific failure.

If a specialist crashes or returns no valid handoff, attempt one manual
recovery using no broader MCP surface than that role. Use the same start/poll
protocol and record `execution_mode="coordinator_manual_tools"`. If recovery
fails, stop and offer a sanitized Issue draft. Never patch code, alter
dependencies, operate Git/GitHub, or invent an artifact in a scientific
session.

## Finish

Resolve pending persistence before Report unless the user explicitly selected
`session_only`. Delegate `QSARIA_REPORT` exactly once. After verifying its
single report artifact, complete the experiment.

Fetch the canonical Markdown report with `qsaria_get_artifact` using up to the
1 MiB preview limit. Create a presentation copy in the temporary Science
workspace containing `experiment_id`, source `artifact_id`, content SHA-256,
and canonical relative path, then save it with the native `save_artifacts`
tool. Do not alter scientific values or treat this copy as canonical.

Present the saved Qsaria report first, then a separate **Claude analysis**.
External research occurs only after results exist and every external claim is
cited. Disable use of Claude memory for experiment state; Qsaria files are the
only persistent scientific state.

Use the handoff JSON Schema installed with this bundle as `HANDOFF_SCHEMA`.
