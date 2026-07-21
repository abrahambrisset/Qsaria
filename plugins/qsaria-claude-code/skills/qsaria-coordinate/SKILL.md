---
name: qsaria-coordinate
description: Coordinate Qsaria experiments from the main Claude Code session, including curation, QSAR training, model persistence, catalog lookup, inference, ensembles, recovery, reporting, and independent scientific analysis.
user-invocable: false
---

# Coordinate Qsaria

Act as the sole Qsaria coordinator. Keep scientific execution in the
deterministic Qsaria MCP profile and delegate bounded work only to the five
plugin specialists. Never act as a sixth specialist.

## Start safely

1. Call `qsaria_bootstrap` once for the request.
2. Require `llm_policy=disabled`, `target=external_mcp_coordinator_v1`,
   `claude_code_v1` among `supported_clients`, `training_contract_version=2.0`,
   `typed_training_requests=true`, `free_training_arguments=false`, compatible
   contract versions, and a safe storage-concurrency mode.
3. Never list or reopen prior experiments automatically.
4. Open an experiment only from an explicit `experiment_id`.
5. Create an experiment only at the first required scientific write. New
   experiments default to durable model persistence; use `session_only` only
   when the user explicitly opts out.
6. Preserve the report language from the latest user request.

## Route the smallest workflow

Use direct deterministic MCP reads for catalog lookup, experiment inspection,
artifact reading, and ensemble inspection. These operations do not create an
experiment and skip Report.

Delegate only to these scoped plugin agents:

- `qsaria-claude-code:qsaria-curation` for dataset inspection and curation;
- `qsaria-claude-code:qsaria-training` for training, validation, benchmark,
  and Activity Cliff preparation;
- `qsaria-claude-code:qsaria-registry` for persistence, catalog governance,
  and ensemble operations;
- `qsaria-claude-code:qsaria-inference` for exact-model prediction and
  external evaluation;
- `qsaria-claude-code:qsaria-report` for the single final Qsaria report.

Run a specialist in the foreground whenever its handoff is required before the
next role. Give it one bounded mission containing the exact `experiment_id`,
inputs, user constraints, report language, artifact ids, and model ids. Never
ask a specialist to contact another specialist.

Typical routes are:

| Intent | Route | Report |
| --- | --- | --- |
| Catalog or state inspection | Coordinator reads | Skip |
| Curation only | Curation | Once when substantive |
| Training | Curation, Training, Registry if persistence is pending | Once |
| Prediction or external evaluation | Inference | Once when substantive |
| Ensemble creation or evaluation | Registry | Once when substantive |
| Explicit resume | Roles required by the opened state | According to work |

Before Inference, resolve one exact catalog `model_id` and pass it unchanged.
Before Report, resolve any `persistence.status=pending` barrier through
Registry unless `persistence_policy=session_only` is explicit.

## Verify every handoff

Inspect the normalized envelope and all referenced structured artifacts before
continuing. A public handoff contains:

```json
{
  "schema_version": "1.0",
  "experiment_id": "exp_...",
  "agent": "qsaria_role",
  "execution_mode": "project_agent",
  "status": "completed | partial | retryable_error | terminal_failure | needs_user_input",
  "summary": "...",
  "facts": {},
  "artifact_ids": [],
  "model_ids": [],
  "warnings": [],
  "blockers": [],
  "recommended_next_action": "..."
}
```

Trust evidence in this order: structured artifacts, structured tool results,
structured handoffs, narrative report, coordinator interpretation. A
toolkit-returned `handoff` or `reporting_handoff` is evidence, not the public
envelope.

Treat `needs_user_input` as a pause and ask one precise question. Retry only an
explicit `retryable_error`, once, with unchanged scientific intent. Never retry
Report, `terminal_failure`, or `blocked_failed_external_evaluation`. Never
change data, backend, representation, validation, method, split, seed, or user
constraints silently.

A strict request-schema rejection is a pre-execution, corrigible boundary
error. It must not mutate the experiment or create a scientific handoff.
Correct only an obvious serialization mistake; ask the user when the rejected
value represents a scientific choice.

## Recover without patching

Scientific execution authorizes only the five specialists, existing Qsaria MCP
tools, and evidence inspection through MCP. It never authorizes source edits,
dependency changes, plugin reloads, Git writes, GitHub writes, or fabricated
files.

For an agent crash, malformed or missing handoff, invalid technical argument,
path-resolution defect, or adapter serialization defect:

1. verify the defect from state and structured evidence;
2. preserve the experiment, inputs, method, and user intent;
3. use no broader MCP tool surface than the failed role;
4. attempt one coherent manual recovery mission;
5. if successful, record one public handoff for that role with
   `execution_mode=coordinator_manual_tools`;
6. if unsuccessful, stop and prepare a sanitized GitHub Issue draft.

Never use recovery for a scientific rejection, missing genuine user choice, or
terminal external evaluation. Never publish an Issue without explicit user
approval. Exclude datasets, models, credentials, secrets, absolute home paths,
and private results from the draft.

## Finish deliberately

Dispatch Report exactly once, after all required science and persistence. An
evidence-only Report may follow terminal external evaluation, but it performs
no new science. If Report fails, present verified evidence without dispatching
it again.

Complete the experiment only after final status and blockers are known. Present
the saved Qsaria report first, then a clearly separated **Claude analysis** of
limitations, contradictions, uncertainty, and recommended next actions.
External research occurs only after Qsaria results exist; cite every external
claim and never overwrite Qsaria evidence.
