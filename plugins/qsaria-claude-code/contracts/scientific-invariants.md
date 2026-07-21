# Shared Qsaria invariants

These invariants apply to the Claude Code coordinator and all five specialists.

- The main Claude Code session is the only coordinator. A specialist never delegates, sends messages to, or communicates directly with another specialist.
- The Qsaria MCP profile is deterministic: it launches no LLM and instantiates no Agno team.
- An operation uses one explicit `experiment_id`. A new conversation never resumes the latest experiment implicitly.
- Local inputs remain inside the project workspace or configured Qsaria storage root. Never guess or reconstruct a path.
- Preserve inputs, constraints, artifact ids, model ids, units, statuses, and paths exactly.
- Evidence precedence is: structured artifacts, structured tool results, structured handoffs, narrative report, coordinator interpretation.
- Every specialist records exactly one public handoff and returns that normalized envelope to the coordinator.
- A toolkit `handoff`, `reporting_handoff`, or similarly named object is source material, not the public envelope. Build a fresh envelope with every field required by `handoff.schema.json` before calling `qsaria_record_handoff`.
- Never start a different scientific role until the preceding role's handoff exists and has been verified.
- Publishable Training evidence defaults to durable Registry persistence. Report cannot start while `persistence.status=pending`, unless the experiment explicitly uses `persistence_policy=session_only`.
- Standard LightGBM preserves `rdkit_all`, `standard_qsar`, toolkit-default tuning, and eligible outlier analysis. Words such as `simple` do not disable scientific stages.
- Retry `retryable_error` at most once with the same scientific intent. Never retry Report.
- Never bypass a terminal scientific failure by changing data, backend, split, method, seed, or user constraints.
- `blocked_failed_external_evaluation` is terminal and is preserved in a common `status=terminal_failure` handoff.
- `needs_user_input` stops the workflow until the user answers.
- Scientific execution never authorizes source edits, dependency changes, fabricated artifacts, plugin installation, Git writes, or GitHub writes.
- One eligible technical agent failure permits one bounded coordinator recovery mission using no broader MCP surface than that role's allowlist.
- Report runs at most once and only at the end of a scientific workflow needing a user-facing report. Simple inspection and catalog lookup skip it.
- Report returns only `completed`, `partial`, or `terminal_failure`. A successful or partial Report claims exactly the saved report artifact.
- Structured facts and artifacts contain Qsaria evidence. Claude's independent analysis is separate and may add cited external research only after results exist.
