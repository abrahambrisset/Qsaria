# Shared Qsaria invariants

These invariants apply to the coordinator and all five project agents.

- Codex is the only coordinator. A project agent never delegates or communicates directly with another project agent.
- The Qsaria MCP profile is deterministic: it launches no LLM and instantiates no Agno team.
- An operation uses one explicit `experiment_id`. A new task never resumes the latest experiment implicitly.
- Local MCP input paths are confined to the task workspace and configured Qsaria storage root. Deployments may extend that trust boundary explicitly with the path-separator-delimited `QSARIA_MCP_ALLOWED_INPUT_ROOTS`; an agent never treats it as permission to guess a path.
- An `s3://` input is accepted at the MCP trust boundary only for the configured bucket. Passing that boundary does not imply that every existing Path-based scientific toolkit can consume an S3 URI.
- V1 S3 state/report persistence requires `QSARIA_S3_SINGLE_WRITER=true`, which is an explicit deployment guarantee that only one Qsaria state writer is active. V1 provides no distributed S3 lock and no database.
- Path-based scientific toolkit artifacts remain local under the experiment layout even when experiment-state and report metadata use S3. Never rewrite or describe a local scientific artifact as an S3 object unless a tool explicitly returns one.
- Inputs, constraints, artifact identifiers, model identifiers, units, statuses, and paths are preserved exactly. Never infer a path or reconstruct a structured payload when a tool returned one.
- Evidence precedence is: structured artifacts, structured tool results, structured handoffs, narrative report, coordinator interpretation.
- Every project agent records exactly one handoff for its mission and returns that same envelope to the coordinator. If an unavailable or failed agent is replaced by the coordinator's single manual-tool recovery mission, the coordinator records one handoff for the affected role with `execution_mode=coordinator_manual_tools` and never claims the agent performed those calls.
- A toolkit-returned `handoff`, `reporting_handoff`, or similarly named payload is scientific source material, not the public MCP handoff envelope. A project agent always builds a fresh public envelope with `schema_version="1.0"`, its exact `experiment_id`, `execution_mode="project_agent"`, and every required field from `handoff.schema.json`; it never submits a toolkit payload directly to `qsaria_record_handoff`.
- A different scientific role never starts while the preceding role has successful tool evidence without its structured handoff. The same role may use multiple tools before recording its one mission handoff.
- Publishable Training evidence defaults to durable Registry persistence. Report cannot start while `persistence.status=pending`; only an explicit user-requested `persistence_policy=session_only` skips catalog materialization.
- An unqualified standard Qsaria LightGBM workflow preserves `rdkit_all`, `standard_qsar`, toolkit-default tuning, and eligible outlier analysis. Presentation words such as `simple` never disable scientific stages.
- A technical `retryable_error` permits at most one same-intent retry by the coordinator, except that the single Report mission is never repeated.
- A scientific terminal failure is never bypassed by silently changing data, backend, split, method, seed, or user constraints.
- `blocked_failed_external_evaluation` is terminal. Its common handoff uses `status=terminal_failure` and preserves the exact `blocked_failed_external_evaluation` value in structured `facts` and the relevant blocker; never add it as a sixth handoff status.
- `needs_user_input` stops the workflow until the user answers.
- A scientific execution request authorizes agents, existing deterministic MCP tools, and read-only evidence inspection only. It never authorizes source edits, dependency changes, fabricated files or directories, plugin reinstallation, Git writes, or GitHub writes. Code maintenance and issue publication each require separate explicit user approval.
- An eligible technical agent failure permits one bounded coordinator recovery mission using no broader MCP surface than the affected role's allowlist. Failure of that recovery stops the workflow and may be followed only by an offered, sanitized GitHub Issue draft.
- Report runs at most once, at the end of a scientific workflow that needs a user-facing report. Simple inspection and catalog lookup skip it.
- Because Report is a single final mission, its handoff status is limited to `completed`, `partial`, or `terminal_failure`; it never returns `retryable_error` or `needs_user_input`. Resolve missing prerequisites before dispatching Report. A successful or partial Report handoff claims exactly the artifact returned by `qsaria_report_save`. A terminal Report handoff claims that artifact when one was already saved, marking it as unvalidated evidence rather than success.
- Structured facts and artifacts contain Qsaria evidence. Codex's independent analysis is presented separately and may add cited external research only after results exist.
