# Qsaria Claude Science invariants

- `QSARIA_COORDINATOR` is the sole coordinator. Specialists never delegate or communicate directly.
- The MCP profile launches no LLM and no Agno team. Claude Science supplies all reasoning.
- A new session never resumes the latest experiment. Open only an explicit `experiment_id`.
- The repository is read-only in the Science sandbox. Scientific writes are limited to `.files` and `data`.
- Inputs, constraints, artifact ids, model ids, units, statuses, and paths are preserved exactly.
- Evidence precedence is structured artifacts, structured tool results, structured handoffs, report narrative, then coordinator interpretation.
- Each specialist records and submits exactly one fresh public handoff. Toolkit handoff-like payloads are evidence, not the public envelope.
- Long writes use one role-scoped start tool and non-blocking state/result polling. A timeout never authorizes duplicate work.
- Publishable Training evidence defaults to Registry persistence below `data/model_assets/internal`.
- Standard LightGBM preserves `rdkit_all`, `standard_qsar`, 50 eligible tuning trials, and eligible outlier analysis.
- Retry one explicit technical `retryable_error` at most once, without changing scientific intent. Never retry Report.
- Never bypass terminal scientific failure by changing data, backend, representation, split, method, seed, or user constraints.
- `blocked_failed_external_evaluation` remains terminal and is preserved in structured facts.
- `needs_user_input` pauses the workflow.
- Scientific execution never authorizes source edits, dependency changes, Git/GitHub actions, plugin installation, or fabricated artifacts.
- Report runs at most once and simple inspection skips it. The canonical report remains under `.files`.
- A native Science report copy carries canonical provenance but never becomes scientific source of truth.
- External research starts only after Qsaria results exist and every external claim is cited.
