# Verification and final presentation

## Verify each handoff

Reject a handoff as incomplete when it lacks required schema fields, has a different experiment id, names the wrong agent, references unknown artifacts, or claims facts unsupported by structured results.

For completed or partial work:

1. Read the experiment state.
2. Resolve every referenced artifact needed for the next decision.
3. Compare row counts, metrics, model ids, statuses, paths, gates, and blockers with the handoff.
4. Route only from verified facts.

For a standard LightGBM mission, also compare the structured result with the
delegated workflow contract. Require `rdkit_all`, `standard_qsar`, the current
default requested tuning-trial count, and enabled outlier analysis when the
task is eligible. Do not use `training_profile=local_standard` as a substitute
for those facts. If the effective workflow differs, stop before Registry and
present the integration mismatch explicitly.

For errors:

- `retryable_error`: one same-intent retry maximum, except for Report, which is never dispatched twice.
- `terminal_failure`: stop that branch and preserve completed artifacts.
- `needs_user_input`: ask the user; do not continue the dependent branch.
- `partial`: continue only if the missing evidence is non-blocking and explicitly disclosed.

An agent orchestration failure may instead enter the single bounded manual-recovery path described in `manual-recovery-and-issues.md`. Verify the recovered MCP results exactly like an agent handoff and require `execution_mode="coordinator_manual_tools"` on its recovery handoff. If the existing tool surface cannot complete the same scientific mission, stop and propose a sanitized GitHub Issue draft. Do not modify code, fabricate inputs, or silently widen the route.

## Report and coordinator analysis

Report is a peer sub-agent and is called once after scientific execution. The coordinator then performs a quick double check against structured evidence.

The final response has two conceptually separate parts:

1. **Qsaria report** — the saved report's familiar structure and factual content.
2. **Codex analysis** — an independent assessment of limitations, uncertainty, contradictions, and recommended next actions.

Do not insert external research into the Qsaria report after it is saved. Add it only to Codex analysis, with direct citations. Make clear when a recommendation is an inference rather than an artifact fact.
