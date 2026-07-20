---
name: qsaria-report
description: Produce the single final evidence-first Qsaria report from completed experiment facts, handoffs, and structured artifacts. Use only inside the qsaria_report project agent when the coordinator requests the final report for a scientific workflow.
---

# Report a Qsaria Experiment

Write one familiar, structured Qsaria report from existing evidence. Do not run new science, reinterpret failed operations as success, or add Codex's independent recommendations.

Read [report-contract.md](references/report-contract.md), [scientific-invariants.md](../../contracts/scientific-invariants.md), and [handoff.schema.json](../../contracts/handoff.schema.json) before using tools.

## Workflow

1. Open the explicit experiment and call `qsaria_report_build_context` once.
2. Use `report_facts` schema version 2 and `report_tables` as the primary reporting contract. Use older handoff fields only when the context explicitly marks a legacy workflow.
3. Resolve contradictions by the shared evidence order. Do not average, reconcile, or invent values.
4. Write entirely in the requested report language while preserving model identifiers, statuses, metric names, column names, artifact identifiers, and paths verbatim.
5. State skipped, disabled, missing, or not-evaluated modules plainly. Distinguish internal evidence, external test evidence, and full-train runs.
6. Treat `report_tables` as verified editorial material, not mandatory output blocks. Select zero to three tables only when they communicate necessary relationships more clearly than prose. You may omit redundant rows or columns and reorder them for clarity, but preserve every displayed label, population, metric, and value from the structured source. Never recompute cells, merge non-comparable populations, or invent a table from narrative text.
7. Save the finished report once with `qsaria_report_save`, then verify the returned report artifact.

## Return the handoff

Build the common envelope with `agent=qsaria_report`, the saved report artifact, factual warnings, blockers, and a concise summary. Use only `completed`, `partial`, or `terminal_failure`: this single final mission never returns `retryable_error` or `needs_user_input`. For `completed` or `partial`, set `artifact_ids` to exactly the single artifact returned by `qsaria_report_save`. If verification fails after saving, return `terminal_failure` and claim that same artifact as unvalidated evidence. Call `qsaria_record_handoff` exactly once, immediately before returning its normalized envelope to the coordinator. Do not include the coordinator's separate analysis.
