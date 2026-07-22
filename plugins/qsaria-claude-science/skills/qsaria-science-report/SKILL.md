---
name: qsaria-science-report
description: Produce and save the single final evidence-first Qsaria report from completed experiment facts, handoffs, and structured artifacts in Claude Science.
---

# Report a Qsaria Experiment

Write one structured Qsaria report from existing evidence. Do not run new
science or include the coordinator's independent recommendations.

Open the explicit experiment and call `qsaria_report_build_context` once. Use
`report_facts` version 2 and `report_tables` as primary material. Resolve
contradictions by evidence precedence. Preserve identifiers, statuses,
metrics, columns, artifact ids, and paths exactly. State skipped, disabled,
missing, and non-evaluated modules plainly.

Select zero to three tables when clearer than prose. You may omit redundant
rows or columns, but never recompute values or merge incomparable populations.
Clearly distinguish internal, external, and full-train evidence. Never claim
ensemble improvement without same-row ensemble evaluation.

Save the report exactly once with `qsaria_report_save` and verify its artifact.
Do not use the asynchronous operation facade for Report and never retry Report.

Build one fresh handoff with `agent="qsaria_report"` and
`execution_mode="project_agent"`. Its status is only `completed`, `partial`, or
`terminal_failure`. A completed or partial handoff claims exactly the saved
report artifact. Record the handoff once and submit the normalized envelope
through `host.submit_output`.
