---
name: qsaria-report
description: Produce and save the single final evidence-first Qsaria report from completed experiment facts, handoffs, and structured artifacts in Claude Code.
user-invocable: false
---

# Report a Qsaria Experiment

Write one familiar structured Qsaria report from existing evidence. Do not run
new science, reinterpret failures as success, or add the coordinator's
independent recommendations.

## Evidence and workflow

1. Open the explicit experiment and call `qsaria_report_build_context` once.
2. Use `report_facts.schema_version=2.0` and `report_tables` as primary sources.
3. Resolve contradictions by evidence precedence: structured artifacts,
   structured tool results, handoffs, narrative text.
4. Write entirely in the requested language while preserving identifiers,
   statuses, metric names, columns, artifact ids, and paths verbatim.
5. State skipped, disabled, missing, and not-evaluated modules plainly.
6. Save once with `qsaria_report_save` and verify the returned artifact.

Treat `report_tables` as verified editorial material, not mandatory blocks.
Select zero to three tables only when clearer than prose. You may omit redundant
rows or columns and reorder them, but every displayed label, population,
metric, and value must be unchanged from structured material. Never recompute
cells or merge non-comparable populations.

For a French training report, use this order: `Resume executif`, `Dataset,
curation et qualite des donnees`, `Environnement de travail`, `Protocole
d'entrainement`, `Optimisation des hyperparametres`, `Analyse des outliers`,
the applicable evaluation results, then `Gouvernance et artefacts generes`.
Use equivalent plain English headings for English reports.

Clearly distinguish internal, external, and non-evaluated full-train evidence.
For external evaluation, state that isolated test data were not used for model
selection. Never claim ensemble improvement without same-row ensemble-level
evaluation.

## Return one public handoff

Build a fresh complete envelope with `agent="qsaria_report"` and
`execution_mode="project_agent"`. Use only `completed`, `partial`, or
`terminal_failure`; never return `retryable_error` or `needs_user_input`.

For completed or partial status, claim exactly the artifact returned by
`qsaria_report_save`. If verification fails after save, return
`terminal_failure` and identify that same artifact as unvalidated evidence.
Never submit a reporting toolkit payload directly.

Call `qsaria_record_handoff` exactly once immediately before returning and
return its normalized envelope. Do not include the separate Claude analysis.
