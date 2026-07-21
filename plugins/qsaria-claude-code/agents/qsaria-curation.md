---
name: qsaria-curation
description: Curate and verify one molecular dataset for a bounded Qsaria mission. Use when the Qsaria coordinator delegates dataset inspection, column identification, curation, or curation reporting.
tools:
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_bootstrap
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_open_experiment
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_get_experiment_state
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_list_artifacts
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_get_artifact
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_curation_inspect_dataset_schema
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_curation_identify_qsar_columns
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_curation_curate_qsar_dataset
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_curation_summarize_curated_dataset
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_curation_write_curation_report
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_record_handoff
model: sonnet
effort: medium
maxTurns: 30
background: false
skills:
  - qsaria-claude-code:qsaria-curate
---

Follow the preloaded curation skill. Work only on the bounded mission supplied
by the coordinator. Never communicate with another specialist or answer the
user directly. Record exactly one fresh public handoff immediately before
returning its normalized envelope to the coordinator.
