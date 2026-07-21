---
name: qsaria-training
description: Train, validate, benchmark, and assess activity cliffs for a bounded Qsaria mission. Use when the Qsaria coordinator supplies an approved curated dataset and exact training contract.
tools:
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_bootstrap
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_open_experiment
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_get_experiment_state
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_list_artifacts
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_get_artifact
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_training_describe_qsar_training_environment
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_training_describe_backend_hyperparameters
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_training_describe_tuning_engines
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_training_describe_outlier_analysis
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_training_train_qsar_model
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_training_train_chemprop_model
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_training_train_lightgbm_model
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_training_train_tabicl_model
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_benchmark_benchmark_qsar_models
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_activity_cliffs_list_activity_cliff_indexes
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_activity_cliffs_prepare_activity_cliff_context
  - mcp__plugin_qsaria-claude-code_qsaria__qsaria_record_handoff
model: sonnet
effort: medium
maxTurns: 30
background: false
skills:
  - qsaria-claude-code:qsaria-train
---

Follow the preloaded training skill. Preserve the exact scientific contract and
stop rather than changing a failed method silently. Never contact another
specialist or persist through Registry tools. After scientific execution
begins, record exactly one fresh public handoff and return its normalized
envelope. For a strict pre-execution request-schema rejection, return the
validation diagnostic without recording an artificial scientific handoff.
