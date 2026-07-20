---
name: qsaria-coordinate
description: Coordinate Qsaria experiments in Codex, including dataset curation, QSAR training, model persistence and catalog lookup, inference, ensemble work, experiment recovery, and final scientific reporting. Use for any user request that asks Codex to run, inspect, resume, or critically analyze a Qsaria workflow.
---

# Coordinate Qsaria

Act as the only workflow coordinator. Keep scientific execution in the deterministic Qsaria MCP profile and delegate role work only to the five Qsaria project agents.

Read [routing.md](references/routing.md), [verification-and-reporting.md](references/verification-and-reporting.md), [manual-recovery-and-issues.md](references/manual-recovery-and-issues.md), [scientific-invariants.md](../../contracts/scientific-invariants.md), and [handoff.schema.json](../../contracts/handoff.schema.json) before running a workflow.

## Start safely

1. Call `qsaria_bootstrap` once for the task and verify that the profile reports `llm_policy=disabled`, compatible contract versions, and its storage-concurrency mode. For S3, require the announced `s3_single_writer` mode and `single_writer_acknowledged=true`; otherwise stop instead of assuming distributed locking.
2. Classify the request using the routing reference.
3. Never reopen the most recent experiment implicitly.
4. Never call `qsaria_list_experiments` during bootstrap. List history only when the user explicitly asks for it.
5. For an explicit `experiment_id`, call `qsaria_open_experiment`. For a new workflow that will write, call `qsaria_create_experiment` only when the first write is required. Model persistence defaults to the durable catalog. Set experiment metadata `persistence_policy=session_only` only when the user explicitly asks not to retain newly trained models.
6. Keep the report language from the latest user request in the experiment contract.

## Route and supervise

- Delegate only to `qsaria_curation`, `qsaria_training`, `qsaria_registry`, `qsaria_inference`, or `qsaria_report`.
- Handle simple catalog reads, state inspection, and artifact reads directly with deterministic MCP tools. This avoids creating an experiment merely to record a sub-agent handoff.
- Give each agent one bounded mission with its `experiment_id`, exact inputs, user constraints, report language, and relevant artifact or model identifiers.
- Never ask one sub-agent to contact another. All results return to you.
- Inspect every returned handoff and its referenced structured artifacts before continuing.
- When Training produces a persistence plan, treat the experiment's persisted `persistence.status=pending` state as a deterministic Registry barrier. Dispatch Registry with the exact candidate manifest or recommended payload route before Report. Never reinterpret a session checkpoint under `.files` as a reusable catalog model.
- Treat `needs_user_input` as a pause and ask one precise question. Do not guess the answer.
- Retry only a handoff explicitly classified `status=retryable_error`, at most once with the same scientific intent. Report is never dispatched a second time. Do not retry `terminal_failure` or change data, backend, validation, method, or constraints silently.
- Preserve `blocked_failed_external_evaluation` as a terminal result.

## Separate execution from maintenance

- During a Qsaria scientific request, stay in execution mode: use project agents, existing deterministic Qsaria MCP tools, and read-only evidence inspection. Never edit source, tests, plugin files, manifests, dependencies, or experiment evidence, and never create placeholder files or directories to satisfy a failing validator.
- Treat phrases such as “solve it”, “try again”, “go ahead”, or “continue” after a workflow failure as permission for an in-scope tool recovery only. They do not authorize a code patch, dependency change, plugin reinstall, Git operation, or GitHub write.
- If a project agent has a technical orchestration failure, attempt one bounded coordinator recovery with the existing MCP tools allowed for that role, as defined in the recovery reference. Preserve the exact scientific intent and all terminal-error rules. Mark a successful recovery handoff with `execution_mode="coordinator_manual_tools"`.
- If manual recovery fails, stop cleanly, explain the defect, and offer a prepared GitHub Issue draft. Never publish an issue without the user's explicit approval.
- Enter maintenance mode only when the user explicitly authorizes modifying the code. Pause the scientific workflow, patch and validate separately, synchronize compatibility, renew the plugin cachebuster, reinstall the plugin, and require a new Codex task with a fresh MCP process before reopening the explicit `experiment_id`.

## Finish deliberately

- Skip Report for catalog lookup, experiment inspection, artifact reading, and simple technical operations.
- For a scientific workflow needing a user-facing report, call `qsaria_report` exactly once after all scientific agents have finished.
- A terminal `blocked_failed_external_evaluation` stops all further scientific execution, but it may still be followed by the experiment's single evidence-only Report mission when a final report is required. If Report was already recorded, reuse verified evidence and never dispatch it again.
- If that single Report mission fails, preserve the failure and present only verified evidence; never hide the missing saved report behind a second Report call.
- Verify the report against structured artifacts and handoffs. If they disagree, correct the presentation from the higher-trust evidence; do not edit the evidence.
- Call `qsaria_complete_experiment` only after the final state and blockers are known.
- Present the Qsaria report first. Then add a clearly separate **Codex analysis** with limitations, contradictions, and actionable recommendations.
- Perform external research only after Qsaria results exist. Cite every external factual claim and never let external research overwrite Qsaria evidence.
