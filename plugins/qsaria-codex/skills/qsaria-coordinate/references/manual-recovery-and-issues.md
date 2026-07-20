# Manual recovery and GitHub escalation

## Classify the failure

- **Agent orchestration failure:** the project agent is unavailable, crashes, returns a malformed or missing handoff, or chooses incorrect tool arguments without producing a terminal scientific result.
- **MCP integration defect:** an existing deterministic tool rejects valid structured evidence because of routing, serialization, path resolution, or another technical adapter problem.
- **Scientific or contract failure:** Qsaria rejects the data or method, external evaluation is terminal, user input is genuinely missing, or recovery would change scientific intent.

Only the first two classes are eligible for coordinator manual recovery. Never use recovery to bypass a scientific or contract failure.

## Recover with existing tools

1. Verify the failure against experiment state, structured tool results, handoffs, and referenced artifacts.
2. Preserve the same `experiment_id`, exact inputs, backend, representation, split, seed, method, user constraints, and persistence policy.
3. If the project agent cannot finish its bounded mission, read that role's skill and use no broader MCP surface than its allowlist in `contracts/agent-tools.json`.
4. Execute at most one coherent manual recovery mission. It may contain the deterministic calls required by the role, but it must not become an open-ended retry loop.
5. Use MCP lifecycle and scientific tools directly. Read-only filesystem inspection is allowed for diagnosis. Do not use shell writes, `apply_patch`, dependency commands, fabricated paths, placeholder files, rewritten artifacts, or edited manifests as a workaround.
6. Keep the normal error policy: one same-intent retry only for `retryable_error`; stop for terminal scientific failures and `blocked_failed_external_evaluation`; ask the user for genuine `needs_user_input`; never dispatch Report twice.
7. If recovery succeeds, verify every structured result and record one handoff for the recovered role with top-level `execution_mode="coordinator_manual_tools"`. State plainly that the coordinator executed the MCP calls; do not claim that the project agent did.

## Stop when recovery fails

Do not patch the repository from an execution task. Return the verified evidence, the exact failed tool and error, the attempted manual recovery, preserved artifacts, and the experiment id. Say that the workflow could not be completed with the installed capabilities.

Then offer to file a bug in the Qsaria repository's **Issues** tab. This is a proposal, not permission to publish. Show the draft first and require explicit approval before any GitHub write.

## Prepare the issue

Target the repository resolved from Git `origin`; for this distribution it is `abrahambrisset/Qsaria`. Use the dedicated Qsaria Codex bug form when available. Prefer a title such as:

```text
[Qsaria Codex] <failing role or MCP tool>: <short symptom>
```

Include:

- expected and actual behavior;
- exact reproduction steps and original user intent;
- affected agent, MCP tool, and `experiment_id`;
- Qsaria/plugin/contract versions from bootstrap and experiment state;
- handoff status, exact error, and retry or manual-recovery actions;
- relevant artifact ids and repository-relative paths;
- whether the failure reproduces in a fresh Codex task and MCP process;
- a minimal, sanitized log excerpt.

Never attach datasets, persisted models, credentials, environment secrets, absolute home-directory paths, or private scientific results without separate user approval. If an authenticated GitHub capability is unavailable, provide the completed title and body for the user to paste manually.

## Enter maintenance only by explicit authorization

An approval to retry, continue, solve, configure, or work around a failure is not code-maintenance authorization. Ask a separate question that explicitly says which source or plugin files would be patched and that the plugin will be validated and reinstalled.

After explicit approval: keep the experiment paused, implement and test the patch, update compatibility signatures, renew the cachebuster, validate and reinstall the plugin, then require a new Codex task. Reopen only the user-supplied `experiment_id` and retry the smallest failed operation; never rerun successful scientific stages unnecessarily.
