# Qsaria for Claude Science

This directory is a Claude Science integration bundle, not a Claude Code or
Codex plugin. It reuses Qsaria's deterministic MCP, experiment contracts,
artifacts, and model store while keeping the repository read-only during a
scientific session.

Compatibility: Claude Science 0.1.21 or later. Preflight blocks the pilot when
an older desktop build is detected. Qsaria exposes 46 unchanged
synchronous tools plus seven durable start/poll tools used to stay below
Science's 60-second MCP call limit. The MCP never launches an LLM or Agno team.

## 1. Preflight

From the Qsaria repository, run:

```bash
.venv/bin/python plugins/qsaria-claude-science/scripts/preflight.py
.venv/bin/python plugins/qsaria-claude-science/scripts/validate_science_bundle.py
```

The preflight prints the exact local connector command, Claude Science artifact
root, Python runtime, and permission paths for this checkout. Do not copy paths
from another machine.

## 2. Folder permissions in Claude Science

In Claude Science settings, grant persistent folder permissions in this order:

1. the Qsaria repository: read-only;
2. `<repo>/.files`: read-write;
3. `<repo>/data`: read-write;
4. the artifact root printed by preflight: read-only;
5. the resolved Python runtime root printed by preflight: read-only.

Restart Claude Science after changing persistent permissions. Never use
`--dangerously-no-sandbox` or `--dangerously-skip-approvals`. The native
Python, R, and shell tools remain available in Science, so the Qsaria profile
instructions additionally forbid source, dependency, Git, and configuration
changes during a scientific session.

## 3. Local MCP connector

Add a **Local command** connector named exactly `qsaria`:

```text
Command: /bin/zsh
Argument: <repo>/plugins/qsaria-claude-science/scripts/launch-mcp.sh
Environment:
  QSARIA_SCIENCE_ARTIFACT_ROOT=<artifact root printed by preflight>
```

The versioned launcher resolves the repository from its own location, uses
`.venv/bin/python` directly, selects `--profile qsaria`, disables the MCP LLM,
prompts, resources, ChatGPT compatibility, and S3 for this local Science
profile, and configures the shared model catalog at
`data/model_assets/catalog/qsaria_model_catalog.json`.

Connect it and inspect `qsaria_bootstrap` before granting recurring approvals.
The bootstrap must report `llm_policy=disabled`,
`external_mcp_coordinator_v1`, and `claude_science_v1`.

## 4. Skills and profiles

Import the seven directories under `skills/` from this repository and branch:

- `qsaria-science-setup`
- `qsaria-science-coordinate`
- `qsaria-science-curate`
- `qsaria-science-train`
- `qsaria-science-registry`
- `qsaria-science-infer`
- `qsaria-science-report`

Invoke `qsaria-science-setup` once. Approve its `host.agents` operations. It
creates or updates, without duplication, `QSARIA_COORDINATOR` plus five
specialist profiles, attaches only their exact MCP allowlists, and selects the
coordinator for the next message. Re-running setup is the supported update
path. Use the current Sonnet 5 alias (`sonnet`) for the five specialists; the
coordinator inherits the model selected by the user.

Do not enable Claude memory for Qsaria experiment state. A new Science chat
does not resume an experiment automatically; reopen only a user-supplied
`experiment_id`. Persisted models remain available through the shared catalog.

## 5. Durability gate

Before the first long scientific workflow, prove that the Science sandbox lets
a detached worker survive its MCP parent:

```bash
.venv/bin/python plugins/qsaria-claude-science/scripts/durability_probe.py start \
  --root .files --seconds 65
```

Record the returned state path, disconnect or restart the `qsaria` connector,
wait at least 67 seconds, reconnect, then run:

```bash
.venv/bin/python plugins/qsaria-claude-science/scripts/durability_probe.py check \
  --state <returned-state-path>
```

Continue only if the status is `completed`. This repository has a passing local
process-group probe, but the real Claude Science sandbox must be checked on the
target installation.

## 6. Expected operation flow

Science specialists start a long operation, receive `operation_id`, poll every
20 seconds, and fetch the final result only after a terminal status. Each
operation persists request, state, heartbeat, result/error, journal, and worker
log under:

```text
.files/sessions/<experiment_id>/qsaria/operations/<operation_id>/
```

No timeout causes duplicate scientific work. A stale heartbeat with a missing
worker becomes `retryable_error`; terminal scientific failures remain terminal.
Report is synchronous in V1 and runs exactly once for a substantive complete
workflow. Its Markdown artifact under `.files` is canonical; a Science
artifact-library copy is presentation-only and records source id and SHA-256.

## 7. Updating and diagnosing

After pulling a newer Qsaria revision:

1. rerun preflight and validation;
2. reconnect the local connector;
3. reimport the seven skills from the intended Git commit;
4. rerun `qsaria-science-setup`;
5. verify `qsaria_bootstrap` and the profile allowlists.

Compatibility metadata is checked with:

```bash
.venv/bin/python plugins/qsaria-claude-science/scripts/sync_qsaria_science.py --check
```

For a failure, preserve `experiment_id`, `operation_id`, the structured state,
worker log, and referenced artifact ids. Never patch Qsaria from a scientific
session; let the coordinator prepare a sanitized Issue draft instead.
