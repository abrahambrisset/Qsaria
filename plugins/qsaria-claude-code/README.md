# Qsaria for Claude Code

Current bundle: `0.2.0`. The only named scientific protocol is
`standard_qsar`; advanced validation is expressed with an explicit strategy.

This personal plugin makes Claude Code the sole coordinator of deterministic
Qsaria workflows. Five restricted specialists perform curation, training,
registry, inference, and reporting work through the existing Qsaria MCP
profile. The MCP server launches no LLM and no Agno team.

## Prerequisites

Run Claude Code from the Qsaria repository root and prepare the project once:

```text
uv sync --extra mcp --extra prediction
```

The plugin deliberately uses the project environment through
`${CLAUDE_PROJECT_DIR}`. It is not a standalone distribution of the Python
runtime.

## Personal installation

Do not install or reload this plugin while another Qsaria client is writing to
the shared model catalog. From the Qsaria repository, add the local marketplace
and install the plugin at project scope:

```text
claude plugin marketplace add .
claude plugin install qsaria-claude-code@personal --scope project
```

In the Claude Code UI, the equivalent flow is to add this repository as a
local marketplace, install `qsaria-claude-code@personal` for the current
project, then run `/reload-plugins`. Confirm the `qsaria` server in `/mcp`.
On a cold start, wait until `/mcp` reports `qsaria` as connected before sending
the first scientific request; loading the project environment can take a few
seconds.

The plugin is personal and local. It is not submitted to an Anthropic public
marketplace.

## Expected interaction

Use ordinary scientific requests. For example:

```text
Train a standard Qsaria LightGBM regression model from
data/pxr_challenge/pxr-challenge_TRAIN.csv. Use SMILES as the molecular column
and pEC50 as the target. Persist the resulting model and give me a critical
analysis after the Qsaria report.
```

The main `qsaria-coordinator` profile routes the request. Users do not call MCP
tools directly.

## Execution and maintenance

Scientific sessions may use only the installed Qsaria agents and deterministic
MCP tools. They never edit source, dependencies, plugin files, Git state, or
GitHub issues. A technical failure permits one bounded manual-tool recovery.
If that fails, the coordinator prepares a sanitized issue draft.

Code maintenance requires separate explicit authorization and a normal Claude
Code session with the Qsaria coordinator profile overridden or the plugin
temporarily disabled. Reload the plugin only after maintenance validation.

## Development checks

The repository-side checks do not start a scientific workflow:

```text
uv run --no-sync python plugins/qsaria-claude-code/scripts/preflight.py
uv run --no-sync python plugins/qsaria-claude-code/scripts/validate_claude_bundle.py
uv run --no-sync python plugins/qsaria-claude-code/scripts/validate_shared_runtime.py
uv run --no-sync python plugins/qsaria-claude-code/scripts/sync_qsaria_claude.py --check
uv run --no-sync python plugins/qsaria-claude-code/scripts/smoke_claude_plugin.py
```

The shared-runtime check verifies the additive multi-client bootstrap and the
exact 46-tool deterministic surface without creating an experiment. The smoke
script runs the official Claude validator when a `claude` CLI is in `PATH`;
otherwise it reports a non-failing skip. Installation and the end-to-end
scientific pilot remain explicit operations.

After a successful standard LightGBM PXR pilot, maintainers can record its
non-sensitive experiment identifier together with refreshed signatures:

```text
uv run --no-sync python plugins/qsaria-claude-code/scripts/sync_qsaria_claude.py \
  --update --pilot-status passed --pilot-experiment-id exp_...
```

Compatibility metadata records the source commit, contract versions, runtime
signature, plugin signature, and whether the signed tree was dirty. It must be
refreshed explicitly after reviewed source changes.
