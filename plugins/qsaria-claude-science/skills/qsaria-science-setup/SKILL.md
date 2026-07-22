---
name: qsaria-science-setup
description: Configure or update the Qsaria connector profiles in Claude Science. Use after importing the Qsaria Science skills, after a bundle update, or when the coordinator and five specialist profiles are missing or stale.
---

# Configure Qsaria Science

Create or reconcile the six dedicated profiles through `host.agents`. Require a
connected local MCP connector named `qsaria` before changing profiles. Never
edit Claude Science's configuration database or Qsaria source files directly.

## Preflight

1. Call `host.agents.list_connectors("qsaria")` and require an authenticated,
   connected connector exposing `qsaria_bootstrap` and the seven operation
   lifecycle tools.
2. Call `host.mcp("qsaria", "qsaria_bootstrap", {})`. Require
   `llm_policy="disabled"`, coordinator contract
   `external_mcp_coordinator_v1`, and `claude_science_v1` among
   `supported_clients`.
3. Require all seven `qsaria-science-*` skills to be installed. Stop with an
   exact missing-skill list otherwise.

## Reconcile profiles

Use `host.agents.create` when a profile is absent and `host.agents.update` when
it exists. Pass an explicit one-item `skill_names` list so every profile is
curated. Read `plugins/qsaria-claude-science/contracts/agent-tools.json` from
the granted read-only Qsaria repository. For each specialist build the exact
anchored regex from its `tools` array with
`"^(?:" + "|".join(re.escape(tool) for tool in tools) + ")$"`. Use the
contract's `coordinator_include_pattern` for `QSARIA_COORDINATOR`.

Create the following mapping:

| Profile | Display name | Skill |
| --- | --- | --- |
| `QSARIA_COORDINATOR` | Qsaria Coordinator | `qsaria-science-coordinate` |
| `QSARIA_CURATION` | Qsaria Curation | `qsaria-science-curate` |
| `QSARIA_TRAINING` | Qsaria Training | `qsaria-science-train` |
| `QSARIA_REGISTRY` | Qsaria Registry | `qsaria-science-registry` |
| `QSARIA_INFERENCE` | Qsaria Inference | `qsaria-science-infer` |
| `QSARIA_REPORT` | Qsaria Report | `qsaria-science-report` |

Reattach `qsaria` with `include_tools_pattern` to replace stale exclusions and
remove every other connector from these dedicated profiles.

Set each system prompt to: obey the attached Qsaria skill; use only the attached
connector surface; never modify source, dependencies, Git, GitHub, or Claude
configuration; never delegate from a specialist; preserve structured evidence.
For the coordinator add: act as the sole coordinator and keep independent
analysis separate from the Qsaria report.

After reconciliation, call `host.agents.get` for every profile and verify the
exact skill, connector, and exclusions. Then call
`host.agents.switch("QSARIA_COORDINATOR")`; explain that the switch applies on
the user's next message.
