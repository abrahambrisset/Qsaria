# Qsaria for Codex

Current bundle: `0.3.1`. The only named scientific protocol is
`standard_qsar`; advanced holdout, repeated holdout, cross-validation,
scaffold, cluster, and full-train workflows use explicit validation strategies.

This repo-local plugin lets Codex coordinate Qsaria without launching an
additional LLM inside the MCP server. The server exposes deterministic Qsaria
operations; Codex remains the only coordinator and reasoning layer.

The existing Agno/Chainlit application is unchanged. This bundle contains
Codex-specific skills and project-agent templates only.

Codex plugins do not currently install project agents as a native plugin
component. The distributable templates therefore live under `assets/agents/`
and are copied explicitly into the repository's `.codex/agents/` directory.
The 0.2.x line adds no hook and no separate asynchronous runner.

## Runtime boundaries

The main Codex task is the sole coordinator. It can delegate bounded work to
five peer project agents: `qsaria_curation`, `qsaria_training`,
`qsaria_registry`, `qsaria_inference`, and `qsaria_report`. Agents never call
one another and each records one structured handoff before returning.

The MCP profile contains the unchanged 46-tool synchronous surface (35
scientific facades and 11 lifecycle/reporting operations) plus seven additive
durable-operation tools for short-timeout clients such as Claude Science.
Codex agents keep their existing 46-tool role contract and do not use the
start/poll facade. Training and Inference have no Registry tools in their agent
allowlists; a publishable Training result creates a deterministic Registry
barrier until its exact persistence plan has been materialized. An explicit
user request may opt out with the experiment policy `session_only`.
Simple catalog and experiment reads are performed directly and do not create
an empty experiment. Report runs at most once, after scientific execution.

Scientific execution and code maintenance are separate modes. If a project
agent has a technical orchestration failure, the coordinator may make one
bounded attempt to complete the same mission with the role's existing MCP
tools. It may inspect evidence but cannot patch source, rewrite artifacts,
fabricate filesystem prerequisites, change dependencies, or reinstall the
plugin. If the manual recovery fails, it stops and offers a sanitized draft
for the repository's GitHub Issues tab. Publishing that issue and entering
code-maintenance mode each require their own explicit user approval.

Experiment state and reports use the configured storage backend. The existing
Path-based scientific toolkits keep their local artifact layout under
`.files/sessions/<experiment_id>/...`; when metadata is backed by S3, this does
not mean those toolkits write scientific files directly to an S3 URI.

## Storage and input boundaries

The auto-approved MCP profile is not a general filesystem reader. Local input
paths are restricted to the task workspace and the configured Qsaria storage
root. A deployment can explicitly add trusted local roots with the
path-separator-delimited `QSARIA_MCP_ALLOWED_INPUT_ROOTS` environment variable.
S3 input URLs are accepted at the trust boundary only when they target the
configured bucket. Authorization at that boundary does not guarantee that an
existing Path-based scientific toolkit supports an S3 URI as its input type.

For V1, S3-backed experiment-state and report persistence requires
`QSARIA_S3_SINGLE_WRITER=true`. This variable is a deployment commitment that
only one Qsaria state writer is active; it is not a lock. V1 adds neither a
distributed S3 lock nor a database. `qsaria_bootstrap` reports the effective
storage-concurrency mode and whether that commitment was acknowledged.

Scientific artifacts produced by the existing Path-based toolkits remain
local in the experiment workflow layout even when state/report metadata use
S3. Codex must preserve the literal artifact references returned by tools and
must not present a local artifact as uploaded to S3.

Training checkpoints remain under `.files` as experiment evidence. Registry
copies reusable models and their catalog artifacts into
`data/model_assets/internal`; it never moves or deletes the session evidence.
Report receives the complete structured reporting material, then selects zero
to three useful tables without changing any displayed scientific value.

## Install for the repository pilot

Run these commands from the Qsaria repository root:

```sh
uv sync --extra mcp --extra prediction
python plugins/qsaria-codex/scripts/preflight.py
codex plugin marketplace add .
codex plugin add qsaria-codex@personal
```

The default preflight is deliberately stricter than an MCP transport check: it
imports Chemprop, LightGBM, and TabICL in the exact environment used by the
plugin. A successful MCP bootstrap does not by itself mean that scientific
training is available.

For a read-only catalog or experiment-inspection deployment that intentionally
does not install training dependencies, use the narrower check explicitly:

```sh
uv sync --extra mcp
python plugins/qsaria-codex/scripts/preflight.py --mcp-only
```

Do not use `--mcp-only` to qualify an environment for curation-to-training
workflows.

Install the project agents separately after reviewing the destination:

```sh
python plugins/qsaria-codex/scripts/install_project_agents.py
python plugins/qsaria-codex/scripts/install_project_agents.py --apply
python plugins/qsaria-codex/scripts/install_project_agents.py --check
```

Merge the Qsaria agent registrations from
`plugins/qsaria-codex/assets/project-config.toml` into `.codex/config.toml`.
This sets the supported project-level limits (`max_depth = 1` and
`max_threads = 5`); the installer intentionally does not rewrite project
configuration.

Start a new Codex task after installation or any plugin update. The MCP
launcher intentionally inherits the task working directory, so open the task
at the Qsaria repository root.

## Example requests

```text
Entraîne un modèle de régression Qsaria à partir de solubility.csv.
La colonne moléculaire est canonical_smiles et la cible est logS.
Donne-moi une analyse critique.
```

Catalog persistence is the default for publishable training results. Add
`Ne persiste pas les modèles ; conserve uniquement les artefacts de session.`
when a deliberately session-only experiment is desired.

```text
Utilise le modèle qsaria-model-… pour prédire new_compounds.csv et signale
les prédictions peu fiables.
```

```text
Rouvre l'expérience exp_… et explique son échec d'évaluation externe.
Ne relance aucun entraînement sans me le proposer.
```

```text
Quels modèles Qsaria persistés conviennent à une régression de solubilité ?
```

New tasks do not resume the latest experiment automatically. Persisted models
remain globally discoverable, and an experiment is reopened only by explicit
identifier.

## Validate and synchronize

```sh
python plugins/qsaria-codex/scripts/validate_bundle.py
python plugins/qsaria-codex/scripts/sync_qsaria_codex.py --check
python plugins/qsaria-codex/scripts/smoke_codex_plugin.py --require-codex
```

When Qsaria prompts, toolkit interfaces, reporting facts, or MCP tool specs
change, review the Codex instructions manually and then record the new source
signatures:

```sh
python plugins/qsaria-codex/scripts/sync_qsaria_codex.py --update
```

The update command records compatibility metadata. It never converts Agno
prompts into Codex prompts and never changes the Agno runtime.

Compatibility provenance covers the exact signed Qsaria source tree, its
latest source commit and Git tree, and whether those signed files are dirty.
The signed tree includes the plugin bundle, MCP/scientific contracts, active
project-agent registrations, the repo marketplace entry, and its CI guard.
`compatibility.json` is deliberately excluded from that tree, so the viable
release sequence is: commit reviewed source changes, run `--update`, validate,
then commit the metadata-only refresh. The check remains stable after that
metadata commit and detects later source or dirty-state drift. CI fetches full
Git history so it can still resolve the preceding source commit after a
metadata-only commit.

The smoke script sets both `HOME` and `CODEX_HOME` to a temporary directory,
then exercises marketplace add, plugin add, plugin list, and MCP list. It never
installs into the user's real Codex home.

For a local plugin release after that review, generate one Codex cachebuster,
refresh compatibility metadata, reinstall from the personal marketplace, and
start a new Codex task:

```sh
python plugins/qsaria-codex/scripts/update_plugin_cachebuster.py
python plugins/qsaria-codex/scripts/sync_qsaria_codex.py --update
python plugins/qsaria-codex/scripts/validate_bundle.py
codex plugin add qsaria-codex@personal
```

The cachebuster preserves the semantic version before `+` and replaces any
older `+codex.…` suffix. It is only a local Codex cache invalidation token;
published Qsaria releases should still choose their semantic version
explicitly.
