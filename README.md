<h1 align="center">
  <img src="docs/qsaria_logo.jpg" alt="Qsaria" width="620">
</h1>

<p align="center">
  <strong>An agentic AI for end-to-end QSAR Modeling</strong>
</p>

<p align="center">
  An independent module within
  <a href="https://github.com/Laboratoire-de-Chemoinformatique/chemspacecopilot">ChemSpace Copilot</a>,
  developed by the
  <a href="https://github.com/Laboratoire-de-Chemoinformatique">Laboratoire de Chémoinformatique de Strasbourg</a>.
</p>

<p align="center">
  <a href="https://github.com/abrahambrisset/Qsaria/actions/workflows/ci.yml"><img src="https://github.com/abrahambrisset/Qsaria/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/abrahambrisset/Qsaria/commits/main"><img src="https://img.shields.io/github/last-commit/abrahambrisset/Qsaria" alt="Last commit"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB" alt="Python 3.11 or 3.12"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/abrahambrisset/Qsaria" alt="MIT License"></a>
  <a href="https://github.com/abrahambrisset/Qsaria/issues"><img src="https://img.shields.io/github/issues/abrahambrisset/Qsaria" alt="Issues"></a>
</p>

> [!WARNING]
> Qsaria is approaching its first public pre-release and remains under active development.

## Overview

Qsaria is an agentic research platform for building, evaluating, and using
Quantitative Structure-Activity Relationship (QSAR) models from end to end.
Starting from a molecular dataset, it can prepare the data, train suitable
models, compare their performance, assess their reliability, persist the best
results, and produce a structured scientific report.

Qsaria is an independent module of ChemSpace Copilot. It can be used through
its native web application or through dedicated MCP-based integrations for
Codex, Claude Code, and Claude Science.

## What Qsaria can do

- **Curate molecular datasets** by standardizing structures, validating targets, handling invalid compounds, and resolving duplicate measurements.
- **Train QSAR models** with LightGBM, Chemprop, and TabICL.
- **Support regression and classification workflows** through backend-appropriate scientific contracts.
- **Compare models and representations** with reproducible benchmark workflows.
- **Evaluate robustness** through multiple validation strategies, external evaluation, outlier analysis, Activity Cliffs, and applicability-domain analysis.
- **Persist and reuse models** through a local registry and model catalog.
- **Run inference and ensembles** on new molecular datasets.
- **Produce evidence-first reports** containing the results, figures, tables, warnings, and scientific context needed to interpret an experiment.

## Specialized workflow

Qsaria organizes scientific work between five specialists:

| Specialist | Role |
| --- | --- |
| **Curation** | Prepares a reliable modelling dataset. |
| **Training** | Trains, validates, tunes, and benchmarks models. |
| **Registry** | Persists models, manages the catalog, and builds ensembles. |
| **Inference** | Runs predictions and external evaluations. |
| **Report** | Produces the final scientific report from the experiment evidence. |

The same scientific toolkits and artifacts are used across the native
application and the external integrations.

## Scientific capabilities

### Backends

| Backend | Main approach |
| --- | --- |
| **LightGBM** | Tabular modelling with molecular descriptors, fingerprints, or precomputed features. |
| **Chemprop** | Message-passing neural networks operating on molecular graphs. |
| **TabICL** | Tabular in-context learning for supported molecular datasets. |

Qsaria 0.4.0 centralizes every LightGBM and TabICL representation in one
strict, cached preparation service. New tabular models persist the complete
recipe and can regenerate features from a new SMILES-only CSV in another
session. Pre-0.4.0 LightGBM and TabICL models without this contract are
intentionally rejected and must be retrained; Chemprop is unaffected.

### Validation

Qsaria provides a standard QSAR workflow through `standard_qsar` and supports
explicit strategies for:

- holdout validation;
- repeated holdout;
- cross-validation;
- full-data training;
- random, scaffold, and cluster-based splitting where supported.

Training requests use strict typed contracts so unsupported or inconsistent
configurations are rejected before an experiment is executed.

### Scientific analysis

Depending on the model, task, and available data, Qsaria can produce:

- performance metrics and diagnostic plots;
- hyperparameter optimization studies;
- outlier analyses;
- Activity Cliff analyses;
- applicability-domain estimates;
- external evaluation results;
- model and representation benchmarks;
- ensemble predictions;
- reusable model bundles and structured reports.

## Ways to use Qsaria

| Interface | Experience |
| --- | --- |
| **Agno + Chainlit** | The native Qsaria web application for development, validation, and interactive research. |
| **Codex** | A Qsaria plugin with specialized agents and MCP tools available directly in Codex. |
| **Claude Code** | A project plugin for running Qsaria workflows from Claude Code. |
| **Claude Science** | A scientific integration with dedicated skills and support for long-running operations. |

Users interact with Qsaria in natural language. The coordinator selects the
appropriate specialists and scientific tools for the requested mission.

## Quick start

### Requirements

- Python 3.11 or 3.12;
- [uv](https://docs.astral.sh/uv/);
- Git;
- an LLM provider for the native Agno/Chainlit application;
- sufficient compute for the selected modelling backend.

Clone the repository and install Qsaria:

```bash
git clone https://github.com/abrahambrisset/Qsaria.git
cd Qsaria
uv sync --extra prediction --extra mcp
```

Prepare the environment:

```bash
cp .env.example .env
uv run chainlit create-secret
```

Configure the generated Chainlit secret and your model provider in `.env`,
then start the native Qsaria team:

```bash
export CS_COPILOT_AGENT_TEAM=qsar
uv run chainlit run chainlit_app.py -w
```

Open [http://localhost:8000](http://localhost:8000).

Some Python modules and commands still use the historical `cs_copilot` and
`cscopilot-*` names for compatibility with ChemSpace Copilot.

## Containers

### Docker Compose

The interactive launcher configures and starts the application:

```bash
docker compose build chainlit-app
./docker-start.sh
```

The first command builds the application image. The launcher then detects the
available hardware, selects the application port, and guides the user through
the main runtime options.

### Apple container

On Apple silicon with Apple's
[`container`](https://github.com/apple/container) CLI installed:

```bash
scripts/apple-container.sh
```

The script builds the image and launches the application. After the first
successful build, start it again without rebuilding with:

```bash
APPLE_CONTAINER_SKIP_BUILD=1 scripts/apple-container.sh
```

To force a clean rebuild before launching:

```bash
APPLE_CONTAINER_NO_CACHE=1 scripts/apple-container.sh
```

### Apptainer

```bash
apptainer build chemspacecopilot.sif scripts/chemspacecopilot.def
AGENT_TEAM=qsar scripts/run_apptainer.sh
```

## Codex integration

Install the scientific environment and the local Qsaria plugin:

```bash
uv sync --extra mcp --extra prediction
python plugins/qsaria-codex/scripts/preflight.py
codex plugin marketplace add .
codex plugin add qsaria-codex@personal
```

Install the Qsaria project agents:

```bash
python plugins/qsaria-codex/scripts/install_project_agents.py
python plugins/qsaria-codex/scripts/install_project_agents.py --apply
python plugins/qsaria-codex/scripts/install_project_agents.py --check
```

The first command previews the planned changes without writing them. Then
merge the agent registrations from
`plugins/qsaria-codex/assets/project-config.toml` into `.codex/config.toml` and
open a new Codex task from the repository root.

See [Qsaria for Codex](plugins/qsaria-codex/README.md) for the full guide.

## Claude Code integration

```bash
uv sync --extra mcp --extra prediction
uv run --no-sync python plugins/qsaria-claude-code/scripts/preflight.py
claude plugin marketplace add .
claude plugin install qsaria-claude-code@personal --scope project
```

Run Claude Code from the repository root. After installation, run
`/reload-plugins`, open `/mcp`, and wait until the `qsaria` server is reported
as connected before starting a scientific request.

See [Qsaria for Claude Code](plugins/qsaria-claude-code/README.md) for the full guide.

## Claude Science integration

Claude Science 0.1.21 or later is required. Prepare the project, then run the
repository preflight and bundle validation:

```bash
uv sync --extra mcp --extra prediction
.venv/bin/python plugins/qsaria-claude-science/scripts/preflight.py
.venv/bin/python plugins/qsaria-claude-science/scripts/validate_science_bundle.py
```

The preflight provides the connector command, folder permissions, and local
paths required by the current installation. In Claude Science:

1. grant the exact repository, `.files`, `data`, artifact, and Python runtime permissions printed by preflight;
2. restart Claude Science, then add and connect a local connector named `qsaria` with the following command;
3. import the seven Qsaria Science skills from `plugins/qsaria-claude-science/skills/`;
4. invoke `qsaria-science-setup` once to create the coordinator and specialists.

```text
Command: /bin/zsh
Argument: <repo>/plugins/qsaria-claude-science/scripts/launch-mcp.sh
Environment:
  QSARIA_SCIENCE_ARTIFACT_ROOT=<artifact root printed by preflight>
```

Before the first long workflow, run the durability test documented by the
bundle:

```bash
.venv/bin/python plugins/qsaria-claude-science/scripts/durability_probe.py start \
  --root .files --seconds 65
```

Disconnect or restart the connector, wait at least 67 seconds, reconnect, and
check the state path returned by the previous command:

```bash
.venv/bin/python plugins/qsaria-claude-science/scripts/durability_probe.py check \
  --state <returned-state-path>
```

Continue with long-running experiments only when the probe reports
`completed`.

See [Qsaria for Claude Science](plugins/qsaria-claude-science/README.md) for the
complete setup, skills, and durability checks.

## Example requests

### Train a model

```text
Train a standard LightGBM regression model from solubility.csv.
The molecular column is canonical_smiles and the target is logS.
Persist the model and give me a critical analysis of the results.
```

### Compare approaches

```text
Compare the Qsaria approaches suited to this classification dataset and
explain the strengths and limitations of the best models.
```

### Predict new compounds

```text
Use model qsaria-model-… to predict new_compounds.csv and flag unreliable
predictions.
```

### Inspect available models

```text
Which persisted Qsaria models are available for a solubility regression task?
```

## Experiments and model persistence

Each scientific workflow receives an `experiment_id`. Qsaria keeps the
experiment evidence and reusable models in separate locations:

```text
.files/sessions/<experiment_id>/   experiment artifacts and reports
data/model_assets/internal/        reusable persisted models
data/model_assets/catalog/         local model catalog
```

A new conversation does not automatically resume an earlier experiment, but
persisted models remain available. An experiment can be reopened explicitly
with its identifier.

The model catalog belongs to the local user environment and is not distributed
as part of the repository.

## Development

Install the complete environment:

```bash
uv sync --extra mcp --extra prediction
```

Run the main repository checks:

```bash
uv run black --check src/ tests/
uv run ruff check src/ tests/
uv run pytest tests/unit/ -v --tb=short
```

Each integration contains its own preflight, validation, synchronization, and
installation documentation under `plugins/`.

## Project status

Qsaria is under active development. Before the first stable release, APIs,
scientific contracts, and installation procedures may still evolve. Chemprop
and TabICL workflows can also require significant compute depending on the
dataset and requested analysis.

For bugs and feature requests, use the
[GitHub Issues](https://github.com/abrahambrisset/Qsaria/issues) tab. When
reporting a failed workflow, include its sanitized experiment identifier,
backend, structured error, and relevant artifact identifiers without sharing
private data or credentials.

## Citation

If you use Qsaria, please cite the
[Qsaria abstract and poster](https://infochim.u-strasbg.fr/IMG/pdf/p2_abraham_brisset_abstract_poster_cs3-2026.pdf).

Related work:
[ChemSpace Copilot](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000527/v2).

## License

Qsaria is distributed under the [MIT License](LICENSE).
