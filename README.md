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

Quantitative structure–activity/property relationship (QSAR/QSPR) modeling is
one of the central tasks in computer-aided molecular design, with applications
ranging from materials to drug discovery [1,2]. QSAR models plays a key role in
virtual screening, where they act as computational filters for prioritizing the
most promising candidates from large chemical libraries for experimental
evaluation. They are also used as scoring functions in de novo molecular
design, guiding generative models toward compounds with desired property
profiles [3]. The development of such models has therefore been a long-standing
area of research in chemoinformatics [1], supported by established best
practices [4,5] and extensive benchmarking of various machine learning (ML)
algorithms and modeling frameworks.

QSAR model development requires multiple steps, including data pre-processing,
feature generation and selection, model fitting, validation, interpretation,
and deployment [5]. With the rapid accumulation of chemical structure and
molecular property data, the development of automated approaches for QSAR
modeling has become an increasingly important task [6,7], because the modeling
must often be repeated for many endpoints and updated as new data become
available. Numerous approaches, varying in molecular representations, available
ML model families, and the possibility of automated pipeline optimization, have
demonstrated that best-practice QSAR modeling can be organized into reproducible
computational pipelines that perform comparably to expert-driven QSAR
development while being more scalable [8–12].

Although automated QSAR workflows allow one to improve the reproducibility and
efficiency of molecular property modeling, they remain limited when
task-specific modeling decisions are required [13]. These decisions are
difficult to encode exhaustively in fixed automated pipelines. Recent progress
in large language models has opened a complementary direction in which large
language model (LLM)-based agents serve as orchestration and decision-support
layers for scientific ML workflows. General-purpose research and
machine-learning agents, such as AIDE [15] and the AI Scientist [14], illustrate
the potential of this paradigm for iterative code generation, experiment
execution, and machine-learning workflow optimization. However, general agents
do not, by default, incorporate the chemistry-specific practices required for
reliable QSAR modeling, including chemical structure standardization,
applicability domain (AD) assessment, chemistry-specific validation (e.g.
scaffold-based), and interpretation of outliers.

This limitation has motivated the development of chemistry-specific agentic
systems [16]. For example, DrugAgent extends a general purpose MLAgentBench [17]
agentic ML framework toward drug discovery tasks by introducing domain knowledge
identification, tool preparation, and iterative exploration of modeling
approaches. ChemLINT focuses on molecular data curation, providing deterministic
tools for data exploration, molecular standardization, and baseline molecular
ML modeling [18]. MolAgent [13] represents one of the most comprehensive agentic
frameworks for molecular property prediction, combining the automated QSAR
workflows with LLM-accessible orchestration and supporting end-to-end workflows
for featurization, model construction, validation, and deployment.

Qsaria is a new end-to-end agentic AI framework for QSAR modeling. It is built
following the methodology implemented in the ChemSpace Copilot agentic AI
framework [19] and is organized as a hierarchical multi-agent system (MAS), in
which ML algorithms from multiple model families are coordinated. Qsaria is
composed of five specialized agents and a team manager, enabling natural
language requests to be converted into complete QSAR workflows that include
chemical data curation, molecular representation, model training, validation,
inference, and reporting.

Qsaria is an independent module of ChemSpace Copilot. It can be used through its native web application or through dedicated MCP-based integrations for Codex, Claude Code, and Claude Science.

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

## References

1. Cherkasov, A. et al. QSAR Modeling: Where Have You Been? Where Are You Going To? J. Med. Chem. 57, 4977–5010 (2014).
2. Muratov, E. N. et al. QSAR without borders. Chem. Soc. Rev. 49, 3525–3564 (2020).
3. Loeffler, H. H. et al. Reinvent 4: Modern AI–driven generative molecule design. J Cheminform 16, 20 (2024).
4. OECD. Guidance Document on the Validation of (Quantitative) Structure-Activity Relationship [(Q)SAR] Models. OECD Series on Testing and Assessment https://doi.org/10.1787/9789264085442-en (2014) doi:10.1787/9789264085442-en.
5. Tropsha, A. Best Practices for QSAR Model Development, Validation, and Exploitation. Mol. Inf. 29, 476–488 (2010).
6. Gedeck, P. et al. Automated QSAR — how good is it in practice? https://doi.org/10.26434/chemrxiv-2026-l1d11 (2026) doi:10.26434/chemrxiv-2026-l1d11.
7. de Oliveira, M. T. & Katekawa, E. On the Virtues of Automated Quantitative Structure–Activity Relationship: The New Kid on the Block. Future Medicinal Chemistry 10, 335–342 (2018).
8. Sá, A. G. C. de & Ascher, D. B. Auto-ADMET: An Effective and Interpretable AutoML Method for Chemical ADMET Property Prediction. Preprint at https://doi.org/10.48550/arXiv.2502.16378 (2025).
9. Dixon, S. L. et al. Autoqsar: An Automated Machine Learning Tool for Best-Practice Quantitative Structure–Activity Relationship Modeling. Future Medicinal Chemistry 8, 1825–1839 (2016).
10. Mervin, L., Voronov, A., Kabeshov, M. & Engkvist, O. QSARtuna: An Automated QSAR Modeling Platform for Molecular Property Prediction in Drug Design. J. Chem. Inf. Model. 64, 5365–5374 (2024).
11. Kausar, S. & Falcao, A. O. An automated framework for QSAR model building. J Cheminform 10, 1 (2018).
12. Gao, Z. et al. Uni-QSAR: an Auto-ML Tool for Molecular Property Prediction. Preprint at https://doi.org/10.48550/arXiv.2304.12239 (2023).
13. Gómez-Tamayo, J. C. et al. MolAgent: Biomolecular Property Estimation in the Agentic Era. J. Chem. Inf. Model. 65, 10808–10818 (2025).
14. Lu, C. et al. Towards end-to-end automation of AI research. Nature 651, 914–919 (2026).
15. Jiang, Z. et al. AIDE: AI-Driven Exploration in the Space of Code. arXiv.org https://arxiv.org/abs/2502.13138v1 (2025).
16. Liu, S. et al. DrugAgent: Automating AI-aided Drug Discovery Programming through LLM Multi-Agent Collaboration. Preprint at https://doi.org/10.48550/arXiv.2411.15692 (2025).
17. Huang, Q., Vora, J., Liang, P. & Leskovec, J. MLAgentBench: evaluating language agents on machine learning experimentation. in Proceedings of the 41st International Conference on Machine Learning vol. 235 20271–20309 (JMLR.org, Vienna, Austria, 2024).
18. van Tilborg, D. & Grisoni, F. ChemLint: Conversational Cheminformaticswith Large Language Models. https://doi.org/10.26434/chemrxiv.15000386/v1 (2026) doi:10.26434/chemrxiv.15000386/v1.
19. Orlov, A. A., Volkov, M., Milova, E. S., Horvath, D. & Varnek, A. ChemSpace Copilot: Agentic AI for Interactive Visualization and Exploration of Chemical Space. ChemRxiv 2026, (2026).

## License

Qsaria is distributed under the [MIT License](LICENSE).
