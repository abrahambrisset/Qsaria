# QSARIA Guide

This document explains how to use QSARIA day to day: architecture, launch
commands for the available containers, LLM model configuration, QSAR workflows,
prompt examples, generated files, and troubleshooting.

Project name: the current GitHub repository is `QSARIA`. Some historical code
paths and files still use the old names `ChemSpace Copilot` or `cs_copilot`.

## 1. Overview

QSARIA is a Chainlit application exposing a multi-agent system for
cheminformatics and QSAR workflows.

The QSAR subsystem is separated from the rest of the app. It handles:

- QSAR dataset curation;
- Chemprop, LightGBM, and TabICL model training;
- explicit benchmarking;
- model registry and catalog persistence;
- inference with persisted models;
- consensus ensembles;
- standardized QSAR reports;
- applicability domain analysis;
- activity cliffs with SALI;
- LaTeX export through `@latex`.

The user entry point is the Chainlit chat.

## 2. Functional Architecture

Simplified architecture:

```text
Chainlit UI
  |
  |-- main router
      |
      |-- general team
      |     |-- ChEMBL downloader
      |     |-- GTM
      |     |-- chemoinformatician
      |     |-- report generator
      |     |-- autoencoder
      |     |-- peptide WAE
      |     |-- retrosynthesis
      |
      |-- QSAR team
            |-- dataset_curation_agent
            |-- qsar_training_agent
            |-- model_registry_agent
            |-- model_inference_agent
            |-- qsar_report_agent
```

QSAR architecture:

```text
QSAR agents
  |
  |-- DatasetCurationToolkit
  |-- QSARTrainingToolkit
  |     |-- MolecularFeatureToolkit (internal)
  |     |-- ActivityCliffToolkit (internal)
  |     |-- ChempropToolkit -> ChempropBackend (internal)
  |     |-- LightGBMToolkit -> LightGBMBackend (internal)
  |     |-- TabICLToolkit   -> TabICLBackend (internal)
  |-- ModelRegistryToolkit
  |-- PredictionInferenceToolkit
  |-- BenchmarkToolkit
  |-- EnsembleToolkit
  |-- qsar_report_agent
```

Important rule: agents should not call internal backend engines or feature
tools directly. They should go through public facades, especially
`QSARTrainingToolkit` for training.

## 3. LLM Model Configuration

Configuration is handled through environment variables or `.modelconf`.

Priority:

```text
environment variables > .modelconf > defaults
```

Supported providers:

- `deepseek`
- `openrouter`
- `ollama`

Main variables:

```bash
MODEL_PROVIDER=deepseek|openrouter|ollama
MODEL_ID=...
MODEL_MAX_TOKENS=8192
DEEPSEEK_API_KEY=...
OPENROUTER_API_KEY=...
OLLAMA_HOST=http://localhost:11434
CS_COPILOT_AGENT_TEAM=qsar
```

OpenRouter example:

```bash
export MODEL_PROVIDER=openrouter
export MODEL_ID="deepseek/deepseek-v4-flash"
export OPENROUTER_API_KEY="..."
export MODEL_MAX_TOKENS=8192
```

Direct DeepSeek example:

```bash
export MODEL_PROVIDER=deepseek
export MODEL_ID="deepseek-chat"
export DEEPSEEK_API_KEY="..."
```

Local Ollama example:

```bash
export MODEL_PROVIDER=ollama
export MODEL_ID="qwen2.5:72b"
export OLLAMA_HOST="http://localhost:11434"
```

## 4. Launch QSARIA

### 4.1 Docker Compose

Standard option for Linux, macOS, or Windows with Docker Desktop:

```bash
docker compose build chainlit-app
./docker-start.sh
```

The script selects a free port between `8000` and `8010`.

Access:

```text
http://localhost:8000
```

Direct production mode:

```bash
docker compose up -d
docker compose logs -f chainlit-app
```

Development mode with hot reload:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Stop:

```bash
docker compose down
```

Full Docker reset:

```bash
docker compose down -v --remove-orphans
docker compose build chainlit-app
```

OpenRouter note with Docker: `docker-start.sh` is historically oriented toward
interactive DeepSeek key entry. For OpenRouter, prefer a `.env` file or
exported variables before running `docker compose up`.

Example:

```bash
export MODEL_PROVIDER=openrouter
export MODEL_ID="deepseek/deepseek-v4-flash"
export OPENROUTER_API_KEY="..."
export CS_COPILOT_AGENT_TEAM=qsar
export MODEL_MAX_TOKENS=4096
docker compose up -d
```

### 4.2 Apple Container

Requirement: Apple Container CLI installed.

Launch with build:

```bash
export MODEL_PROVIDER=openrouter
export MODEL_MAX_TOKENS=4096
export MODEL_ID="deepseek/deepseek-v4-flash"
export OPENROUTER_API_KEY="..."
scripts/apple-container.sh
```

Restart without rebuild if the image already exists:

```bash
export APPLE_CONTAINER_SKIP_BUILD=1
scripts/apple-container.sh
```

Set CPU/RAM:

```bash
export APPLE_CONTAINER_CPUS=8
export APPLE_CONTAINER_MEMORY=12G
scripts/apple-container.sh
```

Logs:

```bash
container logs cs-copilot-apple
```

Stop:

```bash
container stop cs-copilot-apple
```

Apple Container mounts several repository folders into `/app`, including:

- `src`
- `public`
- `examples`
- `.files`
- `data`
- `models`
- `chainlit_app.py`

This allows many Python changes to be tested without rebuilding the image.
A rebuild is still required when dependencies or the Dockerfile change.

### 4.3 Apptainer

Recommended option for Linux GPU machines.

Build the image:

```bash
apptainer build chemspacecopilot.sif scripts/chemspacecopilot.def
```

Launch with DeepSeek:

```bash
export DEEPSEEK_API_KEY="..."
scripts/run_deepseek.sh
```

Launch with OpenRouter:

```bash
export OPENROUTER_API_KEY="..."
export MODEL_ID="deepseek/deepseek-v4-flash"
export MODEL_MAX_TOKENS=4096
scripts/run_openrouter.sh
```

Manual launch:

```bash
export MODEL_PROVIDER=openrouter
export MODEL_ID="deepseek/deepseek-v4-flash"
export OPENROUTER_API_KEY="..."
export MODEL_MAX_TOKENS=4096
scripts/run_apptainer.sh
```

`run_apptainer.sh` mounts the repository into `/app` and runs:

```text
chainlit run chainlit_app.py --host 0.0.0.0 --port 8000
```

It uses `apptainer exec --nv`, so the NVIDIA GPU is exposed when available.

## 5. Storage and Generated Files

By default, local artifacts are written under:

```text
.files/
```

Persisted models are written under:

```text
data/model_assets/internal/
```

External checkpoints can live under:

```text
data/model_assets/checkpoints/
```

Typical QSAR training outputs:

- curated dataset;
- curation report JSON;
- tabular feature files for LightGBM or TabICL;
- training summary JSON;
- test predictions;
- plots;
- applicability domain artifacts;
- activity cliffs;
- complete zip bundle;
- persisted catalog models.

## 6. Routing Shortcuts

Supported shortcuts at the beginning of a message:

```text
@qsaria
@qsar
@latex
```

Examples:

```text
@qsaria Train a LightGBM standard_qsar model for pEC50.
```

```text
@latex
```

`@qsaria` forces routing to the QSAR team. It does not yet open a UI dropdown;
it is a text token recognized by the backend.

`@latex` exports the latest compatible QSAR report to LaTeX when report state
exists.

## 7. Available QSAR Workflows

### 7.1 Curation

Goal: transform a raw CSV into a QSAR-ready training dataset.

Curation handles:

- SMILES and target column identification;
- ChEMBL standardization;
- parent structure extraction;
- ChEMBL checker diagnostics;
- organometallic removal;
- duplicate removal or aggregation according to QSAR identity;
- numeric target conversion;
- outlier detection;
- curation report generation.

Prompt example:

```text
@qsaria Curate this QSAR dataset. The SMILES column is SMILES and the target is pEC50.
```

### 7.2 Standard Training

`standard_qsar` is the only named protocol. It applies one random 80/10/10
split. Chemprop uses its molecular graph, LightGBM uses `rdkit_all` with 50
Optuna/TPE trials by default, and TabICL uses `rdkit_all`. Post-selection
outlier analysis runs whenever it is applicable.

Prompts:

```text
@qsaria Train a Chemprop model for pEC50.
```

```text
@qsaria Train a QSAR LightGBM standard_qsar model to predict pEC50.
```

```text
@qsaria Train a TabICL standard_qsar model for pEC50.
```

For a general training request, QSARIA should use `standard_qsar`. It should
not invent repeated holdout, scaffold, cluster, or cross-validation unless
explicitly requested.

### 7.3 Custom Holdout

Use this when the user explicitly requests a split family and ratio.

Prompts:

```text
@qsaria Train a LightGBM model for pEC50 with RDKit all. Use a random 60/20/20 split.
```

```text
@qsaria Train a LightGBM model for pEC50 with Morgan count. Use a scaffold 60/20/20 split.
```

```text
@qsaria Train a Chemprop model for pEC50 with a scaffold 70/15/15 split.
```

### 7.4 Repeated Holdout

Use this when the user explicitly requests multiple repetitions.

Prompts:

```text
@qsaria Train a LightGBM model for pEC50 with Morgan binary fingerprints.
Use repeated random holdout with 3 repetitions in 70/15/15.
```

```text
@qsaria Train a Chemprop model for pEC50 with repeated scaffold holdout, 3 repetitions, 70/15/15.
```

Expected behavior:

- a different seed per repetition unless the user asks otherwise;
- one persistable model per repetition;
- individual metrics and mean plus standard deviation aggregation;
- worst repetition considered in governance.

### 7.5 Benchmark

Benchmarking is explicit only. QSARIA must not launch a benchmark when the user
only asks for standard training.

Prompt:

```text
@qsaria Run a QSAR benchmark for pEC50.
```

Standard benchmark:

- Chemprop graph;
- LightGBM on `morgan_only`, `rdkit_all`, `morgan_count_only`;
- TabICL on `morgan_only`, `rdkit_all`, `morgan_count_only`;
- `standard_qsar` when no explicit strategy is provided.

An advanced strategy is requested explicitly and applied to every candidate:

```text
@qsaria Run a QSAR benchmark for pEC50 with 3-repeat random holdout.
```

### 7.6 Inference

Predict with a catalog model:

```text
@qsaria List QSAR catalog models for pEC50.
```

```text
@qsaria Predict pEC50 for this file with the best available catalog model.
```

Predict with a specific model:

```text
@qsaria Predict pEC50 with model pxr_challenge_train_...
```

### 7.7 Ensembles

Create a consensus ensemble from catalog models:

```text
@qsaria Create a QSAR ensemble for pEC50 from the best catalog models.
```

Explicitly evaluate an ensemble:

```text
@qsaria Evaluate this QSAR ensemble on this external dataset.
```

Without an evaluation request, QSARIA should only create or summarize the
ensemble.

## 8. Tabular Representations

Modern automatic representations:

- `morgan_only`
- `rdkit_all`
- `morgan_count_only`

Explicit advanced representations:

- `morgan_binary_count_rdkit_all`
- `morgan_rdkit_all`

Prompts:

```text
@qsaria Train a LightGBM model for pEC50 with Morgan count fingerprints.
```

```text
@qsaria Train a TabICL model for pEC50 with RDKit all.
```

```text
@qsaria Explicitly train LightGBM with morgan_binary_count_rdkit_all for pEC50.
```

## 9. QSAR Backends

### Chemprop

- input: CSV with SMILES and target;
- representation: molecular graph;
- best use: graph neural network model;
- no tabular features;
- uses a minimal Chemprop CSV and native `splits_file`;
- in QSARIA, `num_replicates=1` per split; robustness comes from splits, not
  Chemprop replicates.

### LightGBM

- input: tabular dataset with molecular features;
- representations: Morgan, Morgan count, RDKit all;
- best use: fast and strong baseline;
- uses `n_jobs` according to the compute profile;
- forces CPU by default in this project to avoid wasted time on LightGBM GPU
  attempts when OpenCL is unavailable.

### TabICL

- input: tabular dataset;
- at inference time, a SMILES-only CSV is transformed with exactly the same
  tabular recipe used during training;
- uses the configured TabICL checkpoint;
- more RAM-sensitive than LightGBM;
- should be tested carefully on large datasets and wide representations.

Since Qsaria 0.4.0, every new LightGBM or TabICL model carries a strict
representation contract. Older tabular models without that contract are no
longer usable and must be retrained. Feature caches are reconstructible and are
never a model dependency.

### Ensemble

- prediction-only backend;
- combines catalog models;
- provides simple uncertainty through component disagreement.

## 10. Model Governance

Canonical statuses:

- `experimental`
- `workflow_demo`
- `validated`
- `robust_validated`

Main gates:

- Dataset Gate;
- Hardest Split Gate;
- Robustness Gap Gate;
- Random Stability Gate.

Practical rule:

- if a model is useful but does not pass strong thresholds, it is persisted as
  `workflow_demo`;
- `validated` requires passing gates;
- `robust_validated` requires enough repetitions/splits and passing gates;
- a stronger existing catalog model does not block persisting a new run as
  `workflow_demo`.

## 11. What Is Not Active Currently

Cross-validation and nested cross-validation are not active paths. They were
removed from the current workflow until a clean scikit-learn based
implementation is added.

The currently supported validation vocabulary is:

- `holdout`
- `repeated_holdout`
- `split_family`: `random`, `scaffold`, `cluster`
- `split_sizes`: `[train, validation, test]`
- `n_repeats`
- `selection_metric`

## 12. Useful Prompt Examples

### Inspection

```text
@qsaria Describe the available QSAR backends.
```

```text
@qsaria List QSAR catalog models for pEC50.
```

### Curation

```text
@qsaria Curate the uploaded dataset for a QSAR regression task. The target is pEC50.
```

### Simple Training

```text
@qsaria Train a Chemprop model for pEC50.
```

```text
@qsaria Train a LightGBM standard_qsar model for pEC50.
```

```text
@qsaria Train a TabICL standard_qsar model for pEC50.
```

### Custom Training

```text
@qsaria Train a LightGBM model for pEC50 with RDKit all. Use a random 60/20/20 split.
```

```text
@qsaria Train a Chemprop model for pEC50 with a scaffold 60/20/20 split.
```

```text
@qsaria Train a LightGBM model for pEC50 with Morgan count.
Use repeated random holdout with 3 repetitions in 70/15/15.
```

### Benchmark

```text
@qsaria Run a QSAR benchmark for pEC50.
```

```text
@qsaria Run a QSAR benchmark for pEC50 with 3-repeat random holdout.
```

### Prediction

```text
@qsaria Predict pEC50 for the molecules in this file with the best catalog model.
```

```text
@qsaria Predict pEC50 with model <model_id>.
```

### Ensemble

```text
@qsaria Create a QSAR ensemble for pEC50 with the best catalog models.
```

```text
@qsaria Evaluate this QSAR ensemble on this external dataset.
```

### Export

```text
@latex
```

## 13. Quick Checks After Changes

Python checks:

```bash
python -m py_compile \
  src/cs_copilot/agents/prompts.py \
  src/cs_copilot/tools/prediction/qsar_training_toolkit.py \
  src/cs_copilot/tools/prediction/chemprop_toolkit.py \
  src/cs_copilot/tools/prediction/chemprop_backend.py
```

Targeted QSAR unit tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv run pytest \
  tests/unit/test_prediction_backend.py \
  tests/unit/test_qsar_training_toolkit.py \
  tests/unit/test_qsar_validation_strategy.py \
  tests/unit/test_chemprop_adapter.py \
  -q
```

Git verification:

```bash
git diff --check
git status --short
```

## 14. Troubleshooting

### The report does not appear after a long training run

This can come from the LLM provider or a server refresh.

Check:

```bash
docker compose logs -f chainlit-app
container logs cs-copilot-apple
```

In development mode, hot reload can restart the app during training. For long
training tests, prefer production mode.

### OpenRouter returns "input too long"

The selected OpenRouter model has a context limit lower than the context sent
by the app. Change model or reduce generated context.

Useful variables:

```bash
export MODEL_PROVIDER=openrouter
export MODEL_ID="..."
export MODEL_MAX_TOKENS=8192
```

`MODEL_MAX_TOKENS` limits output, not always input. Input limits depend on the
effective OpenRouter model and provider.

### Chemprop receives a bad CLI argument

Check that logs do not contain internal QSAR arguments such as:

```text
--validation-strategy
```

QSAR strategies must be converted into `chemprop_splits.json`, then passed to
Chemprop through:

```text
--splits-file <path>
```

### LightGBM seems to use only one CPU

Check the compute profile in the report and the effective `n_jobs`.

Current policy:

```text
local_light      n_jobs = min(cpu_count, 8)
local_standard   n_jobs = min(cpu_count, 24)
heavy_validation n_jobs = min(cpu_count, 48)
```

### TabICL uses too much RAM

Start with a simple representation:

```text
rdkit_all
morgan_only
morgan_count_only
```

Avoid wide combined representations on RAM-limited machines.

### The bundle does not contain the expected files

Check that the report lists a global training bundle, not only a curation
bundle. A training bundle should contain at least:

- curated dataset;
- curation report;
- training summary;
- model checkpoint;
- test predictions;
- splits;
- metadata;
- AD;
- activity cliffs if available.

## 15. Common Git Commands

Status:

```bash
git status --short --branch
```

Commit:

```bash
git add <files>
git commit -m "Message"
```

Push current branch:

```bash
git push
```

If SSH needs an explicit key:

```bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_github' git push -u origin validation-strategies
```
