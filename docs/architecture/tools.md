# Tools System

Tools are organized as **Toolkit classes** that inherit from `Toolkit` (Agno framework).

**Location**: `src/cs_copilot/tools/`

## Directory Structure

```
tools/
├── databases/          Database integrations
│   ├── base.py        BaseDatabaseToolkit (abstract)
│   ├── chembl.py      ChemblToolkit (REST API + MySQL backends)
│   ├── chembl_fetcher.py  RestChemblFetcher / SqlChemblFetcher strategies
│   └── types.py       Query types and configurations
│
├── chemography/       Dimensionality reduction
│   ├── gtm.py         GTMToolkit (high-level interface)
│   └── gtm_operations.py  Core GTM implementations
│
├── chemistry/         Molecular operations
│   ├── similarity_toolkit.py      Similarity calculations
│   ├── autoencoder_toolkit.py     LSTM autoencoder operations
│   └── descriptors.py             Molecular descriptors
│
├── prediction/        Predictive modeling backends and toolkits
│   ├── backend.py                 Backend contract for pluggable predictors
│   ├── backend_factory.py         Shared backend construction
│   ├── tabular_representations.py Canonical tabular representation registry
│   ├── tabular_feature_preparation.py Shared preparation, contract, and cache
│   ├── qsar_training_toolkit.py   Public QSAR training facade
│   ├── model_registry_toolkit.py  Public model registry and catalog facade
│   ├── prediction_inference_toolkit.py  Public inference facade
│   ├── benchmark_toolkit.py       Explicit benchmark campaign orchestration
│   ├── chemprop_toolkit.py        Internal Chemprop training toolkit
│   ├── lightgbm_toolkit.py        Internal LightGBM training toolkit
│   ├── tabicl_toolkit.py          Internal TabICL training toolkit
│   └── *_backend.py               Backend adapters
│
├── io/                I/O and formatting
│   ├── pointer_pandas_tools.py   DataFrame ops + S3 integration
│   └── formatting.py              SMILES → images, markdown
│
└── constants.py       Configuration constants
```

Agent-facing toolkits register their public methods via `self.register(method)`.
Backend training engines and feature generators remain Python-internal and are
reached through `QSARTrainingToolkit`, not exposed directly to agents.

## QSAR Tabular Features

No workflow or backend maintains its own representation recipe.
`tabular_representations.py` declares immutable RDKit, Morgan binary, Morgan
count, and precomputed components. `TabularFeaturePreparationService` assembles
and caches them once for Training, Inference, External Evaluation, Ensemble,
Applicability Domain, and any future tabular backend.
`MolecularFeatureToolkit` remains the low-level calculation engine; LightGBM and
TabICL receive only matrices whose columns have already been validated and
ordered.

## ChEMBL Backends

The `ChemblToolkit` supports two pluggable data backends via a strategy pattern (`chembl_fetcher.py`):

| Backend | Trigger | Dependency | Use case |
|---------|---------|------------|----------|
| **REST API** | Default (no config needed) | `chembl_webresource_client` (included) | Quick setup, always-on access |
| **MySQL** | Set `CHEMBL_MYSQL_HOST` env var | `pymysql` (included in `uv sync`) | Faster queries, offline use, full SQL |

Backend is auto-detected: MySQL when `CHEMBL_MYSQL_HOST` is present, REST otherwise. The REST API is always reported as available regardless of active backend.

Download the MySQL dump from the [ChEMBL downloads page](https://chembl.gitbook.io/chembl-interface-documentation/downloads) or the [EBI FTP](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/).

## Optional SynPlanner Backend

`SynPlannerToolkit` is part of the codebase, but the external `SynPlanner` package is an optional dependency because its `cgrtools-stable` dependency only ships wheels for selected platforms. Install it on supported systems with:

```bash
uv sync --extra synplanner
```

Without that extra, the SynPlanner agent/toolkit raises a clear install error when its backend is used; the rest of the ChemSpace tool stack and MCP server remain installable.

## Adding a New Tool

1. Create a toolkit in `src/cs_copilot/tools/`:

```python
from agno import Toolkit

class MyNewToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="my_new_toolkit")
        self.register(self.my_tool_function)

    def my_tool_function(self, param: str) -> str:
        """Tool description for LLM."""
        return f"Result: {param}"
```

2. Import and pass to the agent factory's `tools` parameter
