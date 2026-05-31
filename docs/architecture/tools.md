# Tools System

Tools are organized as **Toolkit classes** that inherit from `Toolkit` (Agno framework).

**Location**: `src/cs_copilot/tools/`

## Directory Structure

```
tools/
├── databases/          Database integrations
│   ├── base.py        BaseDatabaseToolkit (abstract)
│   ├── chembl.py      ChemblToolkit (ChEMBL REST API)
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

Each toolkit registers methods as tools via `self.register(method)`. Agents call these tools via the Agno tool-calling mechanism.

## QSAR Tabular Features

`QSARTrainingToolkit` and `BenchmarkToolkit` must not maintain their own
representation lists. They consume `tabular_representations.py`, while
`MolecularFeatureToolkit` generates and caches RDKit all descriptors, Morgan
binary fingerprints, and Morgan count fingerprints for LightGBM/TabICL and any
future tabular backend.

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
