"""Prediction backends and toolkits."""

from .backend import (
    BackendNotAvailableError,
    InvalidPredictionInputError,
    PredictionBackend,
    PredictionBackendError,
    PredictionExecutionError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from .backend_capabilities import (
    BACKEND_CAPABILITIES,
    BackendCapabilities,
    backend_requires_feature_preparation,
    backend_supports_component_orchestration,
    describe_backend_capabilities,
    enrich_backend_environment,
    get_backend_capabilities,
)
from .catalog import CatalogRecommendation, PredictionModelCatalog
from .benchmark_toolkit import BenchmarkToolkit
from .chemprop_backend import ChempropBackend
from .ensemble_backend import EnsembleBackend
from .ensemble_toolkit import EnsembleToolkit
from .lightgbm_backend import LightGBMBackend
from .prediction_inference_toolkit import PredictionInferenceToolkit
from .model_registry_toolkit import ModelRegistryToolkit
from .backend_factory import build_default_prediction_backends
from .qsar_training_toolkit import QSARTrainingToolkit
from .tabicl_backend import TabICLBackend

__all__ = [
    "PredictionBackend",
    "PredictionBackendError",
    "PredictionExecutionError",
    "BackendNotAvailableError",
    "InvalidPredictionInputError",
    "PredictionTaskSpec",
    "PredictionModelRecord",
    "BackendCapabilities",
    "BACKEND_CAPABILITIES",
    "get_backend_capabilities",
    "describe_backend_capabilities",
    "backend_requires_feature_preparation",
    "backend_supports_component_orchestration",
    "enrich_backend_environment",
    "PredictionModelCatalog",
    "CatalogRecommendation",
    "BenchmarkToolkit",
    "ChempropBackend",
    "EnsembleBackend",
    "EnsembleToolkit",
    "LightGBMBackend",
    "PredictionInferenceToolkit",
    "ModelRegistryToolkit",
    "build_default_prediction_backends",
    "QSARTrainingToolkit",
    "TabICLBackend",
]
