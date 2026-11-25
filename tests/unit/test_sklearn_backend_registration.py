"""The scikit-learn baseline backend must be reachable from the standard registries.

A backend that exists but is not registered cannot be selected from a training or
benchmark request, so these tests pin the three registration points: the factory,
the capability contract, and the tuning catalog.
"""

from __future__ import annotations

import json

import pytest

from cs_copilot.tools.prediction.backend_capabilities import (
    BACKEND_CAPABILITIES,
    get_backend_capabilities,
)
from cs_copilot.tools.prediction.backend_factory import build_default_prediction_backends
from cs_copilot.tools.prediction.hyperparameter_tuning import (
    BACKEND_TUNING_CATALOG,
    describe_backend_hyperparameters,
)
from cs_copilot.tools.prediction.sklearn_backend import (
    REGRESSION_ONLY_ESTIMATORS,
    SKLEARN_ESTIMATORS,
    SklearnBackend,
)
from cs_copilot.tools.prediction.tabular_representations import (
    SUPPORTED_TABULAR_REPRESENTATION_NAMES,
)


def test_sklearn_backend_is_built_by_the_default_factory():
    backends = build_default_prediction_backends()

    assert "sklearn" in backends
    assert isinstance(backends["sklearn"], SklearnBackend)
    assert backends["sklearn"].backend_name == "sklearn"


def test_default_factory_keeps_every_pre_existing_backend():
    backends = build_default_prediction_backends()

    for name in ("chemprop", "lightgbm", "tabicl", "ensemble"):
        assert name in backends


def test_sklearn_capabilities_are_registered_and_honest():
    capability = get_backend_capabilities("sklearn")

    assert capability.backend_name == "sklearn"
    assert capability.can_train is True
    assert capability.can_predict is True
    assert capability.requires_feature_preparation is True
    assert capability.prediction_input_kinds == ("tabular_features_csv",)
    assert capability.supported_representations == SUPPORTED_TABULAR_REPRESENTATION_NAMES
    # The backend rejects more than one target column, so it must not claim
    # multi-target support.
    assert capability.multi_target_task_types == ()
    # No estimator here exports predictive uncertainty.
    assert capability.supports_uncertainty == "none"


def test_declared_task_types_match_what_the_backend_actually_accepts():
    capability = get_backend_capabilities("sklearn")

    assert set(capability.supported_task_types) == {
        "regression",
        "classification",
        "multiclass_classification",
    }


def test_sklearn_capabilities_serialize_alongside_the_other_backends():
    payload = {name: capability.as_dict() for name, capability in BACKEND_CAPABILITIES.items()}

    json.dumps(payload)
    assert payload["sklearn"]["backend_name"] == "sklearn"
    assert payload["sklearn"]["catalog_model_filename"] == "best.pkl"


def test_tuning_catalog_declares_sklearn_as_untuned():
    entry = BACKEND_TUNING_CATALOG["sklearn"]

    assert entry["backend_name"] == "sklearn"
    # Declaring tuning support we do not implement would be a false contract.
    assert entry["supports_hyperparameter_tuning"] is False
    assert entry["supported_engines"] == []
    assert entry["default_engine"] is None
    assert entry["default_trials"] is None


def test_capability_record_points_at_the_catalog_entry():
    capability = get_backend_capabilities("sklearn")

    assert capability.hyperparameter_tuning == BACKEND_TUNING_CATALOG["sklearn"]


def test_describe_backend_hyperparameters_includes_sklearn():
    described = describe_backend_hyperparameters("sklearn")

    assert described["backend_name"] == "sklearn"

    everything = describe_backend_hyperparameters()
    assert "sklearn" in everything


@pytest.mark.parametrize("estimator", sorted(SKLEARN_ESTIMATORS))
def test_every_advertised_estimator_is_known_to_the_backend(estimator):
    # Guards against the catalog and the implementation drifting apart.
    assert estimator in SKLEARN_ESTIMATORS
    if estimator in REGRESSION_ONLY_ESTIMATORS:
        assert estimator == "pls"
