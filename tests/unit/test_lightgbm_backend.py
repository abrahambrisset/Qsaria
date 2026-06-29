from __future__ import annotations

import pandas as pd
import pytest

from cs_copilot.tools.prediction.backend import InvalidPredictionInputError, PredictionTaskSpec
import cs_copilot.tools.prediction.lightgbm_backend as lightgbm_backend_module
from cs_copilot.tools.prediction.lightgbm_backend import LightGBMBackend


def test_select_feature_columns_auto_includes_explicit_categorical_columns():
    backend = LightGBMBackend()
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "CCC"],
            "Y": [1.0, 2.0],
            "fp_0001": [0.1, 0.2],
            "series": ["A", "B"],
        }
    )
    task = PredictionTaskSpec(task_type="regression", smiles_columns=["smiles"], target_columns=["Y"])

    feature_columns, categorical_feature_columns = backend._select_feature_columns(
        df,
        task,
        {"categorical_feature_columns": ["series"]},
    )

    assert feature_columns == ["fp_0001", "series"]
    assert categorical_feature_columns == ["series"]


def test_select_feature_columns_rejects_non_numeric_non_categorical_explicit_columns():
    backend = LightGBMBackend()
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "CCC"],
            "Y": [1.0, 2.0],
            "series": ["A", "B"],
        }
    )
    task = PredictionTaskSpec(task_type="regression", smiles_columns=["smiles"], target_columns=["Y"])

    with pytest.raises(InvalidPredictionInputError):
        backend._select_feature_columns(
            df,
            task,
            {"feature_columns": ["series"]},
        )


def test_encode_categorical_frame_preserves_unseen_as_missing_code():
    backend = LightGBMBackend()
    train = pd.DataFrame({"series": ["A", "B", None]})
    encoded_train, mappings = backend._encode_categorical_frame(train, ["series"])

    assert encoded_train["series"].tolist() == [0, 1, -1]

    inference = pd.DataFrame({"series": ["B", "C", None]})
    encoded_inference, _ = backend._encode_categorical_frame(
        inference,
        ["series"],
        category_mappings=mappings,
    )

    assert encoded_inference["series"].tolist() == [1, -1, -1]


def test_lightgbm_defaults_to_cpu_even_when_gpu_is_detected(monkeypatch):
    monkeypatch.setattr(
        lightgbm_backend_module,
        "describe_compute_environment",
        lambda: {"gpu_available": True},
    )

    backend = LightGBMBackend()

    assert backend._resolve_device_type({})[0] == "cpu"
    assert backend._resolve_device_type({"use_gpu": True})[0] == "gpu"
