from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from cs_copilot.tools.prediction.backend import InvalidPredictionInputError, PredictionModelRecord, PredictionTaskSpec
import cs_copilot.tools.prediction.lightgbm_backend as lightgbm_backend_module
from cs_copilot.tools.prediction.lightgbm_backend import LightGBMBackend


class PickleableFakeClassifier:
    def __init__(self, **params):
        self.params = params
        self.class_count = int(params.get("num_class") or 2)

    def fit(self, X_train, y_train, **kwargs):
        self.fit_kwargs = kwargs
        return self

    def predict(self, X):
        values = pd.to_numeric(X.iloc[:, 0], errors="coerce").fillna(0).astype(float)
        if self.class_count <= 2:
            return (values >= 0.5).astype(int).to_numpy()
        return values.round().astype(int).mod(self.class_count).to_numpy()

    def predict_proba(self, X):
        predictions = self.predict(X)
        rows = []
        for pred in predictions:
            row = [0.05] * self.class_count
            row[int(pred)] = 0.9
            rows.append(row)
        return rows


def _fake_classifier_module():
    return SimpleNamespace(
        LGBMClassifier=PickleableFakeClassifier,
        log_evaluation=lambda period: {"callback": "log_evaluation", "period": period},
        early_stopping=lambda stopping_rounds, verbose: {
            "callback": "early_stopping",
            "stopping_rounds": stopping_rounds,
            "verbose": verbose,
        },
    )


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


def test_lightgbm_binary_classification_roundtrip_with_text_labels(tmp_path, monkeypatch):
    backend = LightGBMBackend()
    monkeypatch.setattr(backend, "_ensure_available", lambda: None)
    monkeypatch.setattr(backend, "_import_lightgbm", _fake_classifier_module)
    train_csv = tmp_path / "binary.csv"
    pd.DataFrame(
        {
            "smiles": [f"C{i}" for i in range(12)],
            "x": [0.0, 1.0] * 6,
            "active": ["inactive", "active"] * 6,
        }
    ).to_csv(train_csv, index=False)

    result = backend.train_model(
        str(train_csv),
        str(tmp_path / "model"),
        PredictionTaskSpec(task_type="classification", smiles_columns=["smiles"], target_columns=["active"]),
        extra_args={
            "feature_columns": ["x"],
            "split_payload": [{"train": list(range(8)), "test": list(range(8, 12))}],
            "n_estimators": 5,
            "early_stopping_rounds": 0,
        },
    )

    assert result["task_kind"] == "binary_classification"
    assert result["class_count"] == 2
    assert result["metrics"]["test"]["balanced_accuracy"] == 1.0
    preds = pd.read_csv(result["test_predictions_path"])
    assert {"prediction", "active_true", "positive_probability"}.issubset(preds.columns)

    prediction_csv = tmp_path / "predictions.csv"
    record = PredictionModelRecord(
        model_id="binary",
        backend_name="lightgbm",
        model_path=result["model_path"],
        task=PredictionTaskSpec(task_type="classification", smiles_columns=["smiles"], target_columns=["active"]),
    )
    backend.predict_from_csv(str(train_csv), record, str(prediction_csv))
    assert "positive_probability" in pd.read_csv(prediction_csv).columns


def test_lightgbm_multiclass_classification_roundtrip(tmp_path, monkeypatch):
    backend = LightGBMBackend()
    monkeypatch.setattr(backend, "_ensure_available", lambda: None)
    monkeypatch.setattr(backend, "_import_lightgbm", _fake_classifier_module)
    train_csv = tmp_path / "multiclass.csv"
    labels = ["high", "low", "medium"] * 4
    pd.DataFrame(
        {
            "smiles": [f"C{i}" for i in range(12)],
            "x": [0, 1, 2] * 4,
            "class_label": labels,
        }
    ).to_csv(train_csv, index=False)

    result = backend.train_model(
        str(train_csv),
        str(tmp_path / "model"),
        PredictionTaskSpec(task_type="multiclass_classification", smiles_columns=["smiles"], target_columns=["class_label"]),
        extra_args={
            "feature_columns": ["x"],
            "split_payload": [{"train": list(range(9)), "test": list(range(9, 12))}],
            "n_estimators": 5,
            "early_stopping_rounds": 0,
        },
    )

    assert result["task_kind"] == "multiclass_classification"
    assert result["class_count"] == 3
    assert "roc_auc" not in result["metrics"]["test"]
    assert result["metrics"]["test"]["balanced_accuracy"] == 1.0
