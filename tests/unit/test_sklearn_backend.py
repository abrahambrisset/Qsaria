from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from cs_copilot.tools.prediction.backend import (
    InvalidPredictionInputError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from cs_copilot.tools.prediction.sklearn_backend import (
    SKLEARN_ESTIMATORS,
    SklearnBackend,
    SklearnRunRequest,
)

N_ROWS = 24
TRAIN_IDX = list(range(16))
TEST_IDX = list(range(16, N_ROWS))
FIXED_SPLIT = [{"train": TRAIN_IDX, "test": TEST_IDX}]


def _regression_frame() -> pd.DataFrame:
    """A deterministic, exactly linear frame: y = 2*x1 - 3*x2 + 5."""
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=N_ROWS)
    x2 = rng.normal(scale=100.0, size=N_ROWS)  # deliberately wildly unscaled
    return pd.DataFrame(
        {
            "smiles": [f"C{'C' * (index % 5)}O" for index in range(N_ROWS)],
            "x1": x1,
            "x2": x2,
            "Y": 2.0 * x1 - 3.0 * x2 + 5.0,
        }
    )


def _classification_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "smiles": [f"C{index}" for index in range(N_ROWS)],
            "x1": [float(index % 2) for index in range(N_ROWS)],
            "x2": [float(index) for index in range(N_ROWS)],
            "active": ["inactive", "active"] * (N_ROWS // 2),
        }
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> str:
    frame.to_csv(path, index=False)
    return str(path)


def _request(**overrides: Any) -> SklearnRunRequest:
    payload: Dict[str, Any] = {
        "train_csv": overrides.pop("train_csv"),
        "output_dir": overrides.pop("output_dir"),
        "task_type": overrides.pop("task_type", "regression"),
        "smiles_columns": ["smiles"],
        "target_columns": overrides.pop("target_columns", ["Y"]),
        "feature_columns": overrides.pop("feature_columns", ["x1", "x2"]),
        "split_payload": overrides.pop("split_payload", FIXED_SPLIT),
    }
    payload.update(overrides)
    return SklearnRunRequest(**payload)


def test_dummy_regressor_predicts_the_training_mean(tmp_path):
    """Closed form: DummyRegressor(strategy="mean") must return mean(y_train)."""
    frame = _regression_frame()
    train_csv = _write_csv(frame, tmp_path / "train.csv")
    backend = SklearnBackend()

    result = backend.train_model(
        _request(
            train_csv=train_csv,
            output_dir=str(tmp_path / "dummy_run"),
            estimator="dummy",
        )
    )

    expected_mean = float(frame["Y"].iloc[TRAIN_IDX].mean())
    predictions = pd.read_csv(result["test_predictions_path"])["y_pred"]
    assert len(predictions) == len(TEST_IDX)
    assert np.allclose(predictions.to_numpy(), expected_mean, atol=1e-9)

    # A mean-only model explains no variance on a held-out split.
    assert result["metrics"]["test"]["r2"] < 0.05
    expected_rmse = float(math.sqrt(((frame["Y"].iloc[TEST_IDX] - expected_mean) ** 2).mean()))
    assert result["metrics"]["test"]["rmse"] == pytest.approx(expected_rmse, rel=1e-9)


def test_random_forest_writes_artifacts_and_predicts_heldout_frame(tmp_path):
    frame = _regression_frame()
    train_csv = _write_csv(frame, tmp_path / "train.csv")
    output_dir = tmp_path / "rf_run"
    backend = SklearnBackend()

    result = backend.train_model(
        _request(
            train_csv=train_csv,
            output_dir=str(output_dir),
            estimator="random_forest",
            n_estimators=25,
        )
    )

    model_path = Path(result["model_path"])
    summary_path = Path(result["summary_path"])
    assert model_path.exists() and model_path.name == "best.pkl"
    assert summary_path.name == "cs_copilot_training_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["estimator"] == "random_forest"
    assert summary["feature_scaling"] == "none"
    assert summary["metrics"]["test"]["n"] == len(TEST_IDX)
    assert Path(result["test_predictions_path"]).name == "test_predictions.csv"
    assert Path(result["splits_path"]).exists()

    payload = pickle.loads(model_path.read_bytes())
    assert payload["feature_columns"] == ["x1", "x2"]
    assert payload["backend_name"] == "sklearn"

    holdout = frame.iloc[TEST_IDX].drop(columns=["Y"]).reset_index(drop=True)
    holdout_csv = _write_csv(holdout, tmp_path / "holdout.csv")
    preds_path = tmp_path / "preds.csv"
    record = PredictionModelRecord(
        model_id="rf",
        backend_name="sklearn",
        model_path=str(model_path),
        task=PredictionTaskSpec(
            task_type="regression", smiles_columns=["smiles"], target_columns=["Y"]
        ),
    )
    inference = backend.predict_from_csv(holdout_csv, record, str(preds_path))

    assert inference["rows"] == len(TEST_IDX)
    predictions = pd.read_csv(preds_path)
    assert list(predictions.columns) == ["prediction", "Y"]
    assert predictions["prediction"].notna().all()
    # A forest can only interpolate its training targets.
    train_targets = frame["Y"].iloc[TRAIN_IDX]
    assert predictions["prediction"].min() >= float(train_targets.min()) - 1e-9
    assert predictions["prediction"].max() <= float(train_targets.max()) + 1e-9


def test_scaled_estimators_are_pipelines_and_tree_models_are_not(tmp_path):
    frame = _regression_frame()
    train_csv = _write_csv(frame, tmp_path / "train.csv")
    backend = SklearnBackend()
    observed: Dict[str, bool] = {}

    for estimator in SKLEARN_ESTIMATORS:
        result = backend.train_model(
            _request(
                train_csv=train_csv,
                output_dir=str(tmp_path / f"run_{estimator}"),
                estimator=estimator,
                n_estimators=10,
            )
        )
        payload = pickle.loads(Path(result["model_path"]).read_bytes())
        observed[estimator] = bool(payload["scaled"])
        assert hasattr(payload["model"], "predict")
        assert result["metrics"]["test"]["n"] == len(TEST_IDX)

    assert observed == {
        "dummy": False,
        "random_forest": False,
        "svm": True,
        "knn": True,
        "pls": True,
        "ridge": True,
    }


def test_pls_recovers_an_exactly_linear_target(tmp_path):
    """Closed form: two PLS components on a rank-2 linear target fit it exactly."""
    frame = _regression_frame()
    train_csv = _write_csv(frame, tmp_path / "train.csv")
    backend = SklearnBackend()

    result = backend.train_model(
        _request(
            train_csv=train_csv,
            output_dir=str(tmp_path / "pls_run"),
            estimator="pls",
            n_components=2,
        )
    )

    assert result["metrics"]["test"]["r2"] == pytest.approx(1.0, abs=1e-8)
    predictions = pd.read_csv(result["test_predictions_path"])
    assert np.allclose(
        predictions["y_pred"].to_numpy(),
        frame["Y"].iloc[TEST_IDX].to_numpy(),
        atol=1e-6,
    )


def test_pls_classification_is_rejected(tmp_path):
    frame = _classification_frame()
    train_csv = _write_csv(frame, tmp_path / "train.csv")
    backend = SklearnBackend()

    with pytest.raises(InvalidPredictionInputError, match="regression only"):
        backend.train_model(
            _request(
                train_csv=train_csv,
                output_dir=str(tmp_path / "pls_clf"),
                task_type="classification",
                target_columns=["active"],
                estimator="pls",
            )
        )


def test_feature_column_equal_to_target_is_refused(tmp_path):
    frame = _regression_frame()
    frame["copied_target"] = frame["Y"]
    train_csv = _write_csv(frame, tmp_path / "train.csv")
    backend = SklearnBackend()

    # Named target column requested as a feature.
    with pytest.raises(InvalidPredictionInputError, match="never be used as features"):
        backend.train_model(
            _request(
                train_csv=train_csv,
                output_dir=str(tmp_path / "leak_named"),
                feature_columns=["x1", "Y"],
            )
        )

    # A copy of the target under a different name.
    with pytest.raises(InvalidPredictionInputError, match="identical to target"):
        backend.train_model(
            _request(
                train_csv=train_csv,
                output_dir=str(tmp_path / "leak_copy"),
                feature_columns=["x1", "copied_target"],
            )
        )

    # Auto-selection must drop the target and the SMILES column.
    auto = backend.train_model(
        _request(
            train_csv=_write_csv(_regression_frame(), tmp_path / "clean.csv"),
            output_dir=str(tmp_path / "auto"),
            feature_columns=None,
            estimator="dummy",
        )
    )
    assert auto["feature_columns"] == ["x1", "x2"]


def test_balanced_class_weight_changes_an_imbalanced_classifier(tmp_path):
    rows = 40
    labels = ["active" if index % 10 == 0 else "inactive" for index in range(rows)]
    frame = pd.DataFrame(
        {
            "smiles": [f"C{index}" for index in range(rows)],
            "x1": [float(index % 10) for index in range(rows)],
            "x2": [float(index) for index in range(rows)],
            "active": labels,
        }
    )
    train_csv = _write_csv(frame, tmp_path / "imbalanced.csv")
    split: List[Dict[str, Any]] = [
        {"train": list(range(30)), "test": list(range(30, rows))},
    ]
    backend = SklearnBackend()

    def _run(class_weight: Any, name: str) -> Dict[str, Any]:
        return backend.train_model(
            _request(
                train_csv=train_csv,
                output_dir=str(tmp_path / name),
                task_type="classification",
                target_columns=["active"],
                estimator="random_forest",
                n_estimators=25,
                split_payload=split,
                **({"class_weight": class_weight} if class_weight else {}),
            )
        )

    plain = _run(None, "plain")
    balanced = _run("balanced", "balanced")

    assert plain["class_count"] == 2
    assert plain["effective_train_args"].get("class_weight") is None
    assert balanced["effective_train_args"]["class_weight"] == "balanced"
    plain_model = pickle.loads(Path(plain["model_path"]).read_bytes())["model"]
    balanced_model = pickle.loads(Path(balanced["model_path"]).read_bytes())["model"]
    assert plain_model.class_weight is None
    assert balanced_model.class_weight == "balanced"
    assert "positive_probability" in pd.read_csv(balanced["test_predictions_path"]).columns


def test_class_weight_is_rejected_where_it_has_no_meaning(tmp_path):
    frame = _classification_frame()
    train_csv = _write_csv(frame, tmp_path / "train.csv")
    backend = SklearnBackend()

    with pytest.raises(InvalidPredictionInputError, match="does not support class_weight"):
        backend.train_model(
            _request(
                train_csv=train_csv,
                output_dir=str(tmp_path / "knn_balanced"),
                task_type="classification",
                target_columns=["active"],
                estimator="knn",
                class_weight="balanced",
            )
        )

    with pytest.raises(InvalidPredictionInputError, match="classification-only"):
        backend.train_model(
            _request(
                train_csv=_write_csv(_regression_frame(), tmp_path / "reg.csv"),
                output_dir=str(tmp_path / "reg_balanced"),
                estimator="random_forest",
                class_weight="balanced",
            )
        )


def test_classification_roundtrip_reports_labels_and_probabilities(tmp_path):
    frame = _classification_frame()
    train_csv = _write_csv(frame, tmp_path / "train.csv")
    backend = SklearnBackend()

    result = backend.train_model(
        _request(
            train_csv=train_csv,
            output_dir=str(tmp_path / "clf"),
            task_type="classification",
            target_columns=["active"],
            estimator="ridge",
        )
    )

    assert result["task_kind"] == "binary_classification"
    assert result["class_labels"] == ["inactive", "active"]
    assert result["positive_class_label"] == "active"
    assert result["metrics"]["test"]["accuracy"] == pytest.approx(1.0)
    test_predictions = pd.read_csv(result["test_predictions_path"])
    assert set(test_predictions["y_pred"]) <= {"active", "inactive"}

    preds_path = tmp_path / "clf_preds.csv"
    record = PredictionModelRecord(
        model_id="clf",
        backend_name="sklearn",
        model_path=result["model_path"],
        task=PredictionTaskSpec(
            task_type="classification", smiles_columns=["smiles"], target_columns=["active"]
        ),
    )
    backend.predict_from_csv(train_csv, record, str(preds_path))
    predictions = pd.read_csv(preds_path)
    assert "positive_probability" in predictions.columns
    assert predictions["positive_probability"].between(0.0, 1.0).all()


def test_non_finite_features_and_multi_target_are_rejected(tmp_path):
    frame = _regression_frame()
    frame.loc[3, "x1"] = np.nan
    train_csv = _write_csv(frame, tmp_path / "nan.csv")
    backend = SklearnBackend()

    with pytest.raises(InvalidPredictionInputError, match="finite"):
        backend.train_model(_request(train_csv=train_csv, output_dir=str(tmp_path / "nan_run")))

    clean_csv = _write_csv(_regression_frame(), tmp_path / "clean.csv")
    with pytest.raises(InvalidPredictionInputError, match="exactly one target column"):
        backend.train_model(
            _request(
                train_csv=clean_csv,
                output_dir=str(tmp_path / "multi"),
                target_columns=["Y", "x1"],
            )
        )


def test_describe_environment_declares_single_target_limit():
    backend = SklearnBackend()
    environment = backend.describe_environment()

    assert backend.is_available() is True
    assert environment["backend_name"] == "sklearn"
    assert environment["supports_multi_target"] is False
    assert list(environment["supported_estimators"]) == list(SKLEARN_ESTIMATORS)
    assert any("Single target column" in note for note in environment["limitations"])


def test_validate_model_path_rejects_unknown_artifacts(tmp_path):
    backend = SklearnBackend()
    wrong_suffix = tmp_path / "model.joblib"
    wrong_suffix.write_bytes(b"not a pickle")

    with pytest.raises(InvalidPredictionInputError, match="must end with"):
        backend.validate_model_path(str(wrong_suffix))
    with pytest.raises(InvalidPredictionInputError, match="does not exist"):
        backend.validate_model_path(str(tmp_path / "missing.pkl"))
