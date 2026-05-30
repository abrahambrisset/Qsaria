from __future__ import annotations

import pandas as pd
import pytest

from cs_copilot.tools.prediction.backend import (
    InvalidPredictionInputError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
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
    task = PredictionTaskSpec(
        task_type="regression", smiles_columns=["smiles"], target_columns=["Y"]
    )

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
    task = PredictionTaskSpec(
        task_type="regression", smiles_columns=["smiles"], target_columns=["Y"]
    )

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


def test_classification_helpers_encode_metrics_and_probability_output():
    backend = LightGBMBackend()
    target = pd.Series(["inactive", "active", "active", "inactive"])

    encoded, labels, mapping = backend._encode_classification_target(target, {})

    assert labels == ["inactive", "active"]
    assert mapping == {"inactive": 0, "active": 1}
    assert encoded.tolist() == [0, 1, 1, 0]

    predictions = pd.Series([0, 1, 0, 0])
    probabilities = [[0.8, 0.2], [0.1, 0.9], [0.45, 0.55], [0.7, 0.3]]
    metrics = backend._compute_classification_metrics(
        encoded,
        predictions,
        probabilities,
        labels,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["positive_class"] == "active"

    output = backend._classification_output_frame(
        predicted_codes=predictions,
        probabilities=probabilities,
        class_labels=labels,
        target_column="activity",
        true_codes=encoded,
        true_labels=target,
    )

    assert output["prediction"].tolist() == ["inactive", "active", "inactive", "inactive"]
    assert "probability_inactive" in output.columns
    assert "probability_active" in output.columns
    assert "positive_class_probability" in output.columns


def test_lightgbm_classification_train_predict_roundtrip(tmp_path):
    pytest.importorskip("lightgbm")
    backend = LightGBMBackend()
    train_csv = tmp_path / "classification_train.csv"
    rows = []
    for index in range(60):
        active = index >= 30
        rows.append(
            {
                "smiles": "CCO" if index % 2 == 0 else "CCC",
                "activity": "active" if active else "inactive",
                "feature_x": float(index) / 60.0,
                "feature_y": float(index % 5),
            }
        )
    pd.DataFrame(rows).to_csv(train_csv, index=False)

    task = PredictionTaskSpec(
        task_type="classification",
        smiles_columns=["smiles"],
        target_columns=["activity"],
    )
    result = backend.train_model(
        train_csv=str(train_csv),
        output_dir=str(tmp_path / "model_out"),
        task=task,
        extra_args={
            "feature_columns": ["feature_x", "feature_y"],
            "split_type": "random",
            "split_sizes": [0.7, 0.15, 0.15],
            "random_state": 7,
            "n_estimators": 30,
            "early_stopping_rounds": 5,
            "min_child_samples": 1,
            "num_leaves": 7,
            "device_type": "cpu",
        },
    )

    assert result["class_labels"] == ["inactive", "active"]
    assert result["metrics"]["test"]["balanced_accuracy"] is not None
    predictions = pd.read_csv(result["test_predictions_path"])
    assert "predicted_class" in predictions.columns
    assert "probability_active" in predictions.columns
    assert "positive_class_probability" in predictions.columns

    input_csv = tmp_path / "classification_input.csv"
    pd.DataFrame(
        [
            {"smiles": "CCO", "feature_x": 0.05, "feature_y": 0.0},
            {"smiles": "CCC", "feature_x": 0.95, "feature_y": 4.0},
        ]
    ).to_csv(input_csv, index=False)
    preds_path = tmp_path / "classification_predictions.csv"
    inference_result = backend.predict_from_csv(
        input_csv=str(input_csv),
        model_record=PredictionModelRecord(
            model_id="classification_model",
            backend_name="lightgbm",
            model_path=result["model_path"],
            task=task,
        ),
        preds_path=str(preds_path),
    )

    predicted = pd.read_csv(preds_path)
    assert inference_result["task_type"] == "classification"
    assert predicted["prediction"].isin(["active", "inactive"]).all()
    assert "probability_inactive" in predicted.columns
    assert "probability_active" in predicted.columns
