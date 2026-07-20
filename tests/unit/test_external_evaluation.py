from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cs_copilot.tools.prediction import external_evaluation as external_evaluation_module
from cs_copilot.tools.prediction.backend import PredictionModelRecord, PredictionTaskSpec
from cs_copilot.tools.prediction.external_evaluation import evaluate_model_on_external_dataset


class _BinaryBackend:
    backend_name = "mock"

    def predict_from_csv(self, input_csv, model_record, preds_path, *, return_uncertainty=False):
        df = pd.read_csv(input_csv)
        predictions = [0, 1, 1, 0][: len(df)]
        output = pd.DataFrame(
            {
                "smiles": df["smiles"],
                "prediction": predictions,
                f"{model_record.task.target_columns[0]}_prediction": predictions,
                "positive_probability": [0.05, 0.95, 0.8, 0.2][: len(df)],
            }
        )
        output.to_csv(preds_path, index=False)
        return {"preds_path": preds_path}


class _MultiTargetBinaryBackend:
    backend_name = "mock"

    def predict_from_csv(self, input_csv, model_record, preds_path, *, return_uncertainty=False):
        df = pd.read_csv(input_csv)
        output = pd.DataFrame({"smiles": df["smiles"]})
        for target in model_record.task.target_columns:
            output[f"{target}_prediction"] = [0, 1, 1, 0][: len(df)]
            output[f"{target}_positive_probability"] = [0.1, 0.9, 0.9, 0.1][: len(df)]
        output.to_csv(preds_path, index=False)
        return {"preds_path": preds_path}


class _TargetNamedRegressionBackend:
    backend_name = "mock"

    def predict_from_csv(self, input_csv, model_record, preds_path, *, return_uncertainty=False):
        df = pd.read_csv(input_csv)
        target = model_record.task.target_columns[0]
        assert target not in df.columns
        pd.DataFrame({"smiles": df["smiles"], target: [1.1, 2.2]}).to_csv(preds_path, index=False)
        return {"preds_path": preds_path}


def _record(tmp_path: Path, *, target_columns=None) -> PredictionModelRecord:
    model_root = tmp_path / "model"
    model_root.mkdir()
    model_path = model_root / "best.pkl"
    model_path.write_text("mock")
    metadata_path = model_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_id": "mock_model",
                "backend_name": "mock",
                "status": "workflow_demo",
                "known_metrics": {"legacy": {"accuracy": 0.5}},
                "training_data_summary": {"metrics_status": "not_evaluated"},
                "task": {
                    "task_type": "classification",
                    "smiles_columns": ["smiles"],
                    "target_columns": target_columns or ["Y"],
                },
                "artifacts": {"model_path": "best.pkl"},
            }
        )
        + "\n"
    )
    return PredictionModelRecord(
        model_id="mock_model",
        backend_name="mock",
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        task=PredictionTaskSpec(
            task_type="classification",
            smiles_columns=["smiles"],
            target_columns=target_columns or ["Y"],
        ),
        status="workflow_demo",
        known_metrics={"legacy": {"accuracy": 0.5}},
        training_data_summary={"metrics_status": "not_evaluated"},
    )


def test_external_evaluation_appends_metadata_and_artifacts(tmp_path):
    record = _record(tmp_path)
    test_csv = tmp_path / "external.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCl"],
            "Y": [0, 1, 1, 0],
        }
    ).to_csv(test_csv, index=False)

    first = evaluate_model_on_external_dataset(
        record=record,
        backend=_BinaryBackend(),
        test_csv=str(test_csv),
        evaluation_label="first_panel",
    )
    second = evaluate_model_on_external_dataset(
        record=record,
        backend=_BinaryBackend(),
        test_csv=str(test_csv),
        evaluation_label="second_panel",
    )

    metadata = json.loads(Path(record.metadata_path).read_text())
    assert metadata["known_metrics"] == {"legacy": {"accuracy": 0.5}}
    assert [item["evaluation_id"] for item in metadata["external_evaluations"]] == [
        first["evaluation_id"],
        second["evaluation_id"],
    ]
    assert Path(first["predictions_path"]).exists()
    assert Path(first["metrics_path"]).exists()
    assert Path(first["evaluation_report_path"]).exists()
    assert first["metrics"]["accuracy"] == 1.0
    assert Path(first["artifacts"]["plots"]["Y"]["confusion_matrix"]).exists()
    assert Path(first["artifacts"]["plots"]["Y"]["roc_curve"]).exists()


def test_external_evaluation_preserves_legacy_id_and_suffixes_only_on_collision(
    tmp_path,
    monkeypatch,
):
    record = _record(tmp_path)
    test_csv = tmp_path / "external.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN"], "Y": [0, 1]}).to_csv(
        test_csv,
        index=False,
    )
    fixed_now = datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(external_evaluation_module, "project_now", lambda: fixed_now)

    results = [
        evaluate_model_on_external_dataset(
            record=record,
            backend=_BinaryBackend(),
            test_csv=str(test_csv),
            evaluation_label="same panel",
        )
        for _ in range(3)
    ]

    assert [result["evaluation_id"] for result in results] == [
        "same_panel_20260102_030405",
        "same_panel_20260102_030405_123456",
        "same_panel_20260102_030405_123456_2",
    ]


def test_external_evaluation_requires_all_targets_before_writing(tmp_path):
    record = _record(tmp_path, target_columns=["a", "b"])
    test_csv = tmp_path / "missing_target.csv"
    pd.DataFrame({"smiles": ["CCO"], "a": [1]}).to_csv(test_csv, index=False)

    try:
        evaluate_model_on_external_dataset(
            record=record,
            backend=_MultiTargetBinaryBackend(),
            test_csv=str(test_csv),
        )
    except ValueError as exc:
        assert "missing required target columns" in str(exc)
    else:
        raise AssertionError("missing target columns should fail")

    assert not (Path(record.metadata_path).parent / "evaluations").exists()


def test_external_evaluation_multitarget_writes_metrics_by_target(tmp_path):
    record = _record(tmp_path, target_columns=["a", "b"])
    test_csv = tmp_path / "multi.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCl"],
            "a": [0, 1, 1, 0],
            "b": [1, 0, 1, 0],
        }
    ).to_csv(test_csv, index=False)

    result = evaluate_model_on_external_dataset(
        record=record,
        backend=_MultiTargetBinaryBackend(),
        test_csv=str(test_csv),
        evaluation_label="multi_panel",
    )

    assert result["metrics"]["target_count"] == 2
    assert Path(result["artifacts"]["metrics_by_target"]).exists()
    assert Path(result["artifacts"]["plots"]["a"]["confusion_matrix"]).exists()
    assert Path(result["artifacts"]["plots"]["b"]["confusion_matrix"]).exists()


def test_external_evaluation_handles_target_named_prediction_columns(tmp_path):
    record = _record(tmp_path, target_columns=["pEC50"])
    record.task.task_type = "regression"
    test_csv = tmp_path / "external_regression.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN"],
            "pEC50": [1.0, 2.0],
        }
    ).to_csv(test_csv, index=False)

    result = evaluate_model_on_external_dataset(
        record=record,
        backend=_TargetNamedRegressionBackend(),
        test_csv=str(test_csv),
    )

    predictions = pd.read_csv(result["predictions_path"])
    assert "pEC50" in predictions.columns
    assert "pEC50_prediction" in predictions.columns
    assert predictions["pEC50"].tolist() == [1.0, 2.0]
    assert predictions["pEC50_prediction"].tolist() == [1.1, 2.2]
