from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cs_copilot.tools.prediction.backend import (
    InvalidPredictionInputError,
    PredictionBackend,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from cs_copilot.tools.prediction.backend_capabilities import BackendCapabilities
from cs_copilot.tools.prediction.catalog import PredictionModelCatalog
from cs_copilot.tools.prediction.ensemble_backend import EnsembleBackend
from cs_copilot.tools.prediction.ensemble_toolkit import EnsembleToolkit


class FakeBackend(PredictionBackend):
    backend_name = "fake"

    def __init__(self, offset: float):
        self.offset = offset

    def is_available(self) -> bool:
        return True

    def describe_environment(self):
        return {"backend_name": self.backend_name, "available": True}

    def validate_model_path(self, model_path: str) -> Path:
        path = Path(model_path)
        if not path.exists():
            raise ValueError("missing")
        return path

    def predict_from_csv(
        self, input_csv, model_record, preds_path, *, return_uncertainty=False, extra_args=None
    ):
        df = pd.read_csv(input_csv)
        pd.DataFrame({"prediction": df["x"].astype(float) + self.offset}).to_csv(
            preds_path, index=False
        )
        return {"predictions_path": preds_path}

    def train_model(self, train_csv, output_dir, task, *, extra_args=None):
        raise NotImplementedError


class TabularFakeBackend(FakeBackend):
    backend_name = "fake_tabular"

    def predict_from_csv(
        self, input_csv, model_record, preds_path, *, return_uncertainty=False, extra_args=None
    ):
        df = pd.read_csv(input_csv)
        pd.DataFrame({"prediction": df["fp_0000"].astype(float) + self.offset}).to_csv(
            preds_path, index=False
        )
        return {"predictions_path": preds_path}


class ClassificationFakeBackend(PredictionBackend):
    backend_name = "fake_classification"

    def __init__(self, probabilities):
        self.probabilities = list(probabilities)

    def is_available(self) -> bool:
        return True

    def describe_environment(self):
        return {"backend_name": self.backend_name, "available": True}

    def validate_model_path(self, model_path: str) -> Path:
        path = Path(model_path)
        if not path.exists():
            raise ValueError("missing")
        return path

    def predict_from_csv(
        self, input_csv, model_record, preds_path, *, return_uncertainty=False, extra_args=None
    ):
        df = pd.read_csv(input_csv)
        probabilities = self.probabilities[: len(df)]
        labels = ["active" if value >= 0.5 else "inactive" for value in probabilities]
        pd.DataFrame(
            {
                "prediction": labels,
                "predicted_class": labels,
                "positive_class_probability": probabilities,
                "probability_inactive": [1.0 - value for value in probabilities],
                "probability_active": probabilities,
            }
        ).to_csv(preds_path, index=False)
        return {"predictions_path": preds_path}

    def train_model(self, train_csv, output_dir, task, *, extra_args=None):
        raise NotImplementedError


FAKE_CAPABILITIES = {
    "fake": BackendCapabilities(
        backend_name="fake",
        can_train=False,
        can_predict=True,
        prediction_input_kinds=("smiles_csv",),
        requires_feature_preparation=False,
        supported_task_types=("regression",),
        supported_representations=("fake",),
        supports_applicability_domain=False,
        supports_uncertainty="none",
    ),
    "class_a": BackendCapabilities(
        backend_name="class_a",
        can_train=False,
        can_predict=True,
        prediction_input_kinds=("smiles_csv",),
        requires_feature_preparation=False,
        supported_task_types=("classification",),
        supported_representations=("fake_classification",),
        supports_applicability_domain=False,
        supports_uncertainty="none",
    ),
    "class_b": BackendCapabilities(
        backend_name="class_b",
        can_train=False,
        can_predict=True,
        prediction_input_kinds=("smiles_csv",),
        requires_feature_preparation=False,
        supported_task_types=("classification",),
        supported_representations=("fake_classification",),
        supports_applicability_domain=False,
        supports_uncertainty="none",
    ),
    "fake2": BackendCapabilities(
        backend_name="fake2",
        can_train=False,
        can_predict=True,
        prediction_input_kinds=("smiles_csv",),
        requires_feature_preparation=False,
        supported_task_types=("regression",),
        supported_representations=("fake",),
        supports_applicability_domain=False,
        supports_uncertainty="none",
    ),
    "fake_tabular": BackendCapabilities(
        backend_name="fake_tabular",
        can_train=False,
        can_predict=True,
        prediction_input_kinds=("tabular_features_csv",),
        requires_feature_preparation=True,
        supported_task_types=("regression",),
        supported_representations=("morgan_only",),
        supports_applicability_domain=False,
        supports_uncertainty="none",
    ),
}


def _record(model_id: str, backend: str, model_path: Path, target: str = "pEC50", **kwargs):
    return PredictionModelRecord(
        model_id=model_id,
        backend_name=backend,
        model_path=str(model_path),
        status=kwargs.get("status", "workflow_demo"),
        known_metrics=kwargs.get("known_metrics", {}),
        training_data_summary=kwargs.get("training_data_summary", {}),
        inference_profile=kwargs.get("inference_profile", {}),
        selection_hints=kwargs.get("selection_hints", {}),
        task=PredictionTaskSpec(
            task_type=kwargs.get("task_type", "regression"),
            smiles_columns=["smiles"],
            target_columns=[target],
        ),
    )


def test_ensemble_backend_predicts_component_columns(tmp_path):
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"smiles": ["CC", "CCC"], "x": [1.0, 2.0]}).to_csv(input_csv, index=False)
    model_a = tmp_path / "a.fake"
    model_b = tmp_path / "b.fake"
    model_a.write_text("a")
    model_b.write_text("b")
    ensemble_path = tmp_path / "ensemble.json"
    ensemble_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ensemble_kind": "catalog_consensus_regression",
                "aggregation_strategy": "median",
                "components": [
                    {
                        "model_id": "a",
                        "component_slug": "a",
                        "backend_name": "fake",
                        "model_path": str(model_a),
                        "task": {
                            "task_type": "regression",
                            "smiles_columns": ["smiles"],
                            "target_columns": ["pEC50"],
                        },
                    },
                    {
                        "model_id": "b",
                        "component_slug": "b",
                        "backend_name": "fake2",
                        "model_path": str(model_b),
                        "task": {
                            "task_type": "regression",
                            "smiles_columns": ["smiles"],
                            "target_columns": ["pEC50"],
                        },
                    },
                ],
            }
        )
    )
    backend = EnsembleBackend(
        backends={"fake": FakeBackend(1.0), "fake2": FakeBackend(3.0)},
        backend_capabilities=FAKE_CAPABILITIES,
    )
    output = tmp_path / "preds.csv"
    record = _record("ens", "ensemble", ensemble_path)

    result = backend.predict_from_csv(str(input_csv), record, str(output))

    preds = pd.read_csv(output)
    assert result["component_count"] == 2
    assert list(preds.columns) == [
        "prediction",
        "ensemble_prediction_median",
        "ensemble_prediction_mean",
        "ensemble_prediction_std",
        "ensemble_prediction_min",
        "ensemble_prediction_max",
        "ensemble_component_count",
        "prediction_a",
        "prediction_b",
    ]
    assert preds["prediction"].tolist() == [3.0, 4.0]
    assert preds["ensemble_component_count"].tolist() == [2, 2]
    summary = result["ensemble_inference_summary"]
    assert summary["report_kind"] == "ensemble_inference"
    assert summary["rows_predicted"] == 2
    assert summary["official_prediction_column"] == "ensemble_prediction_median"
    assert summary["uncertainty_strategy"] == "component_disagreement_std"
    assert [component["backend_name"] for component in summary["components"]] == ["fake", "fake2"]
    assert summary["prediction_summary"]["mean"] == 3.5
    assert summary["disagreement_summary"]["max"] == 1.0


def test_ensemble_backend_predicts_classification_probabilities(tmp_path):
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"smiles": ["CC", "CCC", "CCN"]}).to_csv(input_csv, index=False)
    model_a = tmp_path / "a.fake"
    model_b = tmp_path / "b.fake"
    model_a.write_text("a")
    model_b.write_text("b")
    ensemble_path = tmp_path / "ensemble.json"
    ensemble_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ensemble_kind": "catalog_consensus_classification",
                "task_type": "classification",
                "aggregation_strategy": "probability_mean",
                "class_labels": ["inactive", "active"],
                "components": [
                    {
                        "model_id": "a",
                        "component_slug": "a",
                        "backend_name": "class_a",
                        "model_path": str(model_a),
                        "task": {
                            "task_type": "classification",
                            "smiles_columns": ["smiles"],
                            "target_columns": ["activity"],
                        },
                    },
                    {
                        "model_id": "b",
                        "component_slug": "b",
                        "backend_name": "class_b",
                        "model_path": str(model_b),
                        "task": {
                            "task_type": "classification",
                            "smiles_columns": ["smiles"],
                            "target_columns": ["activity"],
                        },
                    },
                ],
            }
        )
    )
    backend = EnsembleBackend(
        backends={
            "class_a": ClassificationFakeBackend([0.2, 0.9, 0.4]),
            "class_b": ClassificationFakeBackend([0.4, 0.7, 0.8]),
        },
        backend_capabilities=FAKE_CAPABILITIES,
    )
    output = tmp_path / "preds.csv"
    record = _record(
        "ens", "ensemble", ensemble_path, target="activity", task_type="classification"
    )

    result = backend.predict_from_csv(str(input_csv), record, str(output))

    preds = pd.read_csv(output)
    assert result["aggregation_strategy"] == "probability_mean"
    assert preds["prediction"].tolist() == ["inactive", "active", "active"]
    assert preds["positive_class_probability"].tolist() == pytest.approx([0.3, 0.8, 0.6])
    assert preds["ensemble_probability_disagreement_std"].tolist() == pytest.approx([0.1, 0.1, 0.2])
    assert preds["prediction_a"].tolist() == ["inactive", "active", "inactive"]
    assert preds["prediction_b"].tolist() == ["inactive", "active", "active"]
    summary = result["ensemble_inference_summary"]
    assert summary["task_kind"] == "classification"
    assert summary["class_labels"] == ["inactive", "active"]
    assert summary["official_prediction_column"] == "ensemble_predicted_class"


def test_ensemble_backend_uses_capabilities_for_tabular_preparation(tmp_path):
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"smiles": ["CC", "CCC"], "x": [1.0, 2.0]}).to_csv(input_csv, index=False)
    model_path = tmp_path / "tabular.fake"
    model_path.write_text("model")
    ensemble_path = tmp_path / "ensemble.json"
    ensemble_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ensemble_kind": "catalog_consensus_regression",
                "aggregation_strategy": "median",
                "components": [
                    {
                        "model_id": "tabular",
                        "component_slug": "tabular",
                        "backend_name": "fake_tabular",
                        "model_path": str(model_path),
                        "inference_profile": {
                            "representation_name": "morgan_only",
                            "feature_columns": ["fp_0000"],
                        },
                        "task": {
                            "task_type": "regression",
                            "smiles_columns": ["smiles"],
                            "target_columns": ["pEC50"],
                        },
                    },
                ],
            }
        )
    )
    backend = EnsembleBackend(
        backends={"fake_tabular": TabularFakeBackend(10.0)},
        backend_capabilities=FAKE_CAPABILITIES,
    )

    def fake_morgan(input_csv, smiles_column="smiles", output_csv=None, **kwargs):
        source = pd.read_csv(input_csv)
        pd.DataFrame({"smiles": source["smiles"], "fp_0000": [5.0, 7.0]}).to_csv(
            output_csv, index=False
        )
        return {"output_csv": output_csv}

    backend.feature_toolkit = SimpleNamespace(smiles_to_morgan_fingerprints=fake_morgan)
    output = tmp_path / "preds.csv"
    record = _record("ens", "ensemble", ensemble_path)

    result = backend.predict_from_csv(str(input_csv), record, str(output))

    preds = pd.read_csv(output)
    assert preds["prediction"].tolist() == [15.0, 17.0]
    component_input = Path(result["component_input_paths"]["tabular"])
    assert component_input.exists()
    assert "fp_0000" in pd.read_csv(component_input).columns


def test_ensemble_backend_rejects_configured_backend_without_capabilities(tmp_path):
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"smiles": ["CC"], "x": [1.0]}).to_csv(input_csv, index=False)
    model_path = tmp_path / "model.fake"
    model_path.write_text("model")
    ensemble_path = tmp_path / "ensemble.json"
    ensemble_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ensemble_kind": "catalog_consensus_regression",
                "aggregation_strategy": "median",
                "components": [
                    {
                        "model_id": "uncap",
                        "component_slug": "uncap",
                        "backend_name": "uncap",
                        "model_path": str(model_path),
                        "task": {
                            "task_type": "regression",
                            "smiles_columns": ["smiles"],
                            "target_columns": ["pEC50"],
                        },
                    },
                ],
            }
        )
    )
    backend = EnsembleBackend(backends={"uncap": FakeBackend(1.0)})
    record = _record("ens", "ensemble", ensemble_path)

    with pytest.raises(Exception, match="no registered capabilities"):
        backend.predict_from_csv(str(input_csv), record, str(tmp_path / "preds.csv"))


def test_ensemble_backend_rejects_invalid_json(tmp_path):
    path = tmp_path / "ensemble.json"
    path.write_text(json.dumps({"schema_version": 1, "ensemble_kind": "wrong", "components": []}))
    with pytest.raises(InvalidPredictionInputError):
        EnsembleBackend(backends={}).validate_model_path(str(path))


def test_create_ensemble_from_catalog_persists_evidence(tmp_path, monkeypatch):
    import cs_copilot.tools.prediction.ensemble_toolkit as ensemble_module

    internal_root = tmp_path / "internal"
    monkeypatch.setattr(ensemble_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    model_a = tmp_path / "a.fake"
    model_b = tmp_path / "b.fake"
    model_a.write_text("a")
    model_b.write_text("b")
    catalog = PredictionModelCatalog(
        records=[
            _record(
                "std_scaffold",
                "fake",
                model_a,
                known_metrics={"scaffold": {"r2": 0.61}},
                training_data_summary={"validation_protocol": "standard_qsar"},
                inference_profile={"representation_name": "morgan_rdkit_basic"},
            ),
            _record(
                "robust_stable",
                "fake2",
                model_b,
                known_metrics={
                    "random": {"r2_mean": 0.58, "r2_std": 0.02},
                    "scaffold": {"r2": 0.59},
                },
                training_data_summary={"validation_protocol": "robust_qsar"},
                inference_profile={"representation_name": "molecular_graph"},
            ),
        ],
        source_path=tmp_path / "catalog.json",
    )
    catalog.save()
    toolkit = EnsembleToolkit(catalog=catalog)
    toolkit.backends = {"fake": FakeBackend(0), "fake2": FakeBackend(0)}
    agent = SimpleNamespace(session_state={})

    result = toolkit.create_ensemble_from_catalog("pEC50", agent=agent)

    assert result["status"] == "workflow_demo"
    assert result["known_metrics"] == {}
    payload = json.loads(Path(result["model_path"]).read_text())
    assert payload["ensemble_kind"] == "catalog_consensus_regression"
    assert payload["evaluations"] == []
    evidence = json.loads(Path(result["selection_evidence_path"]).read_text())
    assert evidence["compatible_count"] == 2
    assert {item["selection_decision"] for item in evidence["candidates"]} == {"included"}


def test_create_ensemble_rejects_incompatible_and_warns_ablation(tmp_path, monkeypatch):
    import cs_copilot.tools.prediction.ensemble_toolkit as ensemble_module

    monkeypatch.setattr(ensemble_module, "DEFAULT_INTERNAL_MODEL_ROOT", tmp_path / "internal")
    model_a = tmp_path / "a.fake"
    model_b = tmp_path / "b.fake"
    model_a.write_text("a")
    model_b.write_text("b")
    deprecated = _record("deprecated", "fake", model_a, status="deprecated")
    ablation = _record(
        "ablation",
        "fake",
        model_b,
        inference_profile={"representation_name": "morgan_only"},
    )
    catalog = PredictionModelCatalog(
        records=[deprecated, ablation], source_path=tmp_path / "catalog.json"
    )
    catalog.save()
    toolkit = EnsembleToolkit(catalog=catalog)
    toolkit.backends = {"fake": FakeBackend(0)}

    with pytest.raises(ValueError):
        toolkit.create_ensemble_from_catalog("pEC50", model_ids=["deprecated"])

    result = toolkit.create_ensemble_from_catalog("pEC50", model_ids=["ablation"])
    evidence = json.loads(Path(result["selection_evidence_path"]).read_text())
    selected = next(item for item in evidence["candidates"] if item["model_id"] == "ablation")
    assert selected["is_ablation"] is True
    assert selected["selection_decision"] == "included"
    assert selected["warnings"]


def test_evaluate_ensemble_appends_evaluations_and_writes_artifacts(tmp_path, monkeypatch):
    import cs_copilot.tools.prediction.ensemble_toolkit as ensemble_module

    internal_root = tmp_path / "internal"
    monkeypatch.setattr(ensemble_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    model_a = tmp_path / "a.fake"
    model_b = tmp_path / "b.fake"
    model_a.write_text("a")
    model_b.write_text("b")
    catalog = PredictionModelCatalog(
        records=[_record("a", "fake", model_a), _record("b", "fake2", model_b)],
        source_path=tmp_path / "catalog.json",
    )
    catalog.save()
    toolkit = EnsembleToolkit(catalog=catalog)
    toolkit.backends = {"fake": FakeBackend(0.0), "fake2": FakeBackend(2.0)}
    toolkit.ensemble_backend = EnsembleBackend(
        backends=toolkit.backends,
        backend_capabilities=FAKE_CAPABILITIES,
    )
    created = toolkit.create_ensemble_from_catalog("pEC50")
    test_csv = tmp_path / "test.csv"
    pd.DataFrame({"smiles": ["CC", "CCC"], "x": [1.0, 2.0], "pEC50": [2.0, 3.0]}).to_csv(
        test_csv, index=False
    )

    first = toolkit.evaluate_ensemble_on_dataset(created["model_id"], str(test_csv), "pEC50")
    second = toolkit.evaluate_ensemble_on_dataset(
        created["model_id"],
        str(test_csv),
        "pEC50",
        evaluation_kind="training_like_or_potentially_leaky",
    )

    assert Path(first["evaluation_summary_path"]).exists()
    assert Path(first["metrics_by_component_path"]).exists()
    assert first["ensemble_metrics"]["rmse"] == 0.0
    payload = json.loads(Path(created["model_path"]).read_text())
    assert len(payload["evaluations"]) == 2
    assert (
        payload["evaluations"][0]["evaluation_id"] != payload["evaluations"][1]["evaluation_id"]
        or second
    )


def test_create_and_evaluate_classification_ensemble_from_catalog(tmp_path, monkeypatch):
    import cs_copilot.tools.prediction.ensemble_toolkit as ensemble_module

    internal_root = tmp_path / "internal"
    monkeypatch.setattr(ensemble_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    model_a = tmp_path / "a.fake"
    model_b = tmp_path / "b.fake"
    model_a.write_text("a")
    model_b.write_text("b")
    catalog = PredictionModelCatalog(
        records=[
            _record(
                "class_a_model",
                "class_a",
                model_a,
                target="activity",
                task_type="classification",
                known_metrics={"test": {"balanced_accuracy": 0.80}},
                inference_profile={
                    "representation_name": "molecular_graph",
                    "class_labels": ["inactive", "active"],
                },
            ),
            _record(
                "class_b_model",
                "class_b",
                model_b,
                target="activity",
                task_type="classification",
                known_metrics={"test": {"balanced_accuracy": 0.75}},
                inference_profile={
                    "representation_name": "morgan_rdkit_basic",
                    "class_labels": ["inactive", "active"],
                },
            ),
        ],
        source_path=tmp_path / "catalog.json",
    )
    catalog.save()
    toolkit = EnsembleToolkit(catalog=catalog)
    toolkit.backends = {
        "class_a": ClassificationFakeBackend([0.2, 0.9, 0.4]),
        "class_b": ClassificationFakeBackend([0.4, 0.7, 0.8]),
    }
    toolkit.ensemble_backend = EnsembleBackend(
        backends=toolkit.backends,
        backend_capabilities=FAKE_CAPABILITIES,
    )

    created = toolkit.create_ensemble_from_catalog("activity", task_type="classification")

    payload = json.loads(Path(created["model_path"]).read_text())
    assert payload["ensemble_kind"] == "catalog_consensus_classification"
    assert payload["aggregation_strategy"] == "probability_mean"
    assert payload["class_labels"] == ["inactive", "active"]

    test_csv = tmp_path / "classification_test.csv"
    pd.DataFrame(
        {
            "smiles": ["CC", "CCC", "CCN"],
            "activity": ["inactive", "active", "active"],
        }
    ).to_csv(test_csv, index=False)

    evaluated = toolkit.evaluate_ensemble_on_dataset(created["model_id"], str(test_csv), "activity")

    assert evaluated["ensemble_metrics"]["accuracy"] == pytest.approx(1.0)
    assert evaluated["ensemble_metrics"]["roc_auc"] == pytest.approx(1.0)
    summary = json.loads(Path(evaluated["evaluation_summary_path"]).read_text())
    assert summary["task_type"] == "classification"
    assert summary["artifacts"]["classification_comparison"] == "classification_comparison.csv"
    updated_payload = json.loads(Path(created["model_path"]).read_text())
    assert len(updated_payload["evaluations"]) == 1
