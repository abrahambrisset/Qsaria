from cs_copilot.tools.prediction import catalog as catalog_module
from cs_copilot.tools.prediction.backend import PredictionModelRecord, PredictionTaskSpec
from cs_copilot.tools.prediction.catalog import PredictionModelCatalog


def test_catalog_load_bootstraps_missing_local_catalog(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", tmp_path / "internal")
    catalog_path = tmp_path / "local" / "model_catalog.json"

    catalog = PredictionModelCatalog.load(str(catalog_path))

    assert catalog.records == []
    assert catalog.schema_version == 2
    assert catalog_path.exists()
    assert catalog_path.read_text() == '{\n  "schema_version": 2,\n  "models": []\n}\n'


def test_prediction_model_record_roundtrip_with_catalog_metadata():
    record = PredictionModelRecord(
        model_id="demo_model",
        backend_name="chemprop",
        model_path="/tmp/demo_model.pt",
        display_name="Demo Model",
        description="Demo description",
        version="1.0.0",
        status="validated",
        owner="qa",
        source="unit_test",
        domain_summary="small molecules for solubility regression",
        strengths=["fast"],
        limitations=["narrow domain"],
        recommended_for=["solubility"],
        not_recommended_for=["toxicity"],
        known_metrics={"rmse": 0.4},
        training_data_summary={"rows": 120},
        inference_profile={"latency_tier": "low"},
        selection_hints={"endpoint_keywords": ["solubility"]},
        task=PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=["solubility"],
        ),
    )

    restored = PredictionModelRecord.from_dict(record.as_dict())

    assert restored.model_id == "demo_model"
    assert restored.status == "validated"
    assert restored.known_metrics["rmse"] == 0.4
    assert restored.selection_hints["endpoint_keywords"] == ["solubility"]


def test_catalog_recommend_prefers_target_matching_entry(tmp_path):
    model_file = tmp_path / "demo.pt"
    model_file.write_text("placeholder")

    catalog_path = tmp_path / "model_catalog.json"
    catalog_path.write_text("""
{
  "schema_version": 1,
  "models": [
    {
      "model_id": "solubility_model",
      "backend_name": "chemprop",
      "model_path": "__MODEL_PATH__",
      "display_name": "Solubility Model",
      "status": "validated",
      "recommended_for": ["aqueous solubility"],
      "task": {
        "task_type": "regression",
        "smiles_columns": ["smiles"],
        "target_columns": ["solubility"],
        "reaction_columns": [],
        "uncertainty_method": null,
        "calibration_method": null
      },
      "tags": {}
    },
    {
      "model_id": "permeability_model",
      "backend_name": "chemprop",
      "model_path": "__MODEL_PATH__",
      "display_name": "Permeability Model",
      "status": "validated",
      "recommended_for": ["permeability"],
      "task": {
        "task_type": "regression",
        "smiles_columns": ["smiles"],
        "target_columns": ["permeability"],
        "reaction_columns": [],
        "uncertainty_method": null,
        "calibration_method": null
      },
      "tags": {}
    }
  ]
}
""".replace("__MODEL_PATH__", str(model_file)))

    catalog = PredictionModelCatalog.load(str(catalog_path))
    recommendation = catalog.recommend(
        task_type="regression",
        target_hint="solubility",
        backend_available=True,
    )

    assert recommendation["selected_model"]["model_id"] == "solubility_model"
    assert recommendation["selected_model"]["score"] >= recommendation["alternatives"][0]["score"]


def test_catalog_search_excludes_missing_paths_by_default(tmp_path):
    catalog_path = tmp_path / "model_catalog.json"
    catalog_path.write_text("""
{
  "schema_version": 1,
  "models": [
    {
      "model_id": "missing_model",
      "backend_name": "chemprop",
      "model_path": "/tmp/does-not-exist.pt",
      "status": "validated",
      "task": {
        "task_type": "regression",
        "smiles_columns": ["smiles"],
        "target_columns": ["solubility"],
        "reaction_columns": [],
        "uncertainty_method": null,
        "calibration_method": null
      },
      "tags": {}
    }
  ]
}
""")

    catalog = PredictionModelCatalog.load(str(catalog_path))

    assert catalog.search(task_type="regression") == []
