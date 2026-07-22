import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from cs_copilot.tools.prediction import catalog as catalog_module
from cs_copilot.tools.prediction.backend import PredictionModelRecord, PredictionTaskSpec
from cs_copilot.tools.prediction.catalog import PredictionModelCatalog


def _catalog_record(model_id, model_path):
    return PredictionModelRecord(
        model_id=model_id,
        backend_name="lightgbm",
        model_path=str(model_path),
        task=PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=["target"],
        ),
    )


def test_catalog_load_is_non_mutating_when_catalog_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", tmp_path / "internal")
    catalog_path = tmp_path / "local" / "model_catalog.json"

    catalog = PredictionModelCatalog.load(str(catalog_path))

    assert catalog.records == []
    assert catalog.schema_version == 2
    assert not catalog_path.exists()


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "not-json",
        json.dumps({"schema_version": 1, "models": []}),
        json.dumps({"schema_version": 2, "models": {}}),
    ],
)
def test_catalog_rejects_empty_malformed_and_pre_v2_payloads(tmp_path, payload):
    catalog_path = tmp_path / "model_catalog.json"
    catalog_path.write_text(payload)

    with pytest.raises(ValueError):
        PredictionModelCatalog.load(str(catalog_path))


def test_catalog_load_existing_is_lock_free(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", tmp_path / "internal")
    catalog_path = tmp_path / "model_catalog.json"
    catalog_path.write_text('{"schema_version": 2, "models": []}\n')
    lock_path = catalog_module.model_catalog_lock_path(catalog_path)

    assert not lock_path.exists()
    catalog = PredictionModelCatalog.load(str(catalog_path))

    assert catalog.records == []
    assert not lock_path.exists()


def test_internal_metadata_discovery_ignores_removed_root_aliases(monkeypatch, tmp_path):
    internal_root = tmp_path / "internal"
    model_root = internal_root / "legacy_aliases"
    model_root.mkdir(parents=True)
    (model_root / "best.pkl").write_text("placeholder")
    (model_root / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": "legacy_aliases",
                "backend_name": "lightgbm",
                "artifacts": {"model_path": "best.pkl"},
                "task": {
                    "task_type": "regression",
                    "smiles_columns": ["smiles"],
                    "target_columns": ["target"],
                },
                "training_data": {"rows": 99},
                "metrics": {"test": {"r2": 0.9}},
            }
        )
    )
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)

    catalog = PredictionModelCatalog.load(str(tmp_path / "missing_catalog.json"))
    record = catalog.get_model("legacy_aliases")

    assert record.training_data_summary == {}
    assert record.known_metrics == {}


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
  "schema_version": 2,
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
  "schema_version": 2,
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


def test_catalog_rebases_legacy_absolute_internal_path_to_active_runtime(monkeypatch, tmp_path):
    internal_root = tmp_path / "data" / "model_assets" / "internal"
    local_model = internal_root / "portable_model" / "model" / "best.pkl"
    local_metadata = internal_root / "portable_model" / "metadata.json"
    local_model.parent.mkdir(parents=True)
    local_model.write_text("model")
    local_metadata.write_text("{}")
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "models": [
                    {
                        "model_id": "portable_model",
                        "backend_name": "lightgbm",
                        "model_path": (
                            "/Users/example/Qsaria/data/model_assets/internal/"
                            "portable_model/model/best.pkl"
                        ),
                        "metadata_path": (
                            "/Users/example/Qsaria/data/model_assets/internal/"
                            "portable_model/metadata.json"
                        ),
                        "status": "validated",
                        "task": {
                            "task_type": "regression",
                            "smiles_columns": ["smiles"],
                            "target_columns": ["pEC50"],
                        },
                    }
                ],
            }
        )
    )

    record = PredictionModelCatalog.load(str(catalog_path)).get_model("portable_model")

    assert record.model_path == str(local_model.resolve())
    assert record.metadata_path == str(local_metadata.resolve())


def test_catalog_persists_internal_paths_relative_to_portable_root(monkeypatch, tmp_path):
    internal_root = tmp_path / "data" / "model_assets" / "internal"
    model_path = internal_root / "portable_model" / "model" / "best.pkl"
    metadata_path = internal_root / "portable_model" / "metadata.json"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("model")
    metadata_path.write_text("{}")
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    catalog_path = tmp_path / "catalog.json"
    catalog = PredictionModelCatalog(records=[], source_path=catalog_path)
    record = _catalog_record("portable_model", model_path)
    record.metadata_path = str(metadata_path)

    catalog.upsert_model(record)

    persisted = json.loads(catalog_path.read_text())["models"][0]
    assert persisted["model_path"] == (
        "data/model_assets/internal/portable_model/model/best.pkl"
    )
    assert persisted["metadata_path"] == (
        "data/model_assets/internal/portable_model/metadata.json"
    )


def test_stale_catalog_instances_merge_concurrent_upserts(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", tmp_path / "internal")
    catalog_path = tmp_path / "model_catalog.json"
    first = PredictionModelCatalog.load(str(catalog_path))
    second = PredictionModelCatalog.load(str(catalog_path))
    records = [
        _catalog_record("model_first", tmp_path / "first.pkl"),
        _catalog_record("model_second", tmp_path / "second.pkl"),
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        pairs = zip((first, second), records, strict=True)
        list(pool.map(lambda pair: pair[0].upsert_model(pair[1]), pairs))

    persisted = PredictionModelCatalog.load(str(catalog_path))
    assert [record.model_id for record in persisted.records] == ["model_first", "model_second"]
    assert not list(tmp_path.glob(".model_catalog.json.*.tmp"))
