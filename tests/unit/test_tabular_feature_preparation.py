from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cs_copilot.tools.prediction.applicability_domain import (
    fit_bounding_box_domain,
    score_record_applicability_domain,
)
from cs_copilot.tools.prediction.backend import (
    InvalidPredictionInputError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from cs_copilot.tools.prediction.external_evaluation import (
    evaluate_model_on_external_dataset,
)
from cs_copilot.tools.prediction.prediction_inference_toolkit import (
    PredictionInferenceToolkit,
)
from cs_copilot.tools.prediction.tabular_feature_preparation import (
    TabularFeaturePreparationService,
    TabularPreparationRequest,
    current_rdkit_version,
    require_tabular_model_contract,
)
from cs_copilot.tools.prediction.tabular_representations import (
    MorganBinaryFingerprint,
    MorganCountFingerprint,
    RDKitDescriptors,
    get_tabular_representation,
)


def _molecular_csv(path: Path, *, with_target: bool = True) -> Path:
    payload = {
        "compound_id": ["a", "b", "c"],
        "smiles": ["CCO", "CCO", "CCN"],
    }
    if with_target:
        payload["pEC50"] = [4.0, 5.0, 6.0]
    pd.DataFrame(payload).to_csv(path, index=False)
    return path


def _prepare(
    service: TabularFeaturePreparationService,
    *,
    input_csv: Path,
    output_dir: Path,
    representation_name: str,
    target_columns: list[str] | None = None,
):
    return service.prepare(
        TabularPreparationRequest(
            input_csv=str(input_csv),
            output_dir=str(output_dir),
            kind="generated",
            representation_name=representation_name,
            smiles_column="smiles",
            target_columns=list(target_columns or []),
            base_columns_to_keep=["compound_id"],
            n_jobs=1,
            purpose="training" if target_columns else "inference",
        )
    )


def _tabular_record(
    tmp_path: Path,
    *,
    contract: dict,
    backend_name: str = "tabicl",
    applicability_domain: dict | None = None,
) -> PredictionModelRecord:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_path = tmp_path / f"{backend_name}.pkl"
    model_path.write_bytes(b"model")
    metadata_path = tmp_path / f"{backend_name}_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_id": f"{backend_name}_model",
                "backend_name": backend_name,
                "tabular_representation_contract": contract,
                "task": {
                    "task_type": "regression",
                    "smiles_columns": ["smiles"],
                    "target_columns": ["pEC50"],
                },
                "artifacts": {"model_path": model_path.name},
            }
        )
        + "\n"
    )
    return PredictionModelRecord(
        model_id=f"{backend_name}_model",
        backend_name=backend_name,
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        task=PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=["pEC50"],
        ),
        applicability_domain=dict(applicability_domain or {}),
        tabular_representation_contract=contract,
    )


class _CapturingTabularBackend:
    backend_name = "tabicl"

    def __init__(self) -> None:
        self.inputs: list[pd.DataFrame] = []

    def predict_from_csv(
        self,
        input_csv,
        model_record,
        preds_path,
        *,
        return_uncertainty=False,
    ):
        frame = pd.read_csv(input_csv)
        contract = require_tabular_model_contract(model_record)
        assert list(frame[contract.feature_columns].columns) == contract.feature_columns
        self.inputs.append(frame)
        pd.DataFrame(
            {
                "smiles": frame["smiles"],
                "prediction": [float(index) for index in range(len(frame))],
            }
        ).to_csv(preds_path, index=False)
        return {"preds_path": str(preds_path), "rows": len(frame)}


class _Catalog:
    def refresh_from_internal_store(self, *, persist=False):
        return None


class _Registry:
    def __init__(self, record: PredictionModelRecord):
        self.record = record
        self.catalog = _Catalog()

    def resolve_record(self, model_id, agent):
        assert model_id == self.record.model_id
        return self.record


def test_representation_registry_is_composed_from_immutable_components():
    rdkit = get_tabular_representation("rdkit_all")
    binary = get_tabular_representation("morgan_only")
    count = get_tabular_representation("morgan_count_only")
    combined = get_tabular_representation("morgan_binary_count_rdkit_all")

    assert rdkit.components == (RDKitDescriptors(descriptor_set="all"),)
    assert binary.components == (MorganBinaryFingerprint(),)
    assert count.components == (MorganCountFingerprint(),)
    assert combined.components == (
        MorganBinaryFingerprint(),
        MorganCountFingerprint(),
        RDKitDescriptors(descriptor_set="all"),
    )
    assert combined.as_dict()["description"].startswith("Binary ECFP/Morgan fingerprint bits")


def test_service_generates_all_registered_component_compositions_and_preserves_rows(
    tmp_path,
):
    source = _molecular_csv(tmp_path / "molecules.csv")
    service = TabularFeaturePreparationService(cache_root=tmp_path / "cache")

    results = {
        name: _prepare(
            service,
            input_csv=source,
            output_dir=tmp_path / name,
            representation_name=name,
            target_columns=["pEC50"],
        )
        for name in (
            "rdkit_all",
            "morgan_only",
            "morgan_count_only",
            "morgan_binary_count_rdkit_all",
            "morgan_rdkit_all",
        )
    }

    assert len(results["rdkit_all"].feature_columns) > 200
    assert all(column.startswith("desc_") for column in results["rdkit_all"].feature_columns)
    assert len(results["morgan_only"].feature_columns) == 2048
    assert all(column.startswith("fp_") for column in results["morgan_only"].feature_columns)
    assert len(results["morgan_count_only"].feature_columns) == 2048
    assert all(column.startswith("cfp_") for column in results["morgan_count_only"].feature_columns)
    assert len(results["morgan_binary_count_rdkit_all"].feature_columns) == 4096 + len(
        results["rdkit_all"].feature_columns
    )
    assert len(results["morgan_rdkit_all"].feature_columns) == 2048 + len(
        results["rdkit_all"].feature_columns
    )

    frame = pd.read_csv(results["morgan_binary_count_rdkit_all"].prepared_csv)
    assert frame["__qsar_row_id"].tolist() == [0, 1, 2]
    assert frame["compound_id"].tolist() == ["a", "b", "c"]
    assert frame["smiles"].tolist()[:2] == ["CCO", "CCO"]
    assert len(frame) == 3


def test_cache_hit_corruption_and_manual_deletion_are_reconstructible(tmp_path):
    source = _molecular_csv(tmp_path / "molecules.csv")
    cache_root = tmp_path / "cache"
    service = TabularFeaturePreparationService(cache_root=cache_root)

    first = _prepare(
        service,
        input_csv=source,
        output_dir=tmp_path / "first",
        representation_name="morgan_count_only",
        target_columns=["pEC50"],
    )
    expected = pd.read_csv(first.prepared_csv).copy()
    second = _prepare(
        service,
        input_csv=source,
        output_dir=tmp_path / "second",
        representation_name="morgan_count_only",
        target_columns=["pEC50"],
    )
    assert second.cache_status == "reused_from_cache"
    assert second.cache_misses == 0

    assembled = cache_root / "assembled" / f"morgan_count_only_{second.cache_key}.csv"
    assembled.write_text("corrupt\n")
    rebuilt = _prepare(
        service,
        input_csv=source,
        output_dir=tmp_path / "rebuilt",
        representation_name="morgan_count_only",
        target_columns=["pEC50"],
    )
    assert rebuilt.cache_status == "generated"
    pd.testing.assert_frame_equal(pd.read_csv(rebuilt.prepared_csv), expected)

    shutil.rmtree(cache_root)
    after_deletion = _prepare(
        service,
        input_csv=source,
        output_dir=tmp_path / "after-deletion",
        representation_name="morgan_count_only",
        target_columns=["pEC50"],
    )
    assert after_deletion.cache_status == "generated"
    pd.testing.assert_frame_equal(pd.read_csv(after_deletion.prepared_csv), expected)


def test_concurrent_requests_share_one_cache_entry(tmp_path):
    source = _molecular_csv(tmp_path / "molecules.csv")
    cache_root = tmp_path / "cache"

    def run(index: int):
        service = TabularFeaturePreparationService(cache_root=cache_root)
        return _prepare(
            service,
            input_csv=source,
            output_dir=tmp_path / f"worker-{index}",
            representation_name="morgan_only",
            target_columns=["pEC50"],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, [1, 2]))

    assert sorted(result.cache_status for result in results) == [
        "generated",
        "reused_from_cache",
    ]
    assert results[0].recipe_signature == results[1].recipe_signature
    assert results[0].feature_columns == results[1].feature_columns


def test_invalid_smiles_are_never_silently_removed(tmp_path):
    source = tmp_path / "invalid.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "not-a-smiles"],
            "pEC50": [4.0, 5.0],
        }
    ).to_csv(source, index=False)
    service = TabularFeaturePreparationService(cache_root=tmp_path / "cache")

    with pytest.raises(InvalidPredictionInputError, match=r"1 row\(s\)"):
        _prepare(
            service,
            input_csv=source,
            output_dir=tmp_path / "out",
            representation_name="rdkit_all",
            target_columns=["pEC50"],
        )


def test_precomputed_features_are_strict_and_do_not_require_rdkit_compatibility(
    tmp_path,
):
    source = tmp_path / "precomputed.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN"],
            "feature_b": [1.0, 2.0],
            "feature_a": [3.0, 4.0],
            "category": ["x", "y"],
        }
    ).to_csv(source, index=False)
    service = TabularFeaturePreparationService(cache_root=tmp_path / "cache")
    result = service.prepare(
        TabularPreparationRequest(
            input_csv=str(source),
            output_dir=str(tmp_path / "out"),
            kind="precomputed",
            representation_name="precomputed_tabular",
            expected_feature_columns=["feature_a", "category", "feature_b"],
            categorical_feature_columns=["category"],
            purpose="training",
        )
    )
    prepared = pd.read_csv(result.prepared_csv)

    assert result.cache_status == "not_applicable"
    assert result.contract.rdkit_version == "not_applicable"
    assert result.contract.categorical_feature_columns == ["category"]
    assert list(prepared.columns[-3:]) == ["feature_a", "category", "feature_b"]

    missing_source = tmp_path / "missing.csv"
    pd.DataFrame({"smiles": ["CCO"], "feature_a": [1.0]}).to_csv(
        missing_source,
        index=False,
    )
    with pytest.raises(InvalidPredictionInputError, match="missing feature columns"):
        service.prepare(
            TabularPreparationRequest(
                input_csv=str(missing_source),
                output_dir=str(tmp_path / "missing-out"),
                kind="precomputed",
                representation_name="precomputed_tabular",
                expected_feature_columns=["feature_a", "feature_b"],
            )
        )


def test_old_or_runtime_incompatible_tabular_contracts_are_rejected(tmp_path):
    source = _molecular_csv(tmp_path / "train.csv")
    service = TabularFeaturePreparationService(cache_root=tmp_path / "cache")
    trained = _prepare(
        service,
        input_csv=source,
        output_dir=tmp_path / "train",
        representation_name="rdkit_all",
        target_columns=["pEC50"],
    )

    old_record = _tabular_record(tmp_path, contract={})
    with pytest.raises(InvalidPredictionInputError, match="pre-0.4.0"):
        service.prepare_for_model(
            input_csv=str(source),
            output_dir=str(tmp_path / "old"),
            record=old_record,
            purpose="inference",
        )

    incompatible = trained.contract.model_dump(mode="json")
    incompatible["rdkit_version"] = "0.0.0"
    incompatible_record = _tabular_record(
        tmp_path / "incompatible",
        contract=incompatible,
    )
    with pytest.raises(
        InvalidPredictionInputError,
        match=f"expected RDKit 0.0.0, installed {current_rdkit_version()}",
    ):
        service.prepare_for_model(
            input_csv=str(source),
            output_dir=str(tmp_path / "incompatible-out"),
            record=incompatible_record,
            purpose="inference",
        )

    bad_signature = trained.contract.model_dump(mode="json")
    bad_signature["recipe_signature"] = "sha256:incorrect"
    bad_record = _tabular_record(tmp_path / "bad-signature", contract=bad_signature)
    with pytest.raises(
        InvalidPredictionInputError,
        match="recipe_signature does not match",
    ):
        require_tabular_model_contract(bad_record)


def test_tabicl_inference_from_new_smiles_rebuilds_features_without_training_cache(
    tmp_path,
):
    training_csv = _molecular_csv(tmp_path / "training.csv")
    cache_root = tmp_path / "cache"
    service = TabularFeaturePreparationService(cache_root=cache_root)
    trained = _prepare(
        service,
        input_csv=training_csv,
        output_dir=tmp_path / "training-features",
        representation_name="rdkit_all",
        target_columns=["pEC50"],
    )
    record = _tabular_record(
        tmp_path,
        contract=trained.contract.model_dump(mode="json"),
    )
    shutil.rmtree(cache_root)

    new_csv = tmp_path / "new-molecules.csv"
    pd.DataFrame({"SMILES": ["c1ccccc1", "CCCl"]}).to_csv(new_csv, index=False)
    backend = _CapturingTabularBackend()
    toolkit = PredictionInferenceToolkit(
        backends={"tabicl": backend},
        registry_toolkit=_Registry(record),
        tabular_feature_service=service,
        register_tools=False,
    )
    result = toolkit.predict_from_csv(
        model_id=record.model_id,
        input_csv=str(new_csv),
        smiles_column="SMILES",
        preds_path=str(tmp_path / "predictions.csv"),
        agent=SimpleNamespace(session_state={}),
    )

    assert Path(result["preds_path"]).exists()
    assert len(backend.inputs) == 1
    assert (
        list(backend.inputs[0][trained.contract.feature_columns].columns)
        == trained.contract.feature_columns
    )
    assert len(backend.inputs[0]) == 2


def test_external_evaluation_uses_the_same_tabular_preparation_service(tmp_path):
    training_csv = _molecular_csv(tmp_path / "training.csv")
    service = TabularFeaturePreparationService(cache_root=tmp_path / "cache")
    trained = _prepare(
        service,
        input_csv=training_csv,
        output_dir=tmp_path / "training-features",
        representation_name="morgan_count_only",
        target_columns=["pEC50"],
    )
    record = _tabular_record(
        tmp_path,
        contract=trained.contract.model_dump(mode="json"),
    )
    shutil.rmtree(tmp_path / "cache")
    external_csv = tmp_path / "external.csv"
    pd.DataFrame(
        {
            "smiles": ["CCCl", "CCBr", "CO"],
            "pEC50": [1.0, 2.0, 3.0],
        }
    ).to_csv(external_csv, index=False)
    backend = _CapturingTabularBackend()

    result = evaluate_model_on_external_dataset(
        record=record,
        backend=backend,
        test_csv=str(external_csv),
        target_columns=["pEC50"],
        tabular_feature_service=service,
    )

    assert Path(result["predictions_path"]).exists()
    assert len(backend.inputs) == 1
    assert (
        list(backend.inputs[0][trained.contract.feature_columns].columns)
        == trained.contract.feature_columns
    )


def test_applicability_domain_reuses_a_prepared_matrix_without_regeneration(
    tmp_path,
):
    source = _molecular_csv(tmp_path / "training.csv")
    service = TabularFeaturePreparationService(cache_root=tmp_path / "cache")
    prepared = _prepare(
        service,
        input_csv=source,
        output_dir=tmp_path / "features",
        representation_name="rdkit_all",
        target_columns=["pEC50"],
    )
    frame = pd.read_csv(prepared.prepared_csv)
    ad_features = [column for column in prepared.feature_columns if frame[column].notna().all()][:3]
    applicability_domain = fit_bounding_box_domain(
        feature_frame=frame,
        feature_columns=ad_features,
        output_dir=tmp_path / "ad",
        model_id="tabicl_model",
        feature_space="rdkit_all",
        representation_name="rdkit_all",
    )
    record = _tabular_record(
        tmp_path,
        contract=prepared.contract.model_dump(mode="json"),
        applicability_domain=applicability_domain,
    )

    class _FailingService:
        def prepare_for_model(self, **kwargs):
            raise AssertionError("AD must reuse the already prepared matrix")

    result = score_record_applicability_domain(
        record=record,
        input_csv=str(source),
        output_dir=tmp_path / "scores",
        score_label="reuse",
        prepared_feature_csv=prepared.prepared_csv,
        tabular_feature_service=_FailingService(),
    )

    assert result["available"] is True
    assert len(result["scores"]) == 3


def test_tabular_backends_do_not_import_feature_generation():
    root = Path(__file__).parents[2] / "src" / "cs_copilot" / "tools" / "prediction"
    for filename in ("lightgbm_backend.py", "tabicl_backend.py"):
        source = (root / filename).read_text()
        assert "MolecularFeatureToolkit" not in source
        assert "smiles_to_morgan_fingerprints" not in source
        assert "smiles_to_rdkit_descriptors" not in source
