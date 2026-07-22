from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pandas as pd
import pytest

_IDENTITY_PATH = Path(__file__).resolve().parents[2] / "src/cs_copilot/tools/curation/identity.py"
_IDENTITY_SPEC = importlib.util.spec_from_file_location(
    "curation_identity_for_tests", _IDENTITY_PATH
)
assert _IDENTITY_SPEC and _IDENTITY_SPEC.loader
_IDENTITY_MODULE = importlib.util.module_from_spec(_IDENTITY_SPEC)
try:
    _IDENTITY_SPEC.loader.exec_module(_IDENTITY_MODULE)
except ModuleNotFoundError as exc:
    _IDENTITY_IMPORT_ERROR = exc
    strip_stereochemistry_from_smiles = None
else:
    _IDENTITY_IMPORT_ERROR = None
    strip_stereochemistry_from_smiles = _IDENTITY_MODULE.strip_stereochemistry_from_smiles


def _load_curation_toolkit():
    if _IDENTITY_IMPORT_ERROR is not None:
        pytest.skip(f"RDKit unavailable: {_IDENTITY_IMPORT_ERROR}")
    pytest.importorskip("agno")
    from cs_copilot.tools.curation import dataset_curation_toolkit as curation_module
    from cs_copilot.tools.curation.dataset_curation_toolkit import DatasetCurationToolkit

    return curation_module, DatasetCurationToolkit


def _fake_chembl_backend(raw_smiles: pd.Series):
    assert strip_stereochemistry_from_smiles is not None
    rows = []
    for row_index, smiles in raw_smiles.items():
        standardized = smiles
        identity = strip_stereochemistry_from_smiles(smiles) if isinstance(smiles, str) else None
        rows.append(
            {
                "row_index": row_index,
                "raw_smiles": smiles,
                "chembl_input_smiles": smiles,
                "standardized_smiles": standardized,
                "qsar_identity_smiles": identity,
                "curation_identity_key": identity,
                "curation_identity_key_type": "qsar_identity_smiles",
                "curation_backend_status": "ok" if identity else "invalid_smiles",
                "checker_issues": "",
                "checker_max_penalty": 0,
                "parent_structure_changed": False,
                "stereochemistry_removed_for_identity": identity != standardized,
            }
        )
    return {
        "backend_name": "chembl_structure_v1",
        "identity_column": "curation_identity_key",
        "standardization_map": pd.DataFrame(rows),
    }


def _failing_chembl_backend(raw_smiles: pd.Series, failed_smiles: set[str]):
    payload = _fake_chembl_backend(raw_smiles)
    standardization_map = payload["standardization_map"]
    failed = standardization_map["raw_smiles"].isin(failed_smiles)
    for column in ("standardized_smiles", "qsar_identity_smiles", "curation_identity_key"):
        standardization_map.loc[failed, column] = None
    standardization_map.loc[failed, "curation_backend_status"] = "standardization_failed"
    standardization_map.loc[failed, "checker_issues"] = "standardization_error:test"
    return payload


def test_curation_public_contract_has_no_backend_selector() -> None:
    _curation_module, DatasetCurationToolkit = _load_curation_toolkit()

    assert (
        "curation_backend"
        not in inspect.signature(DatasetCurationToolkit.curate_qsar_dataset).parameters
    )


def test_curation_initialization_requires_chembl_dependency(monkeypatch) -> None:
    curation_module, DatasetCurationToolkit = _load_curation_toolkit()

    def missing_dependency():
        raise RuntimeError("mandatory chembl dependency missing")

    monkeypatch.setattr(
        curation_module,
        "ensure_chembl_structure_pipeline_available",
        missing_dependency,
    )
    with pytest.raises(RuntimeError, match="mandatory chembl"):
        DatasetCurationToolkit()


def test_one_chembl_row_failure_preserves_partial_dataset(tmp_path, monkeypatch) -> None:
    curation_module, DatasetCurationToolkit = _load_curation_toolkit()
    monkeypatch.setattr(
        curation_module,
        "standardize_with_chembl_structure_v1",
        lambda values: _failing_chembl_backend(values, {"CCF"}),
    )
    source = tmp_path / "partial.csv"
    output = tmp_path / "partial_curated.csv"
    pd.DataFrame({"smiles": ["CCO", "CCF", "CCC"], "pEC50": [4.0, 5.0, 6.0]}).to_csv(
        source, index=False
    )

    result = DatasetCurationToolkit().curate_qsar_dataset(
        dataset_path=str(source),
        task_type="regression",
        smiles_column="smiles",
        target_columns=["pEC50"],
        output_csv=str(output),
    )

    assert result["ready_for_qsar"] is True
    assert result["rows_out"] == 2
    assert result["invalid_smiles_removed"] == 1
    diagnostics = pd.read_csv(result["curation_artifacts"]["standardization_map_csv"])
    assert "standardization_failed" in set(diagnostics["curation_backend_status"])


def test_all_chembl_row_failures_block_dataset(tmp_path, monkeypatch) -> None:
    curation_module, DatasetCurationToolkit = _load_curation_toolkit()
    monkeypatch.setattr(
        curation_module,
        "standardize_with_chembl_structure_v1",
        lambda values: _failing_chembl_backend(values, set(values.dropna())),
    )
    source = tmp_path / "unusable.csv"
    output = tmp_path / "unusable_curated.csv"
    pd.DataFrame({"smiles": ["CCO", "CCC"], "pEC50": [4.0, 6.0]}).to_csv(source, index=False)

    result = DatasetCurationToolkit().curate_qsar_dataset(
        dataset_path=str(source),
        task_type="regression",
        smiles_column="smiles",
        target_columns=["pEC50"],
        output_csv=str(output),
    )

    assert result["ready_for_qsar"] is False
    assert result["rows_out"] == 0
    assert "Dataset is empty after curation." in result["blocking_issues"]


def test_stereo_strip_identity_collapses_enantiomeric_smiles() -> None:
    if strip_stereochemistry_from_smiles is None:
        pytest.skip(f"RDKit unavailable: {_IDENTITY_IMPORT_ERROR}")
    assert strip_stereochemistry_from_smiles("C[C@H](O)N") == strip_stereochemistry_from_smiles(
        "C[C@@H](O)N"
    )


def test_chembl_backend_stereo_identity_aggregates_close_duplicates(tmp_path, monkeypatch) -> None:
    curation_module, DatasetCurationToolkit = _load_curation_toolkit()
    monkeypatch.setattr(
        curation_module, "standardize_with_chembl_structure_v1", _fake_chembl_backend
    )
    source = tmp_path / "stereo.csv"
    output = tmp_path / "curated.csv"
    pd.DataFrame(
        {
            "SMILES": ["C[C@H](O)N", "C[C@@H](O)N", "CCO"],
            "pEC50": [5.0, 5.2, 4.0],
        }
    ).to_csv(source, index=False)

    result = DatasetCurationToolkit().curate_qsar_dataset(
        dataset_path=str(source),
        task_type="regression",
        smiles_column="SMILES",
        target_columns=["pEC50"],
        output_csv=str(output),
    )

    curated = pd.read_csv(output)
    assert result["duplicate_groups_aggregated"] == 1
    assert result["duplicate_conflicting_groups"] == 0
    assert len(curated) == 2
    assert (
        round(float(curated.loc[curated["curation_identity_key"] != "CCO", "pEC50"].iloc[0]), 2)
        == 5.1
    )
    assert result["curation_artifacts"]["standardization_map_csv"].endswith(
        "curation_standardization_map.csv"
    )


def test_chembl_backend_stereo_identity_removes_conflicting_duplicates(
    tmp_path, monkeypatch
) -> None:
    curation_module, DatasetCurationToolkit = _load_curation_toolkit()
    monkeypatch.setattr(
        curation_module, "standardize_with_chembl_structure_v1", _fake_chembl_backend
    )
    source = tmp_path / "stereo_conflict.csv"
    output = tmp_path / "curated_conflict.csv"
    pd.DataFrame(
        {
            "SMILES": ["C[C@H](O)N", "C[C@@H](O)N", "CCO"],
            "pEC50": [5.0, 6.0, 4.0],
        }
    ).to_csv(source, index=False)

    result = DatasetCurationToolkit().curate_qsar_dataset(
        dataset_path=str(source),
        task_type="regression",
        smiles_column="SMILES",
        target_columns=["pEC50"],
        output_csv=str(output),
    )

    curated = pd.read_csv(output)
    assert result["duplicate_groups_aggregated"] == 0
    assert result["duplicate_conflicting_groups"] == 1
    assert len(curated) == 1
    assert curated["smiles"].iloc[0] == "CCO"


def test_chembl_backend_classification_aggregates_concordant_duplicates(
    tmp_path, monkeypatch
) -> None:
    curation_module, DatasetCurationToolkit = _load_curation_toolkit()
    monkeypatch.setattr(
        curation_module, "standardize_with_chembl_structure_v1", _fake_chembl_backend
    )
    source = tmp_path / "classification.csv"
    output = tmp_path / "classification_curated.csv"
    pd.DataFrame(
        {
            "SMILES": ["C[C@H](O)N", "C[C@@H](O)N", "CCO"],
            "active": ["yes", "yes", "no"],
        }
    ).to_csv(source, index=False)

    result = DatasetCurationToolkit().curate_qsar_dataset(
        dataset_path=str(source),
        task_type="classification",
        smiles_column="SMILES",
        target_columns=["active"],
        output_csv=str(output),
    )

    curated = pd.read_csv(output)
    assert result["duplicate_groups_aggregated"] == 1
    assert result["duplicate_conflicting_groups"] == 0
    assert len(curated) == 2
    assert (
        result["target_data_quality"]["classification_target_summary"]["active"]["class_count"] == 2
    )
    assert set(curated["active"]) == {"yes", "no"}


def test_chembl_backend_classification_removes_conflicting_duplicates(
    tmp_path, monkeypatch
) -> None:
    curation_module, DatasetCurationToolkit = _load_curation_toolkit()
    monkeypatch.setattr(
        curation_module, "standardize_with_chembl_structure_v1", _fake_chembl_backend
    )
    source = tmp_path / "classification_conflict.csv"
    output = tmp_path / "classification_conflict_curated.csv"
    pd.DataFrame(
        {
            "SMILES": ["C[C@H](O)N", "C[C@@H](O)N", "CCO"],
            "active": ["yes", "no", "no"],
        }
    ).to_csv(source, index=False)

    result = DatasetCurationToolkit().curate_qsar_dataset(
        dataset_path=str(source),
        task_type="classification",
        smiles_column="SMILES",
        target_columns=["active"],
        output_csv=str(output),
    )

    curated = pd.read_csv(output)
    assert result["duplicate_groups_aggregated"] == 0
    assert result["duplicate_conflicting_groups"] == 1
    assert len(curated) == 1
    assert curated["smiles"].iloc[0] == "CCO"
