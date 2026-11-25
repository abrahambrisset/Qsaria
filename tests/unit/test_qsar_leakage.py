import pandas as pd
import pytest

from cs_copilot.tools.prediction.leakage import audit_molecular_overlap, audit_training_split
from cs_copilot.tools.prediction.training_orchestration import audit_selection_validation


def test_overlap_counts_evaluation_rows_separately_from_similar_analogues():
    report = audit_molecular_overlap(
        ["CCO", "C" * 40, "C" * 120, "CCO"],
        ["OCC", "CCO", "C" * 41, "C" * 121, "not a molecule"],
    )

    assert report["status"] == "partial"
    assert report["training_rows"] == 4
    assert report["evaluation_valid_rows"] == 4
    assert report["exact_overlap_count"] == 2
    assert report["near_duplicate_counts"] == {"0.95": 2, "0.99": 1}
    assert report["nearest_neighbor_similarity"]["max"] == 1.0
    assert report["nearest_neighbor_similarity"]["median"] == pytest.approx(0.9958449)


def test_unavailable_reference_never_reports_zero_overlap():
    report = audit_molecular_overlap([None], ["CCO"])

    assert report["status"] == "unavailable"
    assert report["training_valid_rows"] == 0
    assert report["exact_overlap_count"] is None


def test_training_split_audits_each_heldout_role_and_persists_only_fit_structures(tmp_path):
    frame = pd.DataFrame({"structure": ["C(C)O", "CCN", "NCC", "OCC", "c1ccccc1"]})

    result = audit_training_split(
        frame, "structure", {"train": [0, 1], "val": [2], "test": [3, 4]}, tmp_path
    )

    assert result["leakage"]["validation"]["exact_overlap_count"] == 1
    assert result["leakage"]["test"]["exact_overlap_count"] == 1
    assert result["leakage"]["test"]["evaluation_rows"] == 2
    assert pd.read_csv(result["leakage_reference_path"])["smiles"].tolist() == ["CCO", "CCN"]


def test_selection_audit_uses_pre_refit_training_rows():
    frame = pd.DataFrame({"SMILES": ["OCC", "NCC", "CCO", "CCC"]})
    split = {"train": [0], "val": [1, 2], "test": [3]}

    audit = audit_selection_validation(frame, "smiles", split)

    assert audit["training_rows"] == 1
    assert audit["evaluation_rows"] == 2
    assert audit["exact_overlap_count"] == 1
