from __future__ import annotations

from cs_copilot.tools.prediction.tabular_representations import (
    AUTOMATIC_TABULAR_REPRESENTATION_NAMES,
    SUPPORTED_TABULAR_REPRESENTATION_NAMES,
    default_tabular_representation_for_protocol,
    get_tabular_representation,
    tabular_candidates_for_backend,
)


def test_modern_automatic_pack_is_the_complete_automatic_contract():
    assert AUTOMATIC_TABULAR_REPRESENTATION_NAMES == (
        "rdkit_all",
        "morgan_only",
        "morgan_count_only",
    )
    assert "rdkit_basic_only" not in SUPPORTED_TABULAR_REPRESENTATION_NAMES
    assert "morgan_rdkit_basic" not in SUPPORTED_TABULAR_REPRESENTATION_NAMES


def test_representation_specs_capture_binary_count_and_rdkit_all():
    spec = get_tabular_representation("morgan_binary_count_rdkit_all")
    assert spec.automatic is False
    assert spec.use_morgan_binary is True
    assert spec.use_morgan_count is True
    assert spec.use_rdkit is True
    assert spec.descriptor_set == "all"


def test_standard_qsar_default_uses_rdkit_all():
    assert default_tabular_representation_for_protocol("standard_qsar") == "rdkit_all"


def test_tabular_candidates_use_modern_pack_by_default():
    candidates = tabular_candidates_for_backend("lightgbm")
    assert [candidate["candidate_id"] for candidate in candidates] == [
        "lightgbm_rdkit_all",
        "lightgbm_morgan_only",
        "lightgbm_morgan_count_only",
    ]


def test_explicit_combined_representation_remains_available():
    candidates = tabular_candidates_for_backend(
        "lightgbm",
        representation_names=["morgan_binary_count_rdkit_all"],
    )
    assert [candidate["candidate_id"] for candidate in candidates] == [
        "lightgbm_morgan_binary_count_rdkit_all"
    ]


def test_morgan_rdkit_all_remains_available():
    assert get_tabular_representation("morgan_rdkit_all").descriptor_set == "all"
