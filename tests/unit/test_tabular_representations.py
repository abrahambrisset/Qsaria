from __future__ import annotations

from cs_copilot.tools.prediction.tabular_representations import (
    AUTOMATIC_TABULAR_REPRESENTATION_NAMES,
    LEGACY_TABULAR_REPRESENTATION_NAMES,
    default_tabular_representation_for_protocol,
    get_tabular_representation,
    tabular_candidates_for_backend,
)


def test_modern_automatic_pack_excludes_legacy_rdkit_basic():
    assert AUTOMATIC_TABULAR_REPRESENTATION_NAMES == (
        "rdkit_all",
        "morgan_only",
        "morgan_count_only",
        "morgan_binary_count_rdkit_all",
    )
    assert "rdkit_basic_only" in LEGACY_TABULAR_REPRESENTATION_NAMES
    assert "morgan_rdkit_basic" in LEGACY_TABULAR_REPRESENTATION_NAMES


def test_representation_specs_capture_binary_count_and_rdkit_all():
    spec = get_tabular_representation("morgan_binary_count_rdkit_all")
    assert spec.use_morgan_binary is True
    assert spec.use_morgan_count is True
    assert spec.use_rdkit is True
    assert spec.descriptor_set == "all"


def test_fast_local_default_uses_rdkit_all():
    assert default_tabular_representation_for_protocol("fast_local") == "rdkit_all"


def test_tabular_candidates_use_modern_pack_by_default():
    candidates = tabular_candidates_for_backend("lightgbm")
    assert [candidate["candidate_id"] for candidate in candidates] == [
        "lightgbm_rdkit_all",
        "lightgbm_morgan_only",
        "lightgbm_morgan_count_only",
        "lightgbm_morgan_binary_count_rdkit_all",
    ]
