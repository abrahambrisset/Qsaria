from __future__ import annotations

import pytest

from cs_copilot.tools.prediction.tabular_representations import (
    AUTOMATIC_TABULAR_REPRESENTATION_NAMES,
    LEGACY_TABULAR_REPRESENTATION_NAMES,
    TABICL_AUTOMATIC_TABULAR_REPRESENTATION_NAMES,
    default_tabular_representation_for_protocol,
    get_tabular_representation,
    tabular_candidates_for_backend,
)


def test_lightgbm_automatic_pack_keeps_morgan_variants_and_excludes_legacy_rdkit_basic():
    assert AUTOMATIC_TABULAR_REPRESENTATION_NAMES == (
        "rdkit_all",
        "morgan_only",
        "morgan_count_only",
        "morgan_binary_count_rdkit_all",
    )
    assert "rdkit_basic_only" in LEGACY_TABULAR_REPRESENTATION_NAMES
    assert "morgan_rdkit_basic" in LEGACY_TABULAR_REPRESENTATION_NAMES


def test_chemeleon_rdkit_all_is_tabicl_preferred_representation():
    spec = get_tabular_representation("chemeleon_rdkit_all")
    assert spec.use_chemeleon is True
    assert spec.use_rdkit is True
    assert spec.descriptor_set == "all"
    assert spec.tabicl_preferred is True
    assert spec.tabicl_compatible is True
    assert spec.high_dimensional is False


def test_morgan_count_representation_is_not_tabicl_compatible():
    spec = get_tabular_representation("morgan_count_only")
    assert spec.use_morgan_count is True
    assert spec.high_dimensional is True
    assert spec.tabicl_compatible is False


def test_fast_local_default_uses_rdkit_all_for_generic_tabular_backends():
    assert default_tabular_representation_for_protocol("fast_local") == "rdkit_all"


def test_tabicl_default_uses_chemeleon_rdkit_all_for_all_protocols():
    assert (
        default_tabular_representation_for_protocol("fast_local", backend_name="tabicl")
        == "chemeleon_rdkit_all"
    )
    assert (
        default_tabular_representation_for_protocol("standard_qsar", backend_name="tabicl")
        == "chemeleon_rdkit_all"
    )


def test_lightgbm_candidates_use_morgan_rdkit_pack_by_default():
    candidates = tabular_candidates_for_backend("lightgbm")
    assert [candidate["candidate_id"] for candidate in candidates] == [
        "lightgbm_rdkit_all",
        "lightgbm_morgan_only",
        "lightgbm_morgan_count_only",
        "lightgbm_morgan_binary_count_rdkit_all",
    ]


def test_tabicl_candidates_prefer_chemeleon_rdkit_and_exclude_morgan():
    candidates = tabular_candidates_for_backend("tabicl")
    assert TABICL_AUTOMATIC_TABULAR_REPRESENTATION_NAMES == (
        "chemeleon_rdkit_all",
        "rdkit_all",
    )
    assert [candidate["candidate_id"] for candidate in candidates] == [
        "tabicl_chemeleon_rdkit_all",
        "tabicl_rdkit_all",
    ]
    assert all(candidate["representation_automatic"] for candidate in candidates)


def test_tabicl_candidates_reject_high_dimensional_morgan_representation():
    with pytest.raises(ValueError, match="high-dimensional representation"):
        tabular_candidates_for_backend("tabicl", representation_names=["morgan_count_only"])
