"""Tests for chemical train/test leakage detection."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from cs_copilot.tools.prediction.leakage import (
    LEAKAGE_FINGERPRINT_RADIUS,
    LEAKAGE_FINGERPRINT_SIZE,
    MAX_EXHAUSTIVE_COMPARISONS,
    SEVERITY_SIMILARITY_THRESHOLD,
    compute_leakage_report,
)

# Chemically distant, mutually dissimilar scaffolds used as the "clean split" case.
TRAIN_DISTANT: List[str] = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "C1CCCCC1",
]
TEST_DISTANT: List[str] = [
    "CCCCCCCCN",
    "c1ccc2[nH]ccc2c1",
    "O=S(=O)(N)c1ccc(Cl)cc1",
]


def _reference_tanimoto(left: str, right: str) -> float:
    """Count Tanimoto computed independently of the module under test.

    For count vectors the Tanimoto coefficient is ``sum(min) / sum(max)``.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=LEAKAGE_FINGERPRINT_RADIUS,
        fpSize=LEAKAGE_FINGERPRINT_SIZE,
    )
    vectors = [
        np.asarray(generator.GetCountFingerprint(Chem.MolFromSmiles(smiles)).ToList(), dtype=float)
        for smiles in (left, right)
    ]
    return float(np.minimum(*vectors).sum() / np.maximum(*vectors).sum())


def test_equivalent_smiles_string_is_an_exact_duplicate():
    """``OCC`` in test and ``CCO`` in train is the same molecule, not a near miss."""
    report = compute_leakage_report(["CCO", "c1ccccc1"], ["OCC", "CCCCCCCCN"])

    assert report["exact_duplicates"] == 1
    assert report["exact_duplicate_examples"] == ["CCO"]
    assert report["leakage_suspected"] is True
    assert report["max_nn_similarity"] == pytest.approx(1.0)
    # The duplicate is reported as exact, not padded into the near-duplicate examples.
    assert all(item["test_smiles"] != "CCO" for item in report["near_duplicate_examples"])


def test_single_shared_molecule_is_flagged():
    report = compute_leakage_report(
        TRAIN_DISTANT + ["CC(=O)Nc1ccc(O)cc1"],
        TEST_DISTANT + ["CC(=O)Nc1ccc(O)cc1"],
    )

    assert report["exact_duplicates"] == 1
    assert report["leakage_suspected"] is True
    assert report["severity"] != "none"
    assert report["near_duplicates"]["0.99"] >= 1


def test_disjoint_distant_sets_report_no_leakage():
    report = compute_leakage_report(TRAIN_DISTANT, TEST_DISTANT)

    assert report["exact_duplicates"] == 0
    assert report["exact_duplicate_examples"] == []
    assert report["leakage_suspected"] is False
    assert report["severity"] == "none"
    assert report["max_nn_similarity"] < 0.95
    assert report["near_duplicates"] == {"0.95": 0, "0.99": 0}
    assert report["sampling"] is None


def test_nearest_neighbour_similarity_matches_closed_form():
    """Reported statistics must equal an independent Tanimoto computation."""
    train = ["CCO"]
    test = ["CCCCCCCCN"]
    expected = _reference_tanimoto(test[0], train[0])

    report = compute_leakage_report(train, test)

    assert report["max_nn_similarity"] == pytest.approx(expected, abs=1e-12)
    assert report["mean_nn_similarity"] == pytest.approx(expected, abs=1e-12)
    assert report["median_nn_similarity"] == pytest.approx(expected, abs=1e-12)
    assert report["nn_similarity_percentiles"]["p50"] == pytest.approx(expected, abs=1e-12)
    # Sanity check on the reference itself: identical molecules score exactly 1.
    assert _reference_tanimoto("CCO", "OCC") == pytest.approx(1.0, abs=1e-12)


def test_near_duplicate_analogue_is_caught_without_being_an_exact_duplicate():
    train = ["CCCCCCCCCCCCCCCCCC(=O)O"]
    test = ["CCCCCCCCCCCCCCCCCCC(=O)O"]  # one CH2 longer

    report = compute_leakage_report(train, test, thresholds=(0.9,))

    assert report["exact_duplicates"] == 0
    assert report["near_duplicates"]["0.9"] == 1
    assert report["near_duplicate_examples"][0]["train_smiles"] == train[0]
    assert report["near_duplicate_examples"][0]["similarity"] == pytest.approx(
        _reference_tanimoto(test[0], train[0]), abs=1e-12
    )
    # Severity is graded at 0.99, so a 0.95 analogue is reported but not graded as leakage.
    assert 0.9 <= report["max_nn_similarity"] < SEVERITY_SIMILARITY_THRESHOLD
    assert report["severity"] == "none"
    assert report["leakage_suspected"] is False


def test_severity_scales_with_the_leaked_fraction():
    train = ["CCO", "c1ccccc1", "C1CCCCC1"]
    clean = [f"{'C' * size}N" for size in range(3, 30)]

    low = compute_leakage_report(train, clean + ["CCO"])
    high = compute_leakage_report(train, clean[:4] + ["CCO", "c1ccccc1"])

    assert low["severity"] == "low"
    assert high["severity"] == "high"
    assert low["exact_duplicates"] == 1
    assert high["exact_duplicates"] == 2


def test_invalid_smiles_are_counted_and_skipped():
    report = compute_leakage_report(
        ["CCO", "not-a-molecule", None],
        ["OCC", "", "ZZZ", float("nan")],
    )

    assert report["n_train"] == 3
    assert report["n_test"] == 4
    assert report["n_invalid_train"] == 2
    assert report["n_invalid_test"] == 3
    assert report["exact_duplicates"] == 1
    assert any("could not be parsed" in message for message in report["warnings"])


def test_empty_and_all_invalid_inputs_return_a_well_formed_report():
    for train, test in (([], []), (["CCO"], []), (["bogus"], ["CCO"])):
        report = compute_leakage_report(train, test)

        assert report["max_nn_similarity"] is None
        assert report["median_nn_similarity"] is None
        assert report["mean_nn_similarity"] is None
        assert report["nn_similarity_percentiles"] is None
        assert report["exact_duplicates"] == 0
        assert report["severity"] == "none"
        assert report["leakage_suspected"] is False
        assert report["near_duplicates"] == {"0.95": 0, "0.99": 0}
        assert report["warnings"]


def test_oversized_comparison_is_sampled_and_reported(monkeypatch):
    """Above the cap the report must say it sampled rather than truncate silently."""
    monkeypatch.setattr(
        "cs_copilot.tools.prediction.leakage.MAX_EXHAUSTIVE_COMPARISONS",
        4,
    )
    train = [f"{'C' * size}O" for size in range(2, 12)]
    test = [f"{'C' * size}N" for size in range(2, 12)]

    report = compute_leakage_report(train, test)

    assert report["sampling"] is not None
    assert report["sampling"]["n_train_compared"] < len(train)
    assert report["sampling"]["n_test_compared"] < len(test)
    assert any("sub-sampled" in message for message in report["warnings"])
    # Exact-duplicate detection stays exhaustive even when similarity is sampled.
    leaked = compute_leakage_report(train, test + [train[-1]])
    assert leaked["exact_duplicates"] == 1
    assert leaked["leakage_suspected"] is True


def test_sampling_is_not_triggered_by_default():
    assert MAX_EXHAUSTIVE_COMPARISONS >= 20_000 * 20_000


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        compute_leakage_report(["CCO"], ["CCO"], thresholds=(1.5,))
    with pytest.raises(ValueError):
        compute_leakage_report(["CCO"], ["CCO"], thresholds=())
