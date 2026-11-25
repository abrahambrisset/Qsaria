"""Tests for the deterministic reference dataset.

Reminder: the targets under test are COMPUTED RDKit DESCRIPTORS, never experimental
measurements. These tests check that the generator is reproducible and that the labels
really are the descriptor they claim to be - they say nothing about model quality.
"""

from __future__ import annotations

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

from cs_copilot.tools.benchmarks.reference import (
    MIN_SPLITTABLE_ROWS,
    REFERENCE_CLASSIFICATION_PATH,
    REFERENCE_REGRESSION_PATH,
    TPSA_HIGH_THRESHOLD,
    build_reference_dataset,
    write_reference_datasets,
)

COMMITTED_PATHS = {
    "regression": REFERENCE_REGRESSION_PATH,
    "classification": REFERENCE_CLASSIFICATION_PATH,
}


def test_regenerating_is_byte_identical_to_committed_files(tmp_path):
    """The reproducibility test: regenerating must produce no diff at all."""
    regenerated = write_reference_datasets(tmp_path)

    assert set(regenerated) == set(COMMITTED_PATHS)
    for task, committed in COMMITTED_PATHS.items():
        assert committed.exists(), f"committed {task} fixture is missing: {committed}"
        assert regenerated[task].read_bytes() == committed.read_bytes(), (
            f"{committed.name} drifted from its generator. If an RDKit upgrade caused "
            "this, regenerate with `python -m cs_copilot.tools.benchmarks.reference` "
            "and commit the diff."
        )


def test_build_is_deterministic_across_calls():
    first = build_reference_dataset(task="regression", n=120)
    second = build_reference_dataset(task="regression", n=120)
    pd.testing.assert_frame_equal(first, second)
    # A prefix of a larger build, so n only truncates - it never reshuffles.
    pd.testing.assert_frame_equal(
        first, build_reference_dataset(task="regression", n=200).head(120)
    )


@pytest.mark.parametrize("task,target", [("regression", "clogp"), ("classification", "tpsa_high")])
def test_committed_file_schema_and_size(task, target):
    frame = pd.read_csv(COMMITTED_PATHS[task])

    assert list(frame.columns) == ["compound_id", "smiles", "scaffold", target]
    assert 300 <= len(frame) <= 500
    assert len(frame) > MIN_SPLITTABLE_ROWS
    assert frame["compound_id"].is_unique
    assert frame.notna().all().all()


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_every_smiles_parses_and_is_unique(task):
    frame = pd.read_csv(COMMITTED_PATHS[task])

    unparsable = [smiles for smiles in frame["smiles"] if Chem.MolFromSmiles(smiles) is None]
    assert unparsable == []
    assert frame["smiles"].is_unique
    # Stored SMILES are already RDKit-canonical, so canonicalising is a no-op.
    assert [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in frame["smiles"]] == list(
        frame["smiles"]
    )


def test_both_files_describe_the_same_molecules():
    regression = pd.read_csv(REFERENCE_REGRESSION_PATH)
    classification = pd.read_csv(REFERENCE_CLASSIFICATION_PATH)
    pd.testing.assert_frame_equal(
        regression[["compound_id", "smiles", "scaffold"]],
        classification[["compound_id", "smiles", "scaffold"]],
    )


def test_classification_target_is_balanced_and_binary():
    frame = pd.read_csv(REFERENCE_CLASSIFICATION_PATH)

    assert set(frame["tpsa_high"].unique()) == {0, 1}
    share = frame["tpsa_high"].mean()
    assert 0.25 < share < 0.75, f"tpsa_high is {share:.1%} positive, too imbalanced"


def test_scaffold_column_supports_a_scaffold_split():
    frame = pd.read_csv(REFERENCE_REGRESSION_PATH)

    scaffolds = frame["scaffold"]
    assert scaffolds.nunique() >= 50, "too few distinct scaffolds for a scaffold split"
    # No single scaffold may dominate, or a scaffold split degenerates into one group.
    assert scaffolds.value_counts().iloc[0] <= 0.25 * len(frame)
    assert scaffolds.map(lambda s: Chem.MolFromSmiles(s) is not None).all()


def test_regression_target_is_exactly_the_crippen_descriptor():
    """clogp must BE MolLogP - not a plausible-looking number near it."""
    frame = pd.read_csv(REFERENCE_REGRESSION_PATH)

    expected = [round(Crippen.MolLogP(Chem.MolFromSmiles(s)), 4) for s in frame["smiles"]]
    assert frame["clogp"].tolist() == pytest.approx(expected, abs=1e-9)
    assert frame["clogp"].std() > 0.5, "target has almost no variance to model"


def test_classification_target_is_exactly_the_tpsa_threshold():
    frame = pd.read_csv(REFERENCE_CLASSIFICATION_PATH)

    expected = [
        int(Descriptors.TPSA(Chem.MolFromSmiles(s)) > TPSA_HIGH_THRESHOLD) for s in frame["smiles"]
    ]
    assert frame["tpsa_high"].tolist() == expected


def test_descriptors_match_their_closed_form_values():
    """Anchor the two descriptors on published, hand-computable values.

    Wildman-Crippen logP is an additive atom-contribution sum, so methane is exactly
    one C1 carbon plus four H1 hydrogens, and ethane two carbons plus six hydrogens.
    Ertl TPSA is likewise additive: benzene contributes nothing, and the single
    aromatic nitrogen of pyridine contributes 12.89 A^2.

    This fails if the generator is ever pointed at a different descriptor.
    """
    assert Crippen.MolLogP(Chem.MolFromSmiles("C")) == pytest.approx(0.1441 + 4 * 0.1230, abs=1e-6)
    assert Crippen.MolLogP(Chem.MolFromSmiles("CC")) == pytest.approx(
        2 * 0.1441 + 6 * 0.1230, abs=1e-6
    )
    assert Descriptors.TPSA(Chem.MolFromSmiles("c1ccccc1")) == pytest.approx(0.0, abs=1e-6)
    assert Descriptors.TPSA(Chem.MolFromSmiles("c1ccncc1")) == pytest.approx(12.89, abs=1e-6)

    # And the generator agrees with those anchors on a molecule it actually emits.
    frame = build_reference_dataset(task="regression", n=20)
    probe = frame.iloc[0]
    assert probe["clogp"] == pytest.approx(
        Crippen.MolLogP(Chem.MolFromSmiles(probe["smiles"])), abs=5e-5
    )


@pytest.mark.parametrize("bad_n", [0, 9, 10**6])
def test_rejects_unsupported_row_counts(bad_n):
    with pytest.raises(ValueError, match="n must be between"):
        build_reference_dataset(task="regression", n=bad_n)


def test_rejects_unknown_task():
    with pytest.raises(ValueError, match="task must be"):
        build_reference_dataset(task="ranking", n=50)  # type: ignore[arg-type]
