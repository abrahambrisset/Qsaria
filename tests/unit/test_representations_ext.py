#!/usr/bin/env python
# coding: utf-8
"""Tests for the widened molecular representation space.

Covers the unlocked Morgan parameters, the four new structural fingerprints, and the
feature-cache key that must change whenever those parameters change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from cs_copilot.tools.features.molecular_feature_toolkit import (
    DEFAULT_MORGAN_N_BITS,
    DEFAULT_MORGAN_RADIUS,
    MACCS_N_BITS,
    STRUCTURAL_FINGERPRINTS,
    SUPPORTED_FINGERPRINT_N_BITS,
    MolecularFeatureToolkit,
    resolve_structural_fingerprint,
)
from cs_copilot.tools.prediction.qsar_contracts import GeneratedRepresentation
from cs_copilot.tools.prediction.qsar_training_toolkit import QSARTrainingToolkit
from cs_copilot.tools.prediction.tabular_representations import (
    AUTOMATIC_TABULAR_REPRESENTATION_NAMES,
    SUPPORTED_TABULAR_REPRESENTATION_NAMES,
    get_tabular_representation,
    representation_unavailable_reason,
)

# Small, real, valid drug-like structures; nothing here is an experimental measurement.
SMILES: List[str] = [
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccccc1",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "OCC1OC(O)C(O)C(O)C1O",
]

NEW_REPRESENTATIONS: Dict[str, tuple[str, int]] = {
    "maccs": ("mk_", MACCS_N_BITS),
    "avalon": ("av_", 1024),
    "atom_pair": ("ap_", 2048),
    "topological_torsion": ("tt_", 2048),
}


@pytest.fixture()
def smiles_csv(tmp_path: Path) -> str:
    path = tmp_path / "molecules.csv"
    pd.DataFrame({"smiles": SMILES, "Y": [float(i) for i in range(len(SMILES))]}).to_csv(
        path, index=False
    )
    return str(path)


def _feature_columns(path: str, prefix: str) -> List[str]:
    return [c for c in pd.read_csv(path, nrows=0).columns if str(c).startswith(prefix)]


# --------------------------------------------------------------------------------------
# Morgan radius / n_bits / use_features
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_morgan_accepts_every_supported_radius(
    tmp_path: Path, smiles_csv: str, radius: int
) -> None:
    out = str(tmp_path / f"r{radius}.csv")
    result = MolecularFeatureToolkit().smiles_to_morgan_fingerprints(
        input_csv=smiles_csv, output_csv=out, radius=radius, n_bits=1024
    )
    assert result["radius"] == radius
    assert result["n_bits"] == 1024
    assert result["num_features"] == 1024
    assert len(_feature_columns(out, "fp_")) == 1024


def test_morgan_radius_changes_the_matrix(tmp_path: Path, smiles_csv: str) -> None:
    """A larger radius sees larger substructures, so caffeine must light up more bits."""
    toolkit = MolecularFeatureToolkit()
    frames = {}
    for radius in (1, 2, 3):
        out = str(tmp_path / f"radius_{radius}.csv")
        toolkit.smiles_to_morgan_fingerprints(
            input_csv=smiles_csv, output_csv=out, radius=radius, n_bits=4096
        )
        frames[radius] = pd.read_csv(out)[_feature_columns(out, "fp_")]

    assert not frames[1].equals(frames[2])
    assert not frames[2].equals(frames[3])
    # radius r+1 keeps every environment of radius r and adds strictly larger ones; with
    # 4096 bits over five small molecules, collisions do not mask that growth.
    assert frames[1].to_numpy().sum() < frames[2].to_numpy().sum()
    assert frames[2].to_numpy().sum() < frames[3].to_numpy().sum()


def test_morgan_n_bits_controls_width(tmp_path: Path, smiles_csv: str) -> None:
    for n_bits in (1024, 2048, 4096):
        out = str(tmp_path / f"bits_{n_bits}.csv")
        result = MolecularFeatureToolkit().smiles_to_morgan_fingerprints(
            input_csv=smiles_csv, output_csv=out, n_bits=n_bits
        )
        assert result["num_features"] == n_bits
        assert len(_feature_columns(out, "fp_")) == n_bits


def test_morgan_use_features_gives_a_different_fingerprint(tmp_path: Path, smiles_csv: str) -> None:
    """FCFP uses pharmacophore-style atom invariants, so the bits must differ from ECFP."""
    toolkit = MolecularFeatureToolkit()
    ecfp_path = str(tmp_path / "ecfp.csv")
    fcfp_path = str(tmp_path / "fcfp.csv")
    toolkit.smiles_to_morgan_fingerprints(input_csv=smiles_csv, output_csv=ecfp_path)
    result = toolkit.smiles_to_morgan_fingerprints(
        input_csv=smiles_csv, output_csv=fcfp_path, use_features=True
    )

    assert result["use_features"] is True
    columns = _feature_columns(ecfp_path, "fp_")
    assert not pd.read_csv(ecfp_path)[columns].equals(pd.read_csv(fcfp_path)[columns])


@pytest.mark.parametrize("radius", [0, 4, -1])
def test_morgan_rejects_unsupported_radius(smiles_csv: str, radius: int) -> None:
    with pytest.raises(ValueError, match="Unsupported Morgan radius"):
        MolecularFeatureToolkit().smiles_to_morgan_fingerprints(input_csv=smiles_csv, radius=radius)


@pytest.mark.parametrize("n_bits", [0, -8])
def test_morgan_rejects_non_positive_n_bits(smiles_csv: str, n_bits: int) -> None:
    with pytest.raises(ValueError, match="n_bits must be a positive integer"):
        MolecularFeatureToolkit().smiles_to_morgan_fingerprints(input_csv=smiles_csv, n_bits=n_bits)


def test_contract_restricts_widths_to_the_curated_set(tmp_path: Path, smiles_csv: str) -> None:
    """The helper stays permissive for ad-hoc exports; the agent contract does not."""
    result = MolecularFeatureToolkit().smiles_to_morgan_fingerprints(
        input_csv=smiles_csv, output_csv=str(tmp_path / "tiny.csv"), n_bits=64
    )
    assert result["num_features"] == 64
    assert SUPPORTED_FINGERPRINT_N_BITS == (1024, 2048, 4096)
    with pytest.raises(ValueError):
        GeneratedRepresentation(morgan_n_bits=64)


def test_morgan_defaults_are_unchanged(tmp_path: Path, smiles_csv: str) -> None:
    """The historical default must stay radius=2 / 2048 bits / ECFP invariants."""
    assert (DEFAULT_MORGAN_RADIUS, DEFAULT_MORGAN_N_BITS) == (2, 2048)
    result = MolecularFeatureToolkit().smiles_to_morgan_fingerprints(
        input_csv=smiles_csv, output_csv=str(tmp_path / "default.csv")
    )
    assert (result["radius"], result["n_bits"], result["use_features"]) == (2, 2048, False)


# --------------------------------------------------------------------------------------
# New structural fingerprints
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "expected"), sorted(NEW_REPRESENTATIONS.items()))
def test_structural_fingerprint_width_and_prefix(
    tmp_path: Path, smiles_csv: str, name: str, expected: tuple[str, int]
) -> None:
    prefix, width = expected
    if representation_unavailable_reason(name):
        pytest.skip(f"{name} is not available in this RDKit build")

    out = str(tmp_path / f"{name}.csv")
    result = MolecularFeatureToolkit().smiles_to_structural_fingerprints(
        input_csv=smiles_csv, fingerprint_name=name, output_csv=out
    )

    assert result["feature_prefix"] == prefix
    assert result["n_bits"] == width
    assert result["num_features"] == width
    frame = pd.read_csv(out)
    columns = _feature_columns(out, prefix)
    assert len(columns) == width
    assert len(frame) == len(SMILES)
    # Binary fingerprints: every value is 0/1 and the molecules are not all identical.
    values = frame[columns].to_numpy()
    assert set(values.ravel().tolist()) <= {0, 1}
    assert values.sum() > 0
    assert len(set(map(tuple, values.tolist()))) == len(SMILES)


def test_maccs_matches_rdkit_directly(tmp_path: Path, smiles_csv: str) -> None:
    """Known-answer check: the stored row must be RDKit's own MACCS key vector."""
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys

    out = str(tmp_path / "maccs.csv")
    MolecularFeatureToolkit().smiles_to_structural_fingerprints(
        input_csv=smiles_csv, fingerprint_name="maccs", output_csv=out
    )
    frame = pd.read_csv(out)
    columns = _feature_columns(out, "mk_")

    for row_index, smiles in enumerate(SMILES):
        expected = [int(bit) for bit in MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))]
        assert frame.loc[row_index, columns].tolist() == expected
    # MACCS bit 0 is a documented unused placeholder.
    assert frame["mk_0000"].tolist() == [0] * len(SMILES)


@pytest.mark.parametrize(
    ("name", "generator_factory"),
    [
        ("atom_pair", "GetAtomPairGenerator"),
        ("topological_torsion", "GetTopologicalTorsionGenerator"),
        ("avalon", "avalon"),
    ],
)
def test_hashed_fingerprint_matches_its_own_rdkit_generator(
    tmp_path: Path, smiles_csv: str, name: str, generator_factory: str
) -> None:
    """Each fingerprint must come from ITS generator, not merely be the right width."""
    if representation_unavailable_reason(name):
        pytest.skip(f"{name} is not available in this RDKit build")
    from rdkit import Chem, rdBase  # noqa: F401
    from rdkit.Chem import rdFingerprintGenerator

    prefix, width = NEW_REPRESENTATIONS[name]
    out = str(tmp_path / f"{name}_known.csv")
    MolecularFeatureToolkit().smiles_to_structural_fingerprints(
        input_csv=smiles_csv, fingerprint_name=name, output_csv=out
    )
    frame = pd.read_csv(out)
    columns = _feature_columns(out, prefix)

    for row_index, smiles in enumerate(SMILES):
        mol = Chem.MolFromSmiles(smiles)
        if name == "avalon":
            from rdkit.Avalon import pyAvalonTools

            expected = [int(bit) for bit in pyAvalonTools.GetAvalonFP(mol, nBits=width)]
        else:
            generator = getattr(rdFingerprintGenerator, generator_factory)(fpSize=width)
            expected = generator.GetFingerprintAsNumPy(mol).astype(int).tolist()
        assert frame.loc[row_index, columns].tolist() == expected


def test_atom_pair_and_torsion_are_different_fingerprints(tmp_path: Path, smiles_csv: str) -> None:
    """Guards against wiring both hashed fingerprints to the same generator."""
    toolkit = MolecularFeatureToolkit()
    frames = {}
    for name in ("atom_pair", "topological_torsion"):
        out = str(tmp_path / f"{name}_cmp.csv")
        toolkit.smiles_to_structural_fingerprints(
            input_csv=smiles_csv, fingerprint_name=name, output_csv=out
        )
        prefix = NEW_REPRESENTATIONS[name][0]
        frames[name] = pd.read_csv(out)[_feature_columns(out, prefix)].to_numpy()

    assert (frames["atom_pair"] != frames["topological_torsion"]).any()
    # Benzene has no four-atom torsion path that survives the torsion definition, while it
    # has many atom pairs; the two fingerprints cannot be the same function.
    assert frames["atom_pair"].sum() > frames["topological_torsion"].sum()


def test_maccs_cannot_be_resized() -> None:
    with pytest.raises(ValueError, match="fixed width"):
        resolve_structural_fingerprint("maccs", 2048)


def test_hashed_fingerprints_accept_supported_sizes(tmp_path: Path, smiles_csv: str) -> None:
    out = str(tmp_path / "ap_4096.csv")
    result = MolecularFeatureToolkit().smiles_to_structural_fingerprints(
        input_csv=smiles_csv, fingerprint_name="atom_pair", output_csv=out, n_bits=4096
    )
    assert result["n_bits"] == 4096
    with pytest.raises(ValueError, match="n_bits must be a positive integer"):
        resolve_structural_fingerprint("atom_pair", 0)


def test_unknown_fingerprint_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported fingerprint_name"):
        resolve_structural_fingerprint("not_a_fingerprint")


def test_new_representations_are_registered_but_not_automatic() -> None:
    for name, (prefix, _) in NEW_REPRESENTATIONS.items():
        assert name in SUPPORTED_TABULAR_REPRESENTATION_NAMES
        spec = get_tabular_representation(name)
        assert spec.fingerprints == (name,)
        assert STRUCTURAL_FINGERPRINTS[name]["feature_prefix"] == prefix
        assert spec.automatic is False
    # The automatic comparative set must not have changed.
    assert AUTOMATIC_TABULAR_REPRESENTATION_NAMES == (
        "rdkit_all",
        "morgan_only",
        "morgan_count_only",
    )


def test_existing_representations_keep_their_definitions() -> None:
    for name in ("rdkit_all", "morgan_only", "morgan_count_only", "morgan_rdkit_all"):
        spec = get_tabular_representation(name)
        assert spec.fingerprints == ()
        assert (spec.morgan_radius, spec.morgan_n_bits, spec.morgan_use_features) == (
            2,
            2048,
            False,
        )
    assert get_tabular_representation("morgan_only").use_morgan_binary is True
    assert get_tabular_representation("morgan_count_only").use_morgan_count is True
    assert get_tabular_representation("rdkit_all").descriptor_set == "all"


def test_avalon_availability_is_reported_not_crashed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An RDKit build without Avalon must yield a message, not an import error."""
    import importlib

    import cs_copilot.tools.prediction.tabular_representations as registry

    registry.representation_unavailable_reason.cache_clear()
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if "Avalon" in name else object()
    )
    try:
        reason = registry.representation_unavailable_reason("avalon")
        assert reason is not None
        assert "rdkit.Avalon.pyAvalonTools" in reason
        assert registry.representation_unavailable_reason("maccs") is None
    finally:
        registry.representation_unavailable_reason.cache_clear()


# --------------------------------------------------------------------------------------
# Training wiring and the feature cache key
# --------------------------------------------------------------------------------------


@pytest.fixture()
def training_csv(tmp_path: Path) -> str:
    path = tmp_path / "train.csv"
    pd.DataFrame({"smiles": SMILES, "Y": [float(i) for i in range(len(SMILES))]}).to_csv(
        path, index=False
    )
    return str(path)


def _prepare(tmp_path: Path, training_csv: str, representation_name: str, **kwargs: Any) -> Dict:
    return QSARTrainingToolkit()._prepare_tabular_training_dataset(
        train_csv=training_csv,
        output_dir=str(tmp_path / f"out_{representation_name}"),
        smiles_column="smiles",
        target_columns=["Y"],
        representation_name=representation_name,
        feature_cache_dir=str(tmp_path / "cache"),
        **kwargs,
    )


@pytest.mark.parametrize("name", sorted(NEW_REPRESENTATIONS))
def test_training_builds_each_new_representation(
    tmp_path: Path, training_csv: str, name: str
) -> None:
    if representation_unavailable_reason(name):
        pytest.skip(f"{name} is not available in this RDKit build")
    prefix, width = NEW_REPRESENTATIONS[name]

    prepared = _prepare(tmp_path, training_csv, name)

    columns = prepared["feature_columns"]
    assert len(columns) == width
    assert all(str(column).startswith(prefix) for column in columns)
    assert len(pd.read_csv(prepared["train_csv"])) == len(SMILES)
    steps = {step["step"] for step in prepared["feature_preparation_durations"]["steps"]}
    assert f"{name}_fingerprints" in steps


def test_cache_key_changes_with_morgan_radius(tmp_path: Path, training_csv: str) -> None:
    """A cache hit across different radii would silently return the wrong features."""
    radius_2 = _prepare(tmp_path, training_csv, "morgan_only", morgan_radius=2)
    radius_3 = _prepare(tmp_path, training_csv, "morgan_only", morgan_radius=3)

    key_2 = radius_2["feature_preparation"]["feature_cache_key"]
    key_3 = radius_3["feature_preparation"]["feature_cache_key"]
    assert key_2 != key_3
    assert radius_2["train_csv"] != radius_3["train_csv"]

    columns = radius_2["feature_columns"]
    assert not pd.read_csv(radius_2["train_csv"])[columns].equals(
        pd.read_csv(radius_3["train_csv"])[columns]
    )


def test_cache_key_changes_with_n_bits_and_use_features(tmp_path: Path, training_csv: str) -> None:
    baseline = _prepare(tmp_path, training_csv, "morgan_only")
    wider = _prepare(tmp_path, training_csv, "morgan_only", morgan_n_bits=4096)
    featured = _prepare(tmp_path, training_csv, "morgan_only", morgan_use_features=True)

    keys = {
        baseline["feature_preparation"]["feature_cache_key"],
        wider["feature_preparation"]["feature_cache_key"],
        featured["feature_preparation"]["feature_cache_key"],
    }
    assert len(keys) == 3
    assert len(baseline["feature_columns"]) == 2048
    assert len(wider["feature_columns"]) == 4096


def test_cache_is_reused_for_identical_parameters(tmp_path: Path, training_csv: str) -> None:
    first = _prepare(tmp_path, training_csv, "morgan_only")
    second = _prepare(tmp_path, training_csv, "morgan_only")

    assert (
        first["feature_preparation"]["feature_cache_key"]
        == second["feature_preparation"]["feature_cache_key"]
    )
    assert second["feature_preparation"]["feature_cache_status"] == "reused_from_cache"


def test_default_parameters_reproduce_the_registry_spec(tmp_path: Path, training_csv: str) -> None:
    """Passing nothing must equal passing the historical defaults, bit for bit."""
    implicit = _prepare(tmp_path, training_csv, "morgan_only")
    explicit = _prepare(
        tmp_path,
        training_csv,
        "morgan_only",
        morgan_radius=2,
        morgan_n_bits=2048,
        morgan_use_features=False,
    )
    assert (
        implicit["feature_preparation"]["feature_cache_key"]
        == explicit["feature_preparation"]["feature_cache_key"]
    )
    assert implicit["feature_preparation"]["inference_rebuild_warning"] is None


def test_non_default_morgan_parameters_warn_about_inference(
    tmp_path: Path, training_csv: str
) -> None:
    prepared = _prepare(tmp_path, training_csv, "morgan_only", morgan_radius=3)
    warning = prepared["feature_preparation"]["inference_rebuild_warning"]
    assert warning is not None
    assert "radius=3" in warning


def test_structural_representations_warn_about_inference(tmp_path: Path, training_csv: str) -> None:
    prepared = _prepare(tmp_path, training_csv, "maccs")
    warning = prepared["feature_preparation"]["inference_rebuild_warning"]
    assert warning is not None
    assert "maccs" in warning


def test_generated_representation_contract_accepts_new_names_and_knobs() -> None:
    representation = GeneratedRepresentation(
        name="atom_pair", morgan_radius=3, morgan_n_bits=4096, morgan_use_features=True
    )
    assert representation.name == "atom_pair"
    assert (representation.morgan_radius, representation.morgan_n_bits) == (3, 4096)

    default = GeneratedRepresentation()
    assert (default.name, default.morgan_radius, default.morgan_n_bits) == ("rdkit_all", 2, 2048)
    assert default.morgan_use_features is False

    with pytest.raises(ValueError):
        GeneratedRepresentation(morgan_radius=4)
    with pytest.raises(ValueError):
        GeneratedRepresentation(morgan_n_bits=512)
    with pytest.raises(ValueError):
        GeneratedRepresentation(name="not_a_representation")
