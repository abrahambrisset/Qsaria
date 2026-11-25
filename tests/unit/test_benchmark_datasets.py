"""Offline tests for the public-benchmark dataset layer.

No test here touches the network: every load works from a locally seeded cache.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from cs_copilot.tools.benchmarks import datasets as bd

# sha256 of b"abc" and of the empty byte string (FIPS 180-4 worked examples).
SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
SHA256_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@pytest.fixture
def clean_registry():
    """Restore the module-level registry after a test registers into it."""
    snapshot = dict(bd._REGISTRY)
    try:
        yield
    finally:
        bd._REGISTRY.clear()
        bd._REGISTRY.update(snapshot)


def _seed_cache(cache_dir: Path, name: str, filename: str, frame: pd.DataFrame) -> Path:
    """Write ``frame`` where the loader for ``name`` expects to find its download."""
    target = cache_dir / bd._slug(name) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def _toy_frame() -> pd.DataFrame:
    # Synthetic fixture rows, not measured data.
    return pd.DataFrame({"smiles": ["CCO", "CCC", "c1ccccc1"], "y": [1.0, 2.0, 3.0]})


def test_dataset_checksum_matches_known_sha256(tmp_path):
    abc = tmp_path / "abc.bin"
    abc.write_bytes(b"abc")
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")

    assert bd.dataset_checksum(abc) == SHA256_ABC
    assert bd.dataset_checksum(empty) == SHA256_EMPTY


def test_default_cache_dir_follows_env(monkeypatch, tmp_path):
    monkeypatch.delenv(bd.CACHE_DIR_ENV, raising=False)
    assert bd.default_cache_dir() == bd.DEFAULT_CACHE_DIR

    monkeypatch.setenv(bd.CACHE_DIR_ENV, str(tmp_path / "elsewhere"))
    assert bd.default_cache_dir() == tmp_path / "elsewhere"


def test_register_and_list_round_trip(clean_registry, tmp_path):
    bd.register_dataset(
        "unit_toy",
        bd.csv_loader(filename="toy.csv", smiles_column="smiles", target_columns=["y"]),
        task_type="regression",
        source=bd.LOCAL_SOURCE,
    )

    rows = {row["name"]: row for row in bd.list_datasets(cache_dir=tmp_path)}
    assert rows["unit_toy"] == {
        "name": "unit_toy",
        "task_type": "regression",
        "source": bd.LOCAL_SOURCE,
        "requires_network": False,
        "cached": False,
    }
    assert rows["delaney"]["requires_network"] is True
    assert rows["delaney"]["cached"] is False

    _seed_cache(tmp_path, "unit_toy", "toy.csv", _toy_frame())
    rows = {row["name"]: row for row in bd.list_datasets(cache_dir=tmp_path)}
    assert rows["unit_toy"]["cached"] is True


def test_register_dataset_rejects_bad_task_type(clean_registry):
    with pytest.raises(ValueError, match="unsupported task_type"):
        bd.register_dataset(
            "unit_bad",
            bd.csv_loader(filename="x.csv", smiles_column="smiles", target_columns=["y"]),
            task_type="regresion",
            source=bd.LOCAL_SOURCE,
        )


def test_load_dataset_from_seeded_cache(clean_registry, tmp_path):
    bd.register_dataset(
        "unit_toy",
        bd.csv_loader(
            filename="toy.csv",
            smiles_column="smiles",
            target_columns=["y"],
            url="https://example.invalid/toy.csv",
            licence="CC0",
            version="v1",
            notes="fixture",
        ),
        task_type="regression",
        source="unit fixture",
        default_url="https://example.invalid/toy.csv",
    )
    path = _seed_cache(tmp_path, "unit_toy", "toy.csv", _toy_frame())

    dataset = bd.load_dataset("unit_toy", cache_dir=tmp_path, allow_download=False)

    assert dataset.name == "unit_toy"
    assert dataset.task_type == "regression"
    assert dataset.smiles_column == "smiles"
    assert dataset.target_columns == ["y"]
    assert dataset.canonical_split is None
    assert list(dataset.frame["smiles"]) == ["CCO", "CCC", "c1ccccc1"]
    assert dataset.provenance.source == "unit fixture"
    assert dataset.provenance.url == "https://example.invalid/toy.csv"
    assert dataset.provenance.licence == "CC0"
    assert dataset.provenance.version == "v1"
    assert dataset.provenance.n_rows == 3
    assert dataset.provenance.sha256 == bd.dataset_checksum(path)
    assert dataset.provenance.retrieved_at.endswith("+00:00")
    # The first load records the digest so later loads are verified.
    assert bd._record_path(path).read_text(encoding="utf-8") == dataset.provenance.sha256


def test_corrupted_cache_file_raises_checksum_mismatch(clean_registry, tmp_path):
    bd.register_dataset(
        "unit_toy",
        bd.csv_loader(filename="toy.csv", smiles_column="smiles", target_columns=["y"]),
        task_type="regression",
        source=bd.LOCAL_SOURCE,
    )
    path = _seed_cache(tmp_path, "unit_toy", "toy.csv", _toy_frame())
    bd.load_dataset("unit_toy", cache_dir=tmp_path, allow_download=False)

    path.write_text("smiles,y\nCCO,99999.0\n", encoding="utf-8")

    with pytest.raises(bd.ChecksumMismatchError, match="corrupt or was replaced"):
        bd.load_dataset("unit_toy", cache_dir=tmp_path, allow_download=False)


def test_pinned_checksum_mismatch_raises(clean_registry, tmp_path):
    bd.register_dataset(
        "unit_pinned",
        bd.csv_loader(
            filename="toy.csv",
            smiles_column="smiles",
            target_columns=["y"],
            expected_sha256=SHA256_ABC,
        ),
        task_type="regression",
        source=bd.LOCAL_SOURCE,
    )
    _seed_cache(tmp_path, "unit_pinned", "toy.csv", _toy_frame())

    with pytest.raises(bd.ChecksumMismatchError, match=SHA256_ABC):
        bd.load_dataset("unit_pinned", cache_dir=tmp_path, allow_download=False)


def test_allow_download_false_on_empty_cache_raises(tmp_path):
    with pytest.raises(bd.DatasetNotCachedError, match="allow_download=False"):
        bd.load_dataset("delaney", cache_dir=tmp_path, allow_download=False)


def test_missing_url_raises_instead_of_guessing(clean_registry, tmp_path):
    bd.register_dataset(
        "unit_nourl",
        bd.csv_loader(filename="toy.csv", smiles_column="smiles", target_columns=["y"]),
        task_type="regression",
        source="unverified source",
        default_url=None,
    )

    with pytest.raises(bd.DatasetDownloadError, match="no verified download URL"):
        bd.load_dataset("unit_nourl", cache_dir=tmp_path, allow_download=True)


def test_unknown_dataset_lists_available_names():
    with pytest.raises(KeyError) as excinfo:
        bd.load_dataset("not_a_dataset")

    message = str(excinfo.value)
    assert "not_a_dataset" in message
    for name in ("delaney", "freesolv", "tox21", "hiv"):
        assert name in message


def test_loader_raises_when_upstream_columns_change(clean_registry, tmp_path):
    bd.register_dataset(
        "unit_shifted",
        bd.csv_loader(filename="toy.csv", smiles_column="smiles", target_columns=["expected"]),
        task_type="regression",
        source=bd.LOCAL_SOURCE,
    )
    _seed_cache(tmp_path, "unit_shifted", "toy.csv", _toy_frame())

    with pytest.raises(bd.BenchmarkDatasetError, match="are absent from"):
        bd.load_dataset("unit_shifted", cache_dir=tmp_path, allow_download=False)


@pytest.mark.parametrize(
    ("split", "pattern"),
    [
        ({"train": [0, 1], "test": [1, 2]}, "would leak"),
        ({"train": [0], "test": [99]}, "outside the 3-row frame"),
    ],
)
def test_invalid_canonical_split_is_rejected(clean_registry, tmp_path, split, pattern):
    def _loader(context: bd.LoaderContext) -> bd.BenchmarkDataset:
        return bd.BenchmarkDataset(
            name=context.name,
            task_type=context.task_type,
            smiles_column="smiles",
            target_columns=["y"],
            frame=_toy_frame(),
            canonical_split=split,
            provenance=bd.Provenance(source=context.source),
        )

    bd.register_dataset("unit_split", _loader, task_type="regression", source=bd.LOCAL_SOURCE)

    with pytest.raises(bd.BenchmarkDatasetError, match=pattern):
        bd.load_dataset("unit_split", cache_dir=tmp_path, allow_download=False)


def test_moleculenet_registrations_use_verified_urls():
    rows = {row["name"]: row for row in bd.list_datasets(cache_dir=Path("/nonexistent"))}
    expected = {
        "delaney": "delaney-processed.csv",
        "freesolv": "SAMPL.csv",
        "lipophilicity": "Lipophilicity.csv",
        "bace_classification": "bace.csv",
        "bbbp": "BBBP.csv",
        "clintox": "clintox.csv.gz",
        "tox21": "tox21.csv.gz",
        "sider": "sider.csv.gz",
        "hiv": "HIV.csv",
    }
    for name, filename in expected.items():
        assert name in rows, f"{name} is not registered"
        assert rows[name]["source"] == bd.MOLECULENET_SOURCE
        assert bd._REGISTRY[name].default_url == f"{bd.MOLECULENET_BASE_URL}/{filename}"
    assert rows["delaney"]["task_type"] == "regression"
    assert rows["tox21"]["task_type"] == "classification"


def _install_tdc_stub(monkeypatch, splits):
    class _StubGroup:
        def __init__(self, name, path):
            self.name = name
            self.path = path

        def get_split(self):
            return splits

    tdc_module = types.ModuleType("tdc")
    tdc_module.__version__ = "0.0-stub"
    single_pred = types.ModuleType("tdc.single_pred")
    single_pred.ADME = _StubGroup
    tdc_module.single_pred = single_pred
    monkeypatch.setitem(sys.modules, "tdc", tdc_module)
    monkeypatch.setitem(sys.modules, "tdc.single_pred", single_pred)


def test_tdc_loader_passes_published_split_through(clean_registry, monkeypatch, tmp_path):
    splits = {
        "train": pd.DataFrame({"Drug": ["CCO", "CCC", "CCN"], "Y": [1.0, 2.0, 3.0]}),
        "valid": pd.DataFrame({"Drug": ["CCCl"], "Y": [4.0]}),
        "test": pd.DataFrame({"Drug": ["CCBr", "CCI"], "Y": [5.0, 6.0]}),
    }
    _install_tdc_stub(monkeypatch, splits)
    bd.register_dataset(
        "unit_tdc",
        bd.tdc_loader("Fixture_Dataset", group="ADME"),
        task_type="regression",
        source=bd.TDC_SOURCE,
    )

    dataset = bd.load_dataset("unit_tdc", cache_dir=tmp_path, allow_download=True)

    assert dataset.canonical_split == {"train": [0, 1, 2], "valid": [3], "test": [4, 5]}
    assert list(dataset.frame["Drug"]) == ["CCO", "CCC", "CCN", "CCCl", "CCBr", "CCI"]
    assert dataset.target_columns == ["Y"]
    assert dataset.provenance.version == "0.0-stub"
    assert dataset.provenance.n_rows == 6
    # Rows sit exactly where the published split says they do.
    for split_name, indices in dataset.canonical_split.items():
        assert list(dataset.frame.loc[indices, "Y"]) == list(splits[split_name]["Y"])


def test_tdc_loader_raises_install_hint_without_pytdc(clean_registry, tmp_path):
    if importlib.util.find_spec("tdc") is not None:  # pragma: no cover - env dependent
        pytest.skip("PyTDC is installed; the install-hint path cannot be exercised")

    bd.register_dataset(
        "unit_tdc_missing",
        bd.tdc_loader("Fixture_Dataset", group="ADME"),
        task_type="regression",
        source=bd.TDC_SOURCE,
    )

    with pytest.raises(bd.BenchmarkDatasetError, match="pip install PyTDC"):
        bd.load_dataset("unit_tdc_missing", cache_dir=tmp_path, allow_download=True)


@pytest.mark.live
def test_delaney_downloads_published_esol_set(tmp_path):
    """Deselected by default; opt in with `-m live`. Needs network."""
    dataset = bd.load_dataset("delaney", cache_dir=tmp_path, allow_download=True)

    # ESOL as published by Delaney and redistributed by MoleculeNet: 1128 compounds.
    assert len(dataset.frame) == 1128
    assert dataset.provenance.n_rows == 1128
    assert dataset.frame[dataset.smiles_column].notna().all()
    assert dataset.frame[dataset.target_columns[0]].dtype.kind == "f"
