#!/usr/bin/env python
# coding: utf-8
"""Public-benchmark dataset registry, cache and loaders.

Every loader either returns real, downloaded data or raises.  Nothing in this
module fabricates dataset contents, checksums, row counts or split indices.

A dataset is fetched once into a cache directory (``QSARIA_BENCHMARK_CACHE``,
else ``.files/benchmark_cache``).  The first download records the file's sha256
in a ``<file>.sha256`` sidecar; every later load re-hashes the cached file and
raises :class:`ChecksumMismatchError` when it no longer matches.  A registration
may also pin an expected digest up front via ``expected_sha256``.

Where the source publishes a split (TDC), the published split is passed through
as row indices rather than regenerated -- a regenerated split makes benchmark
numbers incomparable with the literature.  MoleculeNet publishes a recommended
*splitter* (scaffold) and not fixed indices, so those loaders return
``canonical_split=None`` and the caller must apply the platform's splitter.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import pandas as pd

CACHE_DIR_ENV = "QSARIA_BENCHMARK_CACHE"
DEFAULT_CACHE_DIR = Path(".files") / "benchmark_cache"
MOLECULENET_BASE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"
MOLECULENET_SOURCE = "MoleculeNet (DeepChem S3 mirror)"
TDC_SOURCE = "TDC (Therapeutics Data Commons)"
LOCAL_SOURCE = "local"
SUPPORTED_TASK_TYPES = frozenset({"regression", "classification"})
DOWNLOAD_TIMEOUT_S = 120.0
_USER_AGENT = "qsaria-benchmark-loader/1.0 (+stdlib urllib)"
_MOLECULENET_SPLIT_NOTE = (
    "MoleculeNet publishes a recommended splitter (scaffold), not fixed row indices, "
    "so canonical_split is None; apply the platform splitter to reproduce published numbers."
)
_UNSTATED_LICENCE = "not stated at the download URL; see the MoleculeNet publication"


class BenchmarkDatasetError(RuntimeError):
    """Base class for every failure raised by the benchmark dataset layer."""


class DatasetNotCachedError(BenchmarkDatasetError):
    """The dataset is absent from the cache and downloading was not allowed."""


class ChecksumMismatchError(BenchmarkDatasetError):
    """A cached file does not match its recorded sha256 expectation."""


class DatasetDownloadError(BenchmarkDatasetError):
    """The dataset could not be downloaded (offline, HTTP error, no URL)."""


@dataclass(frozen=True)
class Provenance:
    """Where a benchmark frame came from, recorded at load time.

    Fields are ``None`` when the source does not publish the information; they
    are never filled in with a plausible guess.
    """

    source: str
    url: Optional[str] = None
    version: Optional[str] = None
    licence: Optional[str] = None
    sha256: Optional[str] = None
    retrieved_at: Optional[str] = None
    n_rows: Optional[int] = None
    notes: str = ""


@dataclass
class BenchmarkDataset:
    """A loaded public benchmark: the frame plus everything needed to cite it."""

    name: str
    task_type: str
    smiles_column: str
    target_columns: List[str]
    frame: pd.DataFrame
    canonical_split: Optional[Dict[str, List[int]]] = None
    provenance: Provenance = field(default_factory=lambda: Provenance(source="unspecified"))


@dataclass(frozen=True)
class LoaderContext:
    """What a loader function is handed: its registration plus cache policy."""

    name: str
    task_type: str
    source: str
    default_url: Optional[str]
    cache_dir: Path
    allow_download: bool

    def dataset_dir(self) -> Path:
        """Return (without creating) the cache sub-directory for this dataset."""
        return self.cache_dir / _slug(self.name)

    def fetch(
        self,
        *,
        filename: str,
        url: Optional[str] = None,
        expected_sha256: Optional[str] = None,
    ) -> Path:
        """Return a local path to ``filename``, downloading it once if needed.

        Raises
        ------
        DatasetNotCachedError
            The file is missing and ``allow_download`` is False.
        DatasetDownloadError
            No URL is registered, or the download failed (for example offline).
        ChecksumMismatchError
            The cached or downloaded bytes do not match a recorded expectation.
        """
        destination = self.dataset_dir() / filename
        if destination.is_file():
            return _verify_cached(destination, expected_sha256)

        if not self.allow_download:
            raise DatasetNotCachedError(
                f"benchmark dataset '{self.name}' is not cached at {destination} and downloads "
                f"are disabled (allow_download=False). Either call "
                f"load_dataset('{self.name}', allow_download=True) on a machine with network "
                f"access, or copy the file to that path manually."
            )

        source_url = url or self.default_url
        if not source_url:
            raise DatasetDownloadError(
                f"benchmark dataset '{self.name}' has no verified download URL registered, so it "
                f"cannot be fetched automatically. Obtain '{filename}' from the original "
                f"publication and place it at {destination}."
            )

        _download(source_url, destination, expected_sha256)
        return destination


LoaderFn = Callable[[LoaderContext], BenchmarkDataset]


@dataclass(frozen=True)
class _Entry:
    name: str
    loader_fn: LoaderFn
    task_type: str
    source: str
    default_url: Optional[str]


_REGISTRY: Dict[str, _Entry] = {}


def _slug(name: str) -> str:
    """Return a filesystem-safe directory name for a dataset name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "dataset"


def dataset_checksum(path: str | Path) -> str:
    """Return the sha256 hex digest of ``path``, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_cache_dir() -> Path:
    """Return the cache directory: ``$QSARIA_BENCHMARK_CACHE`` or the default."""
    configured = os.getenv(CACHE_DIR_ENV, "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_CACHE_DIR


def _record_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _recorded_checksum(path: Path) -> Optional[str]:
    record = _record_path(path)
    if not record.is_file():
        return None
    text = record.read_text(encoding="utf-8").strip()
    return text or None


def _verify_cached(path: Path, expected_sha256: Optional[str]) -> Path:
    """Hash a cached file and compare it with the pinned or recorded digest."""
    digest = dataset_checksum(path)
    expected = expected_sha256 or _recorded_checksum(path)
    if expected and digest != expected:
        raise ChecksumMismatchError(
            f"cached benchmark file {path} has sha256 {digest} but {expected} was expected. "
            f"The cached copy is corrupt or was replaced; delete it (and {_record_path(path)}) "
            f"and load again to re-download."
        )
    if expected is None:
        _record_path(path).write_text(digest, encoding="utf-8")
    return path


def _download(url: str, destination: Path, expected_sha256: Optional[str]) -> None:
    """Download ``url`` to ``destination`` atomically, verifying any pinned digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
            with open(partial, "wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        raise DatasetDownloadError(
            f"could not download {url} to {destination}: {exc}. If this machine is offline or "
            f"behind a proxy, fetch the file elsewhere and copy it to that path, then load again."
        ) from exc

    digest = dataset_checksum(partial)
    if expected_sha256 and digest != expected_sha256:
        partial.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"downloaded {url} has sha256 {digest} but {expected_sha256} was expected; "
            f"the remote file changed or the transfer was corrupted. Nothing was cached."
        )
    partial.replace(destination)
    _record_path(destination).write_text(digest, encoding="utf-8")


def register_dataset(
    name: str,
    loader_fn: LoaderFn,
    *,
    task_type: str,
    source: str,
    default_url: Optional[str] = None,
) -> None:
    """Register ``loader_fn`` under ``name``.

    Parameters
    ----------
    name
        Registry key, also the cache sub-directory name.
    loader_fn
        Callable taking a :class:`LoaderContext` and returning a
        :class:`BenchmarkDataset`.
    task_type
        One of :data:`SUPPORTED_TASK_TYPES`.
    source
        Human-readable origin, e.g. ``"MoleculeNet (DeepChem S3 mirror)"``.
        Use :data:`LOCAL_SOURCE` for a loader that needs no network.
    default_url
        Canonical download URL, or ``None`` when no URL could be verified.
    """
    if not name or not isinstance(name, str):
        raise ValueError("dataset name must be a non-empty string")
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"unsupported task_type {task_type!r}; expected one of "
            f"{sorted(SUPPORTED_TASK_TYPES)}"
        )
    if not callable(loader_fn):
        raise TypeError(f"loader_fn for '{name}' is not callable")
    _REGISTRY[name] = _Entry(
        name=name,
        loader_fn=loader_fn,
        task_type=task_type,
        source=source,
        default_url=default_url,
    )


def list_datasets(cache_dir: Optional[str | Path] = None) -> List[Dict[str, object]]:
    """List registered datasets, sorted by name.

    ``requires_network`` is True when the loader fetches from a remote source;
    ``cached`` is True when this cache directory already holds files for it, in
    which case the dataset loads offline.
    """
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    rows: List[Dict[str, object]] = []
    for entry in sorted(_REGISTRY.values(), key=lambda item: item.name):
        dataset_dir = root / _slug(entry.name)
        rows.append(
            {
                "name": entry.name,
                "task_type": entry.task_type,
                "source": entry.source,
                "requires_network": entry.source != LOCAL_SOURCE,
                "cached": dataset_dir.is_dir() and any(dataset_dir.iterdir()),
            }
        )
    return rows


def load_dataset(
    name: str,
    cache_dir: Optional[str | Path] = None,
    allow_download: bool = True,
) -> BenchmarkDataset:
    """Load a registered benchmark dataset.

    Parameters
    ----------
    name
        A name from :func:`list_datasets`.
    cache_dir
        Download cache; defaults to :func:`default_cache_dir`.
    allow_download
        When False, a dataset missing from the cache raises
        :class:`DatasetNotCachedError` instead of hitting the network.
    """
    entry = _REGISTRY.get(name)
    if entry is None:
        available = ", ".join(sorted(_REGISTRY)) or "<registry is empty>"
        raise KeyError(f"unknown benchmark dataset '{name}'; available datasets: {available}")

    context = LoaderContext(
        name=entry.name,
        task_type=entry.task_type,
        source=entry.source,
        default_url=entry.default_url,
        cache_dir=Path(cache_dir) if cache_dir is not None else default_cache_dir(),
        allow_download=allow_download,
    )
    dataset = entry.loader_fn(context)
    _validate_dataset(dataset, entry)
    return dataset


def _validate_dataset(dataset: BenchmarkDataset, entry: _Entry) -> None:
    """Fail loudly when a loader returns a frame that does not match its claims."""
    if not isinstance(dataset, BenchmarkDataset):
        raise BenchmarkDatasetError(
            f"loader for '{entry.name}' returned {type(dataset).__name__}, not BenchmarkDataset"
        )
    columns = set(dataset.frame.columns)
    if dataset.smiles_column not in columns:
        raise BenchmarkDatasetError(
            f"benchmark '{dataset.name}': SMILES column '{dataset.smiles_column}' is missing from "
            f"the loaded frame; the upstream file layout changed."
        )
    if not dataset.target_columns:
        raise BenchmarkDatasetError(f"benchmark '{dataset.name}' declares no target columns")
    missing = [column for column in dataset.target_columns if column not in columns]
    if missing:
        raise BenchmarkDatasetError(
            f"benchmark '{dataset.name}': target columns {missing} are missing from the loaded "
            f"frame; the upstream file layout changed."
        )
    if dataset.canonical_split is not None:
        _validate_split(dataset)


def _validate_split(dataset: BenchmarkDataset) -> None:
    n_rows = len(dataset.frame)
    seen: Dict[int, str] = {}
    for split_name, indices in (dataset.canonical_split or {}).items():
        for index in indices:
            if not isinstance(index, int) or isinstance(index, bool):
                raise BenchmarkDatasetError(
                    f"benchmark '{dataset.name}': canonical split '{split_name}' contains a "
                    f"non-integer row index {index!r}"
                )
            if not 0 <= index < n_rows:
                raise BenchmarkDatasetError(
                    f"benchmark '{dataset.name}': canonical split '{split_name}' references row "
                    f"{index}, outside the {n_rows}-row frame"
                )
            previous = seen.get(index)
            if previous is not None:
                raise BenchmarkDatasetError(
                    f"benchmark '{dataset.name}': row {index} appears in both the '{previous}' "
                    f"and '{split_name}' splits, which would leak between them"
                )
            seen[index] = split_name


def _retrieved_at(path: Path) -> str:
    """Return the ISO-8601 UTC time the cached copy was written."""
    stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return stamp.isoformat(timespec="seconds")


def csv_loader(
    *,
    filename: str,
    smiles_column: str,
    target_columns: Sequence[str],
    url: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    licence: Optional[str] = None,
    version: Optional[str] = None,
    notes: str = "",
    read_csv_kwargs: Optional[Mapping[str, object]] = None,
) -> LoaderFn:
    """Build a loader for a single remote CSV (``.csv`` or ``.csv.gz``).

    The returned loader downloads once, verifies the checksum, and raises if the
    declared SMILES or target columns are absent from the file.
    """
    targets = list(target_columns)
    kwargs = dict(read_csv_kwargs or {})

    def _load(context: LoaderContext) -> BenchmarkDataset:
        path = context.fetch(filename=filename, url=url, expected_sha256=expected_sha256)
        frame = pd.read_csv(path, **kwargs)
        missing = [column for column in [smiles_column, *targets] if column not in frame.columns]
        if missing:
            raise BenchmarkDatasetError(
                f"benchmark '{context.name}': columns {missing} are absent from {path}. The "
                f"upstream file layout changed; update the loader rather than guessing columns."
            )
        return BenchmarkDataset(
            name=context.name,
            task_type=context.task_type,
            smiles_column=smiles_column,
            target_columns=list(targets),
            frame=frame,
            canonical_split=None,
            provenance=Provenance(
                source=context.source,
                url=url or context.default_url,
                version=version,
                licence=licence,
                sha256=dataset_checksum(path),
                retrieved_at=_retrieved_at(path),
                n_rows=len(frame),
                notes=notes,
            ),
        )

    return _load


def tdc_loader(
    dataset_name: str,
    *,
    group: str = "ADME",
    smiles_column: str = "Drug",
    target_column: str = "Y",
) -> LoaderFn:
    """Build a loader for a Therapeutics Data Commons single-prediction dataset.

    PyTDC is imported lazily and is not a Qsaria dependency; a clear install hint
    is raised when it is absent.  The frame is assembled by concatenating TDC's
    own ``get_split()`` frames in train/valid/test order, so ``canonical_split``
    holds the published split exactly, as row indices, with no re-splitting.

    Parameters
    ----------
    dataset_name
        TDC dataset name, passed through verbatim (e.g. ``"Caco2_Wang"``).
    group
        TDC ``single_pred`` class name, e.g. ``"ADME"`` or ``"Tox"``.
    """

    def _load(context: LoaderContext) -> BenchmarkDataset:
        try:
            import tdc
            import tdc.single_pred as tdc_single_pred
        except ImportError as exc:
            raise BenchmarkDatasetError(
                f"benchmark '{context.name}' needs PyTDC, which is not installed. Install it with "
                f"`pip install PyTDC` (it is an optional extra, not a Qsaria dependency) and load "
                f"again."
            ) from exc

        dataset_dir = context.dataset_dir()
        if not context.allow_download and not (dataset_dir.is_dir() and any(dataset_dir.iterdir())):
            raise DatasetNotCachedError(
                f"benchmark dataset '{context.name}' is not cached at {dataset_dir} and downloads "
                f"are disabled (allow_download=False). Run it once with allow_download=True, or "
                f"copy an existing TDC cache directory to that path."
            )
        dataset_dir.mkdir(parents=True, exist_ok=True)

        loader_cls = getattr(tdc_single_pred, group, None)
        if loader_cls is None:
            raise BenchmarkDatasetError(
                f"benchmark '{context.name}': tdc.single_pred has no group '{group}'"
            )
        handle = loader_cls(name=dataset_name, path=str(dataset_dir))
        splits = handle.get_split()

        frames: List[pd.DataFrame] = []
        canonical_split: Dict[str, List[int]] = {}
        offset = 0
        for split_name in ("train", "valid", "test"):
            part = splits.get(split_name)
            if part is None:
                continue
            part = pd.DataFrame(part).reset_index(drop=True)
            canonical_split[split_name] = list(range(offset, offset + len(part)))
            offset += len(part)
            frames.append(part)
        if not frames:
            raise BenchmarkDatasetError(
                f"benchmark '{context.name}': TDC returned no train/valid/test splits"
            )
        frame = pd.concat(frames, ignore_index=True)

        return BenchmarkDataset(
            name=context.name,
            task_type=context.task_type,
            smiles_column=smiles_column,
            target_columns=[target_column],
            frame=frame,
            canonical_split=canonical_split,
            provenance=Provenance(
                source=context.source,
                url=context.default_url,
                version=getattr(tdc, "__version__", None),
                licence=None,
                sha256=None,
                retrieved_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                n_rows=len(frame),
                notes=(
                    f"TDC {group} dataset '{dataset_name}'; rows ordered train, valid, test and "
                    f"canonical_split carries TDC's published split as row indices. Licence and "
                    f"file checksum are not exposed by the PyTDC API."
                ),
            ),
        )

    return _load


def _register_moleculenet() -> None:
    """Register the MoleculeNet CSVs hosted on the DeepChem S3 bucket.

    Filenames, SMILES fields and task columns were taken from the DeepChem
    ``molnet`` loader sources and confirmed against the published file headers.
    """
    specs = (
        (
            "delaney",
            "regression",
            "delaney-processed.csv",
            "smiles",
            ["measured log solubility in mols per litre"],
            "ESOL water solubility (logS); DeepChem load_delaney.",
        ),
        (
            "freesolv",
            "regression",
            "SAMPL.csv",
            "smiles",
            ["expt"],
            (
                "FreeSolv hydration free energy; 'expt' is the experimental value and 'calc' the "
                "alchemical calculation. DeepChem load_sampl."
            ),
        ),
        (
            "lipophilicity",
            "regression",
            "Lipophilicity.csv",
            "smiles",
            ["exp"],
            "ChEMBL octanol/water logD at pH 7.4; DeepChem load_lipo.",
        ),
        (
            "bace_classification",
            "classification",
            "bace.csv",
            "mol",
            ["Class"],
            (
                "BACE-1 inhibition; 'mol' holds the SMILES and 'pIC50' the regression target of "
                "the same file. DeepChem load_bace_classification."
            ),
        ),
        (
            "bbbp",
            "classification",
            "BBBP.csv",
            "smiles",
            ["p_np"],
            "Blood-brain barrier penetration; DeepChem load_bbbp.",
        ),
        (
            "clintox",
            "classification",
            "clintox.csv.gz",
            "smiles",
            ["FDA_APPROVED", "CT_TOX"],
            "Clinical trial toxicity, 2 tasks; DeepChem load_clintox.",
        ),
        (
            "tox21",
            "classification",
            "tox21.csv.gz",
            "smiles",
            [
                "NR-AR",
                "NR-AR-LBD",
                "NR-AhR",
                "NR-Aromatase",
                "NR-ER",
                "NR-ER-LBD",
                "NR-PPAR-gamma",
                "SR-ARE",
                "SR-ATAD5",
                "SR-HSE",
                "SR-MMP",
                "SR-p53",
            ],
            "Tox21 challenge, 12 tasks with missing labels; DeepChem load_tox21.",
        ),
        (
            "sider",
            "classification",
            "sider.csv.gz",
            "smiles",
            [
                "Hepatobiliary disorders",
                "Metabolism and nutrition disorders",
                "Product issues",
                "Eye disorders",
                "Investigations",
                "Musculoskeletal and connective tissue disorders",
                "Gastrointestinal disorders",
                "Social circumstances",
                "Immune system disorders",
                "Reproductive system and breast disorders",
                "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                "General disorders and administration site conditions",
                "Endocrine disorders",
                "Surgical and medical procedures",
                "Vascular disorders",
                "Blood and lymphatic system disorders",
                "Skin and subcutaneous tissue disorders",
                "Congenital, familial and genetic disorders",
                "Infections and infestations",
                "Respiratory, thoracic and mediastinal disorders",
                "Psychiatric disorders",
                "Renal and urinary disorders",
                "Pregnancy, puerperium and perinatal conditions",
                "Ear and labyrinth disorders",
                "Cardiac disorders",
                "Nervous system disorders",
                "Injury, poisoning and procedural complications",
            ],
            "SIDER marketed-drug side effects, 27 MedDRA system organ classes; load_sider.",
        ),
        (
            "hiv",
            "classification",
            "HIV.csv",
            "smiles",
            ["HIV_active"],
            (
                "DTP AIDS antiviral screen; 'HIV_active' is the binary task and 'activity' the "
                "three-class CA/CM/CI label. DeepChem load_hiv."
            ),
        ),
    )
    for name, task_type, filename, smiles_column, targets, note in specs:
        url = f"{MOLECULENET_BASE_URL}/{filename}"
        register_dataset(
            name,
            csv_loader(
                filename=filename,
                smiles_column=smiles_column,
                target_columns=targets,
                url=url,
                licence=_UNSTATED_LICENCE,
                notes=f"{note} {_MOLECULENET_SPLIT_NOTE}",
            ),
            task_type=task_type,
            source=MOLECULENET_SOURCE,
            default_url=url,
        )


def _register_tdc() -> None:
    """Register a few canonical TDC ADMET-group datasets (needs PyTDC at load time)."""
    specs = (
        ("tdc_caco2_wang", "Caco2_Wang", "ADME", "regression"),
        ("tdc_solubility_aqsoldb", "Solubility_AqSolDB", "ADME", "regression"),
        ("tdc_bbb_martins", "BBB_Martins", "ADME", "classification"),
        ("tdc_herg", "hERG", "Tox", "classification"),
    )
    for name, dataset_name, group, task_type in specs:
        register_dataset(
            name,
            tdc_loader(dataset_name, group=group),
            task_type=task_type,
            source=TDC_SOURCE,
            default_url=None,
        )


_register_moleculenet()
_register_tdc()

__all__ = [
    "CACHE_DIR_ENV",
    "DEFAULT_CACHE_DIR",
    "LOCAL_SOURCE",
    "MOLECULENET_BASE_URL",
    "SUPPORTED_TASK_TYPES",
    "BenchmarkDataset",
    "BenchmarkDatasetError",
    "ChecksumMismatchError",
    "DatasetDownloadError",
    "DatasetNotCachedError",
    "LoaderContext",
    "LoaderFn",
    "Provenance",
    "csv_loader",
    "dataset_checksum",
    "default_cache_dir",
    "list_datasets",
    "load_dataset",
    "register_dataset",
    "tdc_loader",
]
