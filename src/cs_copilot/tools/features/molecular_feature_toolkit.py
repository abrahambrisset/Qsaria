#!/usr/bin/env python
# coding: utf-8
"""
Toolkit for explicit molecular feature generation from curated QSAR datasets.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from agno.tools.toolkit import Toolkit
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.standardize import (
    resolve_smiles_column_name,
    standardize_smiles_column,
)

try:  # pragma: no cover - depends on how RDKit was built
    from rdkit.Avalon import pyAvalonTools
except ImportError:  # pragma: no cover - exercised only on Avalon-less builds
    pyAvalonTools = None

logger = logging.getLogger(__name__)

AVALON_UNAVAILABLE_MESSAGE = (
    "Avalon fingerprints are unavailable: this RDKit build does not provide "
    "rdkit.Avalon.pyAvalonTools. Install an RDKit build with Avalon support or pick another "
    "fingerprint."
)

DEFAULT_MORGAN_RADIUS = 2
DEFAULT_MORGAN_N_BITS = 2048
SUPPORTED_MORGAN_RADII: tuple[int, ...] = (1, 2, 3)
#: Widths offered through the agent-facing training contract.  This helper itself accepts any
#: positive width so tests and ad-hoc exports can use small fingerprints.
SUPPORTED_FINGERPRINT_N_BITS: tuple[int, ...] = (1024, 2048, 4096)
MACCS_N_BITS = 167

#: Fixed-rule descriptions of the non-Morgan structural fingerprints.  ``resizable`` is False
#: for key-based fingerprints whose bit count is defined by the key set itself.
STRUCTURAL_FINGERPRINTS: Dict[str, Dict[str, Any]] = {
    "maccs": {
        "feature_prefix": "mk_",
        "default_n_bits": MACCS_N_BITS,
        "resizable": False,
        "description": "166 MACCS structural keys returned as a 167-bit vector (bit 0 unused).",
    },
    "avalon": {
        "feature_prefix": "av_",
        "default_n_bits": 1024,
        "resizable": True,
        "description": "Avalon substructure fingerprint (rdkit.Avalon.pyAvalonTools).",
    },
    "atom_pair": {
        "feature_prefix": "ap_",
        "default_n_bits": 2048,
        "resizable": True,
        "description": "Hashed atom-pair fingerprint (topological distance between atom types).",
    },
    "topological_torsion": {
        "feature_prefix": "tt_",
        "default_n_bits": 2048,
        "resizable": True,
        "description": "Hashed topological-torsion fingerprint over four-atom paths.",
    },
}


def _resolve_output_csv(output_csv: Optional[str], input_csv: str, suffix: str) -> str:
    if output_csv:
        return S3.path(output_csv)
    stem = Path(input_csv).stem or "molecular_features"
    return S3.path(f"features/{stem}{suffix}")


def _ensure_parent_dir(path: str) -> None:
    if str(path).startswith("s3://"):
        return
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _feature_column_names(prefix: str, n_bits: int) -> List[str]:
    return [f"{prefix}{index:04d}" for index in range(n_bits)]


def _normalize_fingerprint_kind(fingerprint_kind: str) -> str:
    normalized = str(fingerprint_kind or "binary").strip().lower()
    aliases = {
        "bit": "binary",
        "bits": "binary",
        "binary": "binary",
        "count": "count",
        "counts": "count",
        "count_based": "count",
        "count-based": "count",
    }
    if normalized not in aliases:
        raise ValueError("Unsupported fingerprint_kind. Supported values are 'binary' and 'count'.")
    return aliases[normalized]


_BASIC_RDKIT_DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "FractionCSP3": Descriptors.FractionCSP3,
}

_ALL_RDKIT_DESCRIPTOR_FUNCS = dict(Descriptors._descList)


def _resolve_rdkit_descriptor_funcs(descriptor_set: str) -> Dict[str, Any]:
    normalized = descriptor_set.strip().lower()
    if normalized == "basic":
        return _BASIC_RDKIT_DESCRIPTOR_FUNCS
    if normalized == "all":
        return _ALL_RDKIT_DESCRIPTOR_FUNCS
    raise ValueError("Unsupported descriptor_set. Supported values are 'basic' and 'all'.")


def _coerce_n_jobs(n_jobs: Optional[int]) -> int:
    try:
        return max(1, int(n_jobs or 1))
    except (TypeError, ValueError):
        return 1


def _map_rows(worker, worker_args: list[tuple], n_jobs: int) -> list:
    if n_jobs == 1:
        return [worker(args) for args in worker_args]
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            return list(executor.map(worker, worker_args))
    except (OSError, PermissionError) as exc:
        logger.warning(
            "Process pool unavailable; falling back to serial feature generation: %s", exc
        )
        return [worker(args) for args in worker_args]


def _validate_morgan_parameters(radius: int, n_bits: int) -> tuple[int, int]:
    """Validate Morgan radius/size and return the coerced ``(radius, n_bits)`` pair."""
    if radius not in SUPPORTED_MORGAN_RADII:
        raise ValueError(
            f"Unsupported Morgan radius={radius!r}. Supported radii are "
            f"{list(SUPPORTED_MORGAN_RADII)}."
        )
    return int(radius), _validate_n_bits(n_bits)


def _validate_n_bits(n_bits: int) -> int:
    """Validate a folded fingerprint width and return it as an int."""
    if not isinstance(n_bits, int) or isinstance(n_bits, bool) or n_bits <= 0:
        raise ValueError("n_bits must be a positive integer.")
    return int(n_bits)


def resolve_structural_fingerprint(
    fingerprint_name: str, n_bits: Optional[int] = None
) -> Dict[str, Any]:
    """Return the resolved spec (``name``/``feature_prefix``/``n_bits``) for a structural FP.

    Raises ``ValueError`` for an unknown fingerprint, for a size override on a key-based
    fingerprint whose width is fixed, and for an unsupported size on a hashed fingerprint.
    """
    normalized = str(fingerprint_name or "").strip().lower()
    entry = STRUCTURAL_FINGERPRINTS.get(normalized)
    if entry is None:
        raise ValueError(
            f"Unsupported fingerprint_name={fingerprint_name!r}. Supported values are "
            f"{sorted(STRUCTURAL_FINGERPRINTS)}."
        )
    if not entry["resizable"]:
        resolved_n_bits = int(entry["default_n_bits"])
        if n_bits is not None and int(n_bits) != resolved_n_bits:
            raise ValueError(
                f"{normalized} has a fixed width of {resolved_n_bits} bits and cannot be resized "
                f"to {int(n_bits)}."
            )
    else:
        resolved_n_bits = _validate_n_bits(
            int(entry["default_n_bits"]) if n_bits is None else n_bits
        )
    if normalized == "avalon" and pyAvalonTools is None:
        raise ValueError(AVALON_UNAVAILABLE_MESSAGE)
    return {
        "name": normalized,
        "feature_prefix": str(entry["feature_prefix"]),
        "n_bits": resolved_n_bits,
        "description": str(entry["description"]),
    }


def _morgan_row_worker(args: tuple[str, int, int, str, bool]) -> List[int]:
    smiles, radius, n_bits, fingerprint_kind, use_features = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not compute Morgan fingerprint for standardized SMILES: {smiles}")
    invariants_generator = (
        rdFingerprintGenerator.GetMorganFeatureAtomInvGen() if use_features else None
    )
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, atomInvariantsGenerator=invariants_generator
    )
    if fingerprint_kind == "count":
        fingerprint = fp_generator.GetCountFingerprintAsNumPy(mol)
    else:
        fingerprint = fp_generator.GetFingerprintAsNumPy(mol)
    return fingerprint.astype(int).tolist()


def _structural_fp_row_worker(args: tuple[str, str, int]) -> List[int]:
    smiles, fingerprint_name, n_bits = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            f"Could not compute {fingerprint_name} fingerprint for standardized SMILES: {smiles}"
        )
    if fingerprint_name == "maccs":
        return [int(bit) for bit in MACCSkeys.GenMACCSKeys(mol)]
    if fingerprint_name == "avalon":
        if pyAvalonTools is None:
            raise ValueError(AVALON_UNAVAILABLE_MESSAGE)
        return [int(bit) for bit in pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)]
    if fingerprint_name == "atom_pair":
        generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    elif fingerprint_name == "topological_torsion":
        generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
    else:  # pragma: no cover - guarded by resolve_structural_fingerprint
        raise ValueError(f"Unsupported fingerprint_name={fingerprint_name!r}.")
    return generator.GetFingerprintAsNumPy(mol).astype(int).tolist()


def _rdkit_descriptor_row_worker(args: tuple[str, str]) -> List[float]:
    smiles, descriptor_set = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not compute RDKit descriptors for standardized SMILES: {smiles}")
    return [float(func(mol)) for func in _resolve_rdkit_descriptor_funcs(descriptor_set).values()]


def _build_base_output_dataframe(
    working: pd.DataFrame,
    *,
    include_input_columns: bool,
    input_columns_to_keep: Optional[List[str]],
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    required = list(required_columns or [])
    if input_columns_to_keep is not None:
        requested_columns = list(input_columns_to_keep)
        for column in required:
            if column not in requested_columns:
                requested_columns.insert(0, column)
        missing_columns = [column for column in requested_columns if column not in working.columns]
        if missing_columns:
            raise ValueError(f"Requested input columns to keep are missing: {missing_columns}")
        return working[requested_columns].copy()
    if include_input_columns:
        if not required:
            return working.copy()
        ordered = [column for column in required if column in working.columns]
        ordered.extend(column for column in working.columns if column not in ordered)
        return working[ordered].copy()
    if required:
        missing_required = [column for column in required if column not in working.columns]
        if missing_required:
            raise ValueError(f"Required output columns are missing: {missing_required}")
        return working[required].copy()
    return pd.DataFrame(index=working.index)


def _normalize_input_columns_to_keep(
    input_columns_to_keep: Optional[List[str]],
    *,
    requested_smiles_column: str,
    resolved_smiles_column: str,
) -> Optional[List[str]]:
    if input_columns_to_keep is None:
        return None

    normalized: List[str] = []
    smiles_aliases = {
        str(requested_smiles_column),
        str(resolved_smiles_column),
        "smiles",
        "SMILES",
    }
    for column in input_columns_to_keep:
        replacement = (
            "smiles" if str(column) in smiles_aliases or str(column).lower() == "smiles" else column
        )
        if replacement not in normalized:
            normalized.append(replacement)
    return normalized


def _coerce_join_on(join_on: Optional[List[str]]) -> List[str]:
    if join_on is None:
        return ["smiles"]
    if isinstance(join_on, str):
        return [join_on]
    if not join_on:
        raise ValueError("join_on cannot be empty.")
    return list(join_on)


def _validate_join_columns(df: pd.DataFrame, join_on: List[str], *, df_name: str) -> None:
    missing = [column for column in join_on if column not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing join columns: {missing}")


def _validate_unique_keys(df: pd.DataFrame, join_on: List[str], *, df_name: str) -> None:
    duplicate_mask = df.duplicated(subset=join_on, keep=False)
    if duplicate_mask.any():
        raise ValueError(
            f"{df_name} contains duplicate rows for join keys {join_on}; V1 requires a strict 1:1 join."
        )


def _canonicalize_join_columns(
    df: pd.DataFrame,
    join_on: List[str],
    *,
    df_name: str,
    canonicalize_smiles: bool = True,
) -> pd.DataFrame:
    """Normalize join columns in-memory before strict table joins.

    At the moment we canonicalize `smiles` because chemically equivalent rows can
    still differ textually between intermediate artifacts. This keeps the join
    strict while reducing accidental mismatches caused by representation drift.
    """
    normalized = df.copy()
    if "smiles" in join_on and canonicalize_smiles:
        if "smiles" not in normalized.columns:
            raise ValueError(f"{df_name} is missing join column 'smiles'.")
        normalized = standardize_smiles_column(normalized, "smiles")
        invalid_mask = normalized["smiles"].isna()
        if invalid_mask.any():
            invalid_rows = int(invalid_mask.sum())
            raise ValueError(
                f"{df_name} has {invalid_rows} row(s) with invalid or missing standardized SMILES in join column 'smiles'."
            )
    return normalized


class MolecularFeatureToolkit(Toolkit):
    """Explicit tools for transforming molecular datasets into tabular features."""

    def __init__(self, *, block_direct_qsar_training: bool = False):
        super().__init__("molecular_features")
        self.block_direct_qsar_training = block_direct_qsar_training
        self.register(self.smiles_to_morgan_fingerprints)
        self.register(self.smiles_to_structural_fingerprints)
        self.register(self.smiles_to_rdkit_descriptors)
        self.register(self.build_tabular_qsar_dataset)

    def _guard_direct_qsar_training(self, tool_name: str) -> None:
        if not self.block_direct_qsar_training:
            return
        raise ValueError(
            f"{tool_name} is a low-level export/debug helper and is blocked in the QSAR training agent. "
            "For LightGBM or TabICL training with Morgan/RDKit representations, call "
            "train_lightgbm_model or train_tabicl_model with representation_name instead; those tools "
            "prepare and cache features internally."
        )

    def smiles_to_morgan_fingerprints(
        self,
        input_csv: str,
        smiles_column: str = "smiles",
        output_csv: Optional[str] = None,
        radius: int = DEFAULT_MORGAN_RADIUS,
        n_bits: int = DEFAULT_MORGAN_N_BITS,
        include_input_columns: bool = False,
        input_columns_to_keep: Optional[List[str]] = None,
        feature_prefix: str = "fp_",
        fingerprint_kind: str = "binary",
        n_jobs: Optional[int] = None,
        use_features: bool = False,
    ) -> Dict[str, Any]:
        """
        Transform a SMILES column into a tabular CSV of Morgan fingerprints.

        Parameters
        ----------
        radius:
            Morgan neighbourhood radius, one of ``(1, 2, 3)``. Defaults to 2 (ECFP4-like).
        n_bits:
            Folded fingerprint width, any positive integer. Defaults to 2048; the training
            contract restricts agent-chosen widths to ``SUPPORTED_FINGERPRINT_N_BITS``.
        use_features:
            When True use feature-invariant atom types (FCFP-style) instead of the default
            connectivity invariants (ECFP-style). Defaults to False.

        This tool is intentionally explicit and testable on its own so future
        tabular backends (for example TabICL or tree models) can reuse the same
        featurization step without coupling it to training.
        """
        self._guard_direct_qsar_training("smiles_to_morgan_fingerprints")
        started_at = time.monotonic()
        normalized_fingerprint_kind = _normalize_fingerprint_kind(fingerprint_kind)
        resolved_n_jobs = _coerce_n_jobs(n_jobs)
        radius, n_bits = _validate_morgan_parameters(radius, n_bits)
        use_features = bool(use_features)
        if not feature_prefix:
            raise ValueError("feature_prefix cannot be empty.")

        with S3.open(input_csv, "r") as fh:
            df = pd.read_csv(fh)

        resolved_smiles_column = resolve_smiles_column_name(df, smiles_column)
        working = standardize_smiles_column(df.copy(), resolved_smiles_column)
        if resolved_smiles_column != "smiles":
            working["smiles"] = working[resolved_smiles_column]
            working = working.drop(columns=[resolved_smiles_column])
        normalized_columns_to_keep = _normalize_input_columns_to_keep(
            input_columns_to_keep,
            requested_smiles_column=smiles_column,
            resolved_smiles_column=resolved_smiles_column,
        )

        invalid_mask = working["smiles"].isna()
        if invalid_mask.any():
            invalid_rows = int(invalid_mask.sum())
            raise ValueError(
                f"Cannot generate Morgan fingerprints: {invalid_rows} row(s) have invalid or missing standardized SMILES."
            )

        feature_columns = _feature_column_names(feature_prefix, n_bits)
        worker_args = [
            (smiles, radius, n_bits, normalized_fingerprint_kind, use_features)
            for smiles in working["smiles"].tolist()
        ]
        fingerprint_rows = _map_rows(_morgan_row_worker, worker_args, resolved_n_jobs)

        feature_df = pd.DataFrame(fingerprint_rows, columns=feature_columns)

        base_df = _build_base_output_dataframe(
            working,
            include_input_columns=include_input_columns,
            input_columns_to_keep=normalized_columns_to_keep,
            required_columns=["smiles"],
        )

        output_df = pd.concat(
            [base_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1
        )

        resolved_output_csv = _resolve_output_csv(output_csv, input_csv, "_morgan_fp.csv")
        with S3.open(resolved_output_csv, "w") as fh:
            output_df.to_csv(fh, index=False)

        kept_columns = list(base_df.columns)
        return {
            "output_csv": resolved_output_csv,
            "rows_in": int(len(df)),
            "rows_out": int(len(output_df)),
            "smiles_column": "smiles",
            "source_smiles_column": resolved_smiles_column,
            "radius": radius,
            "n_bits": n_bits,
            "fingerprint_kind": normalized_fingerprint_kind,
            "use_features": use_features,
            "n_jobs": resolved_n_jobs,
            "num_features": len(feature_columns),
            "feature_prefix": feature_prefix,
            "feature_columns_sample": feature_columns[:5],
            "input_columns_kept": kept_columns,
            "duration_seconds": round(time.monotonic() - started_at, 3),
        }

    def smiles_to_structural_fingerprints(
        self,
        input_csv: str,
        fingerprint_name: str,
        smiles_column: str = "smiles",
        output_csv: Optional[str] = None,
        n_bits: Optional[int] = None,
        include_input_columns: bool = False,
        input_columns_to_keep: Optional[List[str]] = None,
        feature_prefix: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Transform a SMILES column into a tabular CSV of non-Morgan structural fingerprints.

        Parameters
        ----------
        fingerprint_name:
            One of ``maccs`` (167 fixed keys), ``avalon``, ``atom_pair`` or
            ``topological_torsion``. The hashed fingerprints accept any positive ``n_bits``;
            ``maccs`` has a fixed width and rejects any override.
        feature_prefix:
            Column prefix. Defaults to the fingerprint's registered prefix (``mk_``, ``av_``,
            ``ap_``, ``tt_``) so applicability-domain feature-kind mapping stays stable.

        All fingerprints here are produced as binary bit vectors.
        """
        self._guard_direct_qsar_training("smiles_to_structural_fingerprints")
        started_at = time.monotonic()
        resolved_n_jobs = _coerce_n_jobs(n_jobs)
        fingerprint = resolve_structural_fingerprint(fingerprint_name, n_bits)
        resolved_prefix = feature_prefix or fingerprint["feature_prefix"]
        if not resolved_prefix:
            raise ValueError("feature_prefix cannot be empty.")

        with S3.open(input_csv, "r") as fh:
            df = pd.read_csv(fh)

        resolved_smiles_column = resolve_smiles_column_name(df, smiles_column)
        working = standardize_smiles_column(df.copy(), resolved_smiles_column)
        if resolved_smiles_column != "smiles":
            working["smiles"] = working[resolved_smiles_column]
            working = working.drop(columns=[resolved_smiles_column])
        normalized_columns_to_keep = _normalize_input_columns_to_keep(
            input_columns_to_keep,
            requested_smiles_column=smiles_column,
            resolved_smiles_column=resolved_smiles_column,
        )

        invalid_mask = working["smiles"].isna()
        if invalid_mask.any():
            invalid_rows = int(invalid_mask.sum())
            raise ValueError(
                f"Cannot generate {fingerprint['name']} fingerprints: {invalid_rows} row(s) have "
                "invalid or missing standardized SMILES."
            )

        feature_columns = _feature_column_names(resolved_prefix, fingerprint["n_bits"])
        worker_args = [
            (smiles, fingerprint["name"], fingerprint["n_bits"])
            for smiles in working["smiles"].tolist()
        ]
        fingerprint_rows = _map_rows(_structural_fp_row_worker, worker_args, resolved_n_jobs)
        feature_df = pd.DataFrame(fingerprint_rows, columns=feature_columns)

        base_df = _build_base_output_dataframe(
            working,
            include_input_columns=include_input_columns,
            input_columns_to_keep=normalized_columns_to_keep,
            required_columns=["smiles"],
        )
        output_df = pd.concat(
            [base_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1
        )

        resolved_output_csv = _resolve_output_csv(
            output_csv, input_csv, f"_{fingerprint['name']}_fp.csv"
        )
        with S3.open(resolved_output_csv, "w") as fh:
            output_df.to_csv(fh, index=False)

        return {
            "output_csv": resolved_output_csv,
            "rows_in": int(len(df)),
            "rows_out": int(len(output_df)),
            "smiles_column": "smiles",
            "source_smiles_column": resolved_smiles_column,
            "fingerprint_name": fingerprint["name"],
            "n_bits": fingerprint["n_bits"],
            "fingerprint_kind": "binary",
            "n_jobs": resolved_n_jobs,
            "num_features": len(feature_columns),
            "feature_prefix": resolved_prefix,
            "feature_columns_sample": feature_columns[:5],
            "input_columns_kept": list(base_df.columns),
            "duration_seconds": round(time.monotonic() - started_at, 3),
        }

    def smiles_to_rdkit_descriptors(
        self,
        input_csv: str,
        smiles_column: str = "smiles",
        output_csv: Optional[str] = None,
        descriptor_set: str = "basic",
        include_input_columns: bool = False,
        input_columns_to_keep: Optional[List[str]] = None,
        n_jobs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Transform a SMILES column into a tabular CSV of RDKit descriptors.

        Supported descriptor sets:
        - `basic`: short, fixed descriptor set for lightweight workflows
        - `all`: full RDKit descriptor list exposed by `Descriptors._descList`
        """
        self._guard_direct_qsar_training("smiles_to_rdkit_descriptors")
        started_at = time.monotonic()
        resolved_n_jobs = _coerce_n_jobs(n_jobs)
        descriptor_funcs = _resolve_rdkit_descriptor_funcs(descriptor_set)

        with S3.open(input_csv, "r") as fh:
            df = pd.read_csv(fh)

        resolved_smiles_column = resolve_smiles_column_name(df, smiles_column)
        working = standardize_smiles_column(df.copy(), resolved_smiles_column)
        if resolved_smiles_column != "smiles":
            working["smiles"] = working[resolved_smiles_column]
            working = working.drop(columns=[resolved_smiles_column])
        normalized_columns_to_keep = _normalize_input_columns_to_keep(
            input_columns_to_keep,
            requested_smiles_column=smiles_column,
            resolved_smiles_column=resolved_smiles_column,
        )

        invalid_mask = working["smiles"].isna()
        if invalid_mask.any():
            invalid_rows = int(invalid_mask.sum())
            raise ValueError(
                f"Cannot generate RDKit descriptors: {invalid_rows} row(s) have invalid or missing standardized SMILES."
            )

        descriptor_names = list(descriptor_funcs.keys())
        descriptor_columns = [f"desc_{name}" for name in descriptor_names]

        worker_args = [(smiles, descriptor_set) for smiles in working["smiles"].tolist()]
        descriptor_rows = _map_rows(_rdkit_descriptor_row_worker, worker_args, resolved_n_jobs)
        descriptor_df = pd.DataFrame(descriptor_rows, columns=descriptor_columns)
        base_df = _build_base_output_dataframe(
            working,
            include_input_columns=include_input_columns,
            input_columns_to_keep=normalized_columns_to_keep,
            required_columns=["smiles"],
        )
        output_df = pd.concat(
            [base_df.reset_index(drop=True), descriptor_df.reset_index(drop=True)], axis=1
        )

        resolved_output_csv = _resolve_output_csv(output_csv, input_csv, "_rdkit_desc.csv")
        with S3.open(resolved_output_csv, "w") as fh:
            output_df.to_csv(fh, index=False)

        kept_columns = list(base_df.columns)
        return {
            "output_csv": resolved_output_csv,
            "rows_in": int(len(df)),
            "rows_out": int(len(output_df)),
            "smiles_column": "smiles",
            "source_smiles_column": resolved_smiles_column,
            "descriptor_set": descriptor_set,
            "n_jobs": resolved_n_jobs,
            "num_descriptors": len(descriptor_columns),
            "descriptor_names": descriptor_names,
            "descriptor_columns_sample": descriptor_columns[:5],
            "input_columns_kept": kept_columns,
            "duration_seconds": round(time.monotonic() - started_at, 3),
        }

    def build_tabular_qsar_dataset(
        self,
        base_csv: str,
        output_csv: Optional[str] = None,
        feature_csvs: Optional[List[str]] = None,
        join_on: Optional[List[str]] = None,
        base_columns_to_keep: Optional[List[str]] = None,
        drop_duplicate_feature_columns: bool = True,
        canonicalize_smiles_join: bool = True,
    ) -> Dict[str, Any]:
        """
        Build a final tabular QSAR dataset by combining a curated base CSV with
        one or more precomputed molecular feature tables.
        """
        self._guard_direct_qsar_training("build_tabular_qsar_dataset")
        started_at = time.monotonic()
        if not feature_csvs:
            raise ValueError("feature_csvs must contain at least one feature table.")

        join_columns = _coerce_join_on(join_on)

        with S3.open(base_csv, "r") as fh:
            base_df = pd.read_csv(fh)
        base_df = _canonicalize_join_columns(
            base_df,
            join_columns,
            df_name="base_csv",
            canonicalize_smiles=canonicalize_smiles_join,
        )

        _validate_join_columns(base_df, join_columns, df_name="base_csv")
        _validate_unique_keys(base_df, join_columns, df_name="base_csv")

        if base_columns_to_keep is not None:
            missing_columns = [
                column for column in base_columns_to_keep if column not in base_df.columns
            ]
            if missing_columns:
                raise ValueError(f"base_csv is missing requested base columns: {missing_columns}")
            assembled_df = base_df[base_columns_to_keep].copy()
            base_columns_kept = list(base_columns_to_keep)
        else:
            assembled_df = base_df.copy()
            base_columns_kept = list(base_df.columns)

        feature_sources: List[str] = []
        added_feature_columns: List[str] = []

        for index, feature_csv in enumerate(feature_csvs, start=1):
            with S3.open(feature_csv, "r") as fh:
                feature_df = pd.read_csv(fh)

            source_name = f"feature_csv[{index}]"
            feature_df = _canonicalize_join_columns(
                feature_df,
                join_columns,
                df_name=source_name,
                canonicalize_smiles=canonicalize_smiles_join,
            )
            _validate_join_columns(feature_df, join_columns, df_name=source_name)
            _validate_unique_keys(feature_df, join_columns, df_name=source_name)

            non_join_columns = [
                column for column in feature_df.columns if column not in join_columns
            ]
            if not non_join_columns:
                raise ValueError(
                    f"{source_name} does not contain any feature columns beyond join keys {join_columns}."
                )

            colliding_columns = [
                column for column in non_join_columns if column in assembled_df.columns
            ]
            if colliding_columns:
                if drop_duplicate_feature_columns:
                    non_join_columns = [
                        column for column in non_join_columns if column not in colliding_columns
                    ]
                else:
                    raise ValueError(
                        f"{source_name} has feature columns that already exist in the assembled dataset: {colliding_columns}"
                    )

            if not non_join_columns:
                raise ValueError(
                    f"{source_name} only contributed duplicate feature columns after collision filtering."
                )

            feature_subset = feature_df[join_columns + non_join_columns].copy()
            merged_df = assembled_df.merge(
                feature_subset,
                on=join_columns,
                how="left",
                sort=False,
                validate="one_to_one",
            )

            if len(merged_df) != len(assembled_df):
                raise ValueError(
                    f"{source_name} changed the row count during merge; expected {len(assembled_df)} rows but got {len(merged_df)}."
                )

            missing_feature_rows = merged_df[non_join_columns].isna().all(axis=1)
            if missing_feature_rows.any():
                missing_count = int(missing_feature_rows.sum())
                missing_examples = (
                    merged_df.loc[missing_feature_rows, join_columns]
                    .head(3)
                    .to_dict(orient="records")
                )
                raise ValueError(
                    f"{source_name} could not be joined for {missing_count} base row(s) using keys {join_columns}. "
                    f"Examples: {missing_examples}"
                )

            assembled_df = merged_df
            feature_sources.append(S3.path(feature_csv))
            added_feature_columns.extend(non_join_columns)

        resolved_output_csv = _resolve_output_csv(output_csv, base_csv, "_tabular_qsar.csv")
        _ensure_parent_dir(resolved_output_csv)
        with S3.open(resolved_output_csv, "w") as fh:
            assembled_df.to_csv(fh, index=False)

        return {
            "output_csv": resolved_output_csv,
            "rows_in_base": int(len(base_df)),
            "rows_out": int(len(assembled_df)),
            "join_on": join_columns,
            "base_columns_kept": base_columns_kept,
            "feature_sources": feature_sources,
            "num_feature_tables": len(feature_sources),
            "num_added_feature_columns": len(added_feature_columns),
            "final_column_count": int(len(assembled_df.columns)),
            "feature_column_samples": added_feature_columns[:5],
            "canonicalize_smiles_join": canonicalize_smiles_join,
            "duration_seconds": round(time.monotonic() - started_at, 3),
        }
