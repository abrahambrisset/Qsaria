#!/usr/bin/env python
# coding: utf-8
"""Chemprop input materialization helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from .backend import InvalidPredictionInputError, PredictionTaskSpec
from .training_orchestration import strip_unnamed_columns


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"path": str(path)}
    try:
        stat = path.stat()
        payload.update({"size_bytes": stat.st_size, "mtime": stat.st_mtime})
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        payload["sha256"] = digest.hexdigest()
    except Exception:
        pass
    return payload


def _validate_split_payload(split_payload: Sequence[Mapping[str, Sequence[int]]], row_count: int) -> Dict[str, int]:
    if not split_payload or not isinstance(split_payload[0], Mapping):
        raise InvalidPredictionInputError("Chemprop split payload must contain one train/val/test mapping.")
    split_map = split_payload[0]
    counts: Dict[str, int] = {}
    seen: set[int] = set()
    for split_name in ("train", "val", "test"):
        raw_indices = split_map.get(split_name)
        if raw_indices is None:
            raise InvalidPredictionInputError(f"Chemprop split payload is missing `{split_name}` indices.")
        indices = [int(index) for index in raw_indices]
        if not indices:
            raise InvalidPredictionInputError(f"Chemprop split payload has an empty `{split_name}` split.")
        invalid = [index for index in indices if index < 0 or index >= row_count]
        if invalid:
            raise InvalidPredictionInputError(
                f"Chemprop split payload has out-of-range `{split_name}` indices: {invalid[:5]}"
            )
        overlap = seen.intersection(indices)
        if overlap:
            raise InvalidPredictionInputError(
                f"Chemprop split payload has overlapping indices in `{split_name}`: {sorted(overlap)[:5]}"
            )
        seen.update(indices)
        counts[split_name] = len(indices)
    if len(seen) != row_count:
        missing = sorted(set(range(row_count)).difference(seen))
        raise InvalidPredictionInputError(
            f"Chemprop split payload does not assign every row; missing indices: {missing[:5]}"
        )
    return counts


def materialize_chemprop_inputs(
    *,
    source_csv: str,
    output_dir: Path,
    task: PredictionTaskSpec,
    split_payload: Sequence[Mapping[str, Sequence[int]]],
    split_label: str,
    seed: int | None,
) -> Dict[str, Any]:
    """Write Chemprop-clean CSV and native splits-file aligned to that CSV."""
    source_path = Path(source_csv).expanduser()
    df = strip_unnamed_columns(pd.read_csv(source_path))

    smiles_columns = list(task.smiles_columns or ["smiles"])
    target_columns = list(task.target_columns or [])
    required_columns = list(dict.fromkeys([*smiles_columns, *target_columns]))
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise InvalidPredictionInputError(f"Chemprop input is missing columns: {missing}")

    for column in smiles_columns:
        values = df[column]
        empty = values.isna() | values.astype(str).str.strip().eq("")
        if bool(empty.any()):
            raise InvalidPredictionInputError(
                f"Chemprop input column `{column}` contains empty SMILES at rows {empty[empty].index[:5].tolist()}."
            )

    clean = df[required_columns].copy()
    if task.task_type == "regression":
        for column in target_columns:
            numeric = pd.to_numeric(clean[column], errors="coerce")
            missing_target = numeric.isna()
            if bool(missing_target.any()):
                raise InvalidPredictionInputError(
                    f"Chemprop regression target `{column}` contains non-numeric or missing values "
                    f"at rows {missing_target[missing_target].index[:5].tolist()}."
                )
            clean[column] = numeric

    split_counts = _validate_split_payload(split_payload, len(clean))
    canonical_split = [
        {
            split_name: [int(index) for index in split_payload[0][split_name]]
            for split_name in ("train", "val", "test")
        }
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    training_csv = output_dir / "chemprop_training_input.csv"
    splits_file = output_dir / "chemprop_splits.json"
    manifest_path = output_dir / "chemprop_input_manifest.json"

    clean.to_csv(training_csv, index=False)
    splits_file.write_text(json.dumps(canonical_split, indent=2) + "\n")
    manifest = {
        "source_dataset": _file_fingerprint(source_path),
        "chemprop_training_input_csv": str(training_csv),
        "chemprop_splits_file": str(splits_file),
        "smiles_columns": smiles_columns,
        "target_columns": target_columns,
        "task_type": task.task_type,
        "row_count": int(len(clean)),
        "columns": list(clean.columns),
        "split_label": split_label,
        "seed": seed,
        "split_counts": split_counts,
        "split_fractions": {
            key: value / float(len(clean)) for key, value in split_counts.items()
        },
        "index_alignment": (
            "splits_file indices refer to row positions in chemprop_training_input.csv; "
            "the adapter does not drop rows."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "split_payload": canonical_split,
    }
