#!/usr/bin/env python
# coding: utf-8
"""Chemprop input materialization helpers."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from .backend import InvalidPredictionInputError, PredictionTaskSpec
from .training_orchestration import (
    build_classification_prediction_frame,
    classification_task_kind,
    encode_classification_labels,
    is_classification_task,
    is_multiclass_task,
    json_safe_label,
    resolve_class_labels,
    strip_unnamed_columns,
)


def parse_chemprop_multiclass_probabilities(
    values: pd.Series,
    *,
    class_count: int | None = None,
) -> pd.DataFrame:
    """Parse Chemprop's ``<target>_prob`` vector strings into numeric columns."""
    rows = []
    expected = int(class_count) if class_count is not None else None
    for row_index, value in values.items():
        try:
            parsed = value if isinstance(value, (list, tuple)) else ast.literal_eval(str(value))
        except (SyntaxError, ValueError) as exc:
            raise InvalidPredictionInputError(
                f"Chemprop multiclass probabilities are invalid at row {row_index}: {value!r}."
            ) from exc
        if not isinstance(parsed, (list, tuple)):
            raise InvalidPredictionInputError(
                f"Chemprop multiclass probabilities must be a vector at row {row_index}."
            )
        if expected is None:
            expected = len(parsed)
        if len(parsed) != expected:
            raise InvalidPredictionInputError(
                "Chemprop multiclass probability vector length does not match the model's "
                f"class count ({len(parsed)} != {expected}) at row {row_index}."
            )
        numeric = pd.to_numeric(pd.Series(list(parsed)), errors="coerce")
        if numeric.isna().any():
            raise InvalidPredictionInputError(
                f"Chemprop multiclass probabilities contain non-numeric values at row {row_index}."
            )
        rows.append(numeric.tolist())
    return pd.DataFrame(rows, index=values.index, dtype=float).reset_index(drop=True)


def normalize_chemprop_classification_predictions(
    predictions: pd.DataFrame,
    *,
    task: PredictionTaskSpec,
    classification_targets: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Convert native Chemprop classification output to Qsaria's shared column contract."""
    targets = list(task.target_columns or [])
    metadata = dict(classification_targets or {})
    source_columns = [
        column
        for column in predictions.columns
        if column not in set(targets + [f"{target}_prob" for target in targets])
    ]
    output = predictions[source_columns].reset_index(drop=True).copy()
    for target_index, target in enumerate(targets):
        if target not in predictions.columns:
            raise InvalidPredictionInputError(
                f"Chemprop prediction output is missing target column `{target}`."
            )
        target_metadata = dict(metadata.get(target) or {})
        class_labels = list(target_metadata.get("class_labels") or [])
        if is_multiclass_task(task.task_type):
            probability_column = f"{target}_prob"
            if probability_column not in predictions.columns:
                raise InvalidPredictionInputError(
                    f"Chemprop multiclass output is missing `{probability_column}`."
                )
            probabilities = parse_chemprop_multiclass_probabilities(
                predictions[probability_column],
                class_count=target_metadata.get("class_count") or None,
            )
            if not class_labels:
                class_labels = list(range(probabilities.shape[1]))
            predicted_codes = pd.to_numeric(predictions[target], errors="coerce")
        else:
            class_labels = class_labels or [0, 1]
            positive_probability = pd.to_numeric(predictions[target], errors="coerce").clip(0, 1)
            probabilities = pd.DataFrame({0: 1.0 - positive_probability, 1: positive_probability})
            predicted_codes = (positive_probability >= 0.5).astype(int)
        canonical = build_classification_prediction_frame(
            predicted_codes=predicted_codes,
            class_labels=class_labels,
            target_column=target,
            probabilities=probabilities,
            primary_target=target_index == 0,
        )
        output = pd.concat([output, canonical], axis=1)
    return output


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


def _validate_split_payload(
    split_payload: Sequence[Mapping[str, Sequence[int]]],
    row_count: int,
    *,
    allow_empty_test: bool = False,
) -> Dict[str, int]:
    if not split_payload or not isinstance(split_payload[0], Mapping):
        raise InvalidPredictionInputError(
            "Chemprop split payload must contain one train/test or train/val/test mapping."
        )
    split_map = split_payload[0]
    counts: Dict[str, int] = {}
    seen: set[int] = set()
    if "test" not in split_map and "val" not in split_map and "validation" not in split_map:
        split_names = ("train",)
    else:
        split_names = (
            ("train", "val", "test")
            if ("val" in split_map or "validation" in split_map)
            else ("train", "test")
        )
    for split_name in split_names:
        raw_indices = split_map.get(split_name)
        if split_name == "val" and raw_indices is None:
            raw_indices = split_map.get("validation")
        if raw_indices is None:
            raise InvalidPredictionInputError(
                f"Chemprop split payload is missing `{split_name}` indices."
            )
        indices = [int(index) for index in raw_indices]
        if not indices and not (allow_empty_test and split_name == "test"):
            raise InvalidPredictionInputError(
                f"Chemprop split payload has an empty `{split_name}` split."
            )
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
    allow_empty_test: bool = False,
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
    class_metadata: Dict[str, Any] = {}
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
    elif is_classification_task(task.task_type):
        for column in target_columns:
            labels = resolve_class_labels(clean[column])
            minimum_classes = 3 if is_multiclass_task(task.task_type) else 2
            if len(labels) < minimum_classes or (
                not is_multiclass_task(task.task_type) and len(labels) != 2
            ):
                expected = "at least three" if is_multiclass_task(task.task_type) else "exactly two"
                raise InvalidPredictionInputError(
                    f"Chemprop {classification_task_kind(task.task_type, len(labels))} target "
                    f"`{column}` requires {expected} classes; found {len(labels)}."
                )
            encoded, mapping = encode_classification_labels(clean[column], labels)
            missing_target = encoded.isna()
            if bool(missing_target.any()):
                raise InvalidPredictionInputError(
                    f"Chemprop classification target `{column}` contains missing labels at rows "
                    f"{missing_target[missing_target].index[:5].tolist()}."
                )
            clean[column] = encoded.astype(int)
            class_metadata[column] = {
                "class_labels": [json_safe_label(label) for label in labels],
                "class_count": len(labels),
                "label_mapping": mapping,
                "task_kind": classification_task_kind(task.task_type, len(labels)),
                **(
                    {"positive_class_label": json_safe_label(labels[1])} if len(labels) == 2 else {}
                ),
            }
        if is_multiclass_task(task.task_type):
            class_counts = {item["class_count"] for item in class_metadata.values()}
            if len(class_counts) > 1:
                raise InvalidPredictionInputError(
                    "Chemprop multi-target multiclass classification requires every target "
                    "to use the same number of classes."
                )

    split_counts = _validate_split_payload(
        split_payload,
        len(clean),
        allow_empty_test=allow_empty_test,
    )
    split_map = split_payload[0]
    canonical_split = [
        {
            split_name: [
                int(index)
                for index in (
                    split_map.get(split_name)
                    if split_name != "val" or split_map.get(split_name) is not None
                    else split_map.get("validation")
                )
            ]
            for split_name in split_counts
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
        "classification_targets": class_metadata,
        "split_label": split_label,
        "seed": seed,
        "split_counts": split_counts,
        "split_fractions": {key: value / float(len(clean)) for key, value in split_counts.items()},
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
