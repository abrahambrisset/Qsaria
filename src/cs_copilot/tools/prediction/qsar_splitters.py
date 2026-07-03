#!/usr/bin/env python
# coding: utf-8
"""Common deterministic QSARIA split helpers."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .backend import InvalidPredictionInputError


def _murcko_scaffold_smiles(smiles: str) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        return "NO_SCAFFOLD"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "NO_SCAFFOLD"
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "NO_SCAFFOLD"
    except Exception:
        return "NO_SCAFFOLD"


def _normalize_split_sizes(split_sizes: List[float]) -> List[float]:
    if len(split_sizes) not in (2, 3):
        raise InvalidPredictionInputError("split_sizes must contain [train, test] or [train, val, test].")
    total = float(sum(split_sizes))
    if total <= 0:
        raise InvalidPredictionInputError("split_sizes must sum to a positive value.")
    normalized = [float(value) / total for value in split_sizes]
    if normalized[0] <= 0 or normalized[-1] <= 0 or any(value < 0 for value in normalized):
        raise InvalidPredictionInputError("split_sizes require positive train/test and non-negative validation.")
    if len(normalized) == 3 and normalized[1] == 0:
        return [normalized[0], normalized[2]]
    return normalized


def _target_counts(n_rows: int, split_sizes: List[float]) -> Dict[str, int]:
    split_sizes = _normalize_split_sizes(split_sizes)
    if len(split_sizes) == 2:
        train_size, test_size = split_sizes
        test_n = int(round(n_rows * test_size))
        train_n = n_rows - test_n
        if train_n <= 0 or test_n <= 0:
            raise InvalidPredictionInputError(
                "split_sizes produced an empty split; adjust split sizes or provide more rows."
            )
        return {"train": train_n, "test": test_n}
    train_size, val_size, test_size = split_sizes
    test_n = int(round(n_rows * test_size))
    val_n = int(round(n_rows * val_size))
    train_n = n_rows - test_n - val_n
    if train_n <= 0 or val_n <= 0 or test_n <= 0:
        raise InvalidPredictionInputError(
            "split_sizes produced an empty split; adjust split sizes or provide more rows."
        )
    return {"train": train_n, "val": val_n, "test": test_n}


def _finalize_split(
    assigned: Dict[str, List[int]],
    *,
    split_type: str,
    split_sizes: List[float],
    random_state: int,
) -> List[Dict[str, Any]]:
    clean = {
        split_name: sorted(set(int(i) for i in indices))
        for split_name, indices in assigned.items()
        if indices
    }
    payload_for_hash = {key: clean[key] for key in sorted(clean)}
    split_hash = hashlib.sha256(
        json.dumps(payload_for_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    clean["metadata"] = {
        "split_type": split_type,
        "split_sizes": _normalize_split_sizes(split_sizes),
        "split_counts": {key: len(value) for key, value in payload_for_hash.items()},
        "has_validation": bool(clean.get("val")),
        "random_state": int(random_state),
        "split_hash": split_hash,
    }
    return [clean]


def _group_balanced_split_payload(
    group_labels: List[str],
    *,
    split_type: str,
    split_sizes: List[float],
    random_state: int,
) -> List[Dict[str, Any]]:
    n_rows = len(group_labels)
    counts = _target_counts(n_rows, split_sizes)
    rng = np.random.default_rng(random_state)

    groups: Dict[str, List[int]] = {}
    for idx, label in enumerate(group_labels):
        groups.setdefault(label or "UNGROUPED", []).append(idx)

    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    split_names = tuple(counts)
    assigned = {split_name: [] for split_name in split_names}
    current = {split_name: 0 for split_name in split_names}

    for _, indices in group_items:
        best_split = None
        best_score = None
        for split_name in split_names:
            remaining = counts[split_name] - current[split_name]
            projected = remaining - len(indices)
            overflow_penalty = 0 if projected >= 0 else abs(projected) * 10_000
            fill_ratio = current[split_name] / max(counts[split_name], 1)
            score = overflow_penalty + fill_ratio
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name
        assigned[best_split].extend(indices)
        current[best_split] += len(indices)

    all_indices = set(range(n_rows))
    allocated = set().union(*(set(indices) for indices in assigned.values()))
    remainder = sorted(all_indices - allocated)
    if remainder:
        assigned["train"].extend(remainder)

    # ponytail: grouped toy datasets can starve a target split; move one row rather than return invalid splits.
    for split_name in split_names:
        if not assigned[split_name]:
            donor = max(split_names, key=lambda name: len(assigned[name]))
            if len(assigned[donor]) <= 1:
                raise InvalidPredictionInputError("Grouped split could not create non-empty requested splits.")
            assigned[split_name].append(assigned[donor].pop())

    return _finalize_split(
        assigned,
        split_type=split_type,
        split_sizes=split_sizes,
        random_state=random_state,
    )


def _feature_cluster_labels(
    df: pd.DataFrame,
    *,
    feature_columns: Optional[List[str]],
    random_state: int,
) -> List[str]:
    if not feature_columns:
        raise InvalidPredictionInputError("KMeans split requires explicit numeric feature columns.")
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise InvalidPredictionInputError(f"KMeans split is missing feature columns: {missing}")
    n_rows = len(df)
    matrix = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(matrix)
    n_clusters = max(3, min(20, int(math.sqrt(max(n_rows, 1) / 2.0))))
    n_clusters = min(n_clusters, n_rows)
    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(scaled)
    return [f"cluster_{label}" for label in labels.tolist()]


def build_qsar_split_payload(
    *,
    df: pd.DataFrame,
    split_type: str,
    split_sizes: List[float],
    random_state: int,
    smiles_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    n_rows = len(df)
    if n_rows < 10:
        raise InvalidPredictionInputError("Split generation requires at least 10 rows.")

    counts = _target_counts(n_rows, split_sizes)
    indices = np.arange(n_rows)

    if split_type == "random":
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=counts["test"],
            random_state=random_state,
            shuffle=True,
        )
        if "val" in counts:
            relative_val_size = counts["val"] / float(counts["train"] + counts["val"])
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=relative_val_size,
                random_state=random_state,
                shuffle=True,
            )
            assigned = {
                "train": list(train_idx),
                "val": list(val_idx),
                "test": list(test_idx),
            }
        else:
            assigned = {
                "train": list(train_val_idx),
                "test": list(test_idx),
            }
        return _finalize_split(
            assigned,
            split_type=split_type,
            split_sizes=split_sizes,
            random_state=random_state,
        )

    if split_type == "scaffold_balanced":
        if not smiles_column or smiles_column not in df.columns:
            raise InvalidPredictionInputError(
                "Scaffold split requires a valid smiles column in the training dataset."
            )
        scaffolds = [_murcko_scaffold_smiles(smiles) for smiles in df[smiles_column].tolist()]
        return _group_balanced_split_payload(
            scaffolds,
            split_type=split_type,
            split_sizes=split_sizes,
            random_state=random_state,
        )

    if split_type == "kmeans":
        labels = _feature_cluster_labels(
            df,
            feature_columns=feature_columns,
            random_state=random_state,
        )
        return _group_balanced_split_payload(
            labels,
            split_type=split_type,
            split_sizes=split_sizes,
            random_state=random_state,
        )

    raise InvalidPredictionInputError(
        f"Unsupported split_type for QSARIA split generation: {split_type}"
    )


def build_tabular_split_payload(**kwargs: Any) -> List[Dict[str, Any]]:
    return build_qsar_split_payload(**kwargs)


def build_repeated_kfold_split_payloads(
    *,
    df: pd.DataFrame,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> Dict[str, List[Dict[str, Any]]]:
    n_rows = len(df)
    if n_rows < n_splits:
        raise InvalidPredictionInputError("Cross-validation requires at least n_splits rows.")
    if n_splits < 2:
        raise InvalidPredictionInputError("Cross-validation requires n_splits >= 2.")
    if n_repeats < 1:
        raise InvalidPredictionInputError("Cross-validation requires n_repeats >= 1.")

    payloads: Dict[str, List[Dict[str, Any]]] = {}
    splitter = RepeatedKFold(
        n_splits=int(n_splits),
        n_repeats=int(n_repeats),
        random_state=int(random_state),
    )
    for run_index, (train_idx, test_idx) in enumerate(splitter.split(np.arange(n_rows)), start=1):
        repeat_index = ((run_index - 1) // n_splits) + 1
        fold_index = ((run_index - 1) % n_splits) + 1
        label = f"cv_repeat_{repeat_index}_fold_{fold_index}"
        assigned = {
            "train": [int(idx) for idx in train_idx.tolist()],
            "test": [int(idx) for idx in test_idx.tolist()],
        }
        payload = _finalize_split(
            assigned,
            split_type="cross_validation",
            split_sizes=[len(train_idx) / n_rows, len(test_idx) / n_rows],
            random_state=random_state,
        )
        payload[0]["metadata"].update(
            {
                "cv_repeat": repeat_index,
                "cv_fold": fold_index,
                "cv_run_index": run_index,
                "n_splits": int(n_splits),
                "n_repeats": int(n_repeats),
            }
        )
        payloads[label] = payload
    return payloads


def build_full_train_split_payload(*, df: pd.DataFrame) -> List[Dict[str, Any]]:
    n_rows = len(df)
    if n_rows < 1:
        raise InvalidPredictionInputError("Final refit requires at least one row.")
    train_indices = list(range(n_rows))
    split_hash = hashlib.sha256(
        json.dumps({"train": train_indices}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return [
        {
            "train": train_indices,
            "metadata": {
                "split_type": "final_refit",
                "split_sizes": [1.0],
                "split_counts": {"train": int(n_rows)},
                "has_validation": False,
                "split_hash": split_hash,
                "final_refit": True,
            },
        }
    ]
