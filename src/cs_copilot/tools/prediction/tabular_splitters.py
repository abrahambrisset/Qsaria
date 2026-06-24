#!/usr/bin/env python
# coding: utf-8
"""
Deterministic tabular split helpers for non-Chemprop backends.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from cs_copilot.storage import S3

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


def _target_counts(n_rows: int, split_sizes: List[float]) -> Dict[str, int]:
    train_size, val_size, test_size = split_sizes
    test_n = int(round(n_rows * test_size))
    val_n = int(round(n_rows * val_size))
    train_n = n_rows - test_n - val_n
    if train_n <= 0 or val_n <= 0 or test_n <= 0:
        raise InvalidPredictionInputError(
            "split_sizes produced an empty split; adjust split sizes or provide more rows."
        )
    return {"train": train_n, "val": val_n, "test": test_n}


def _group_balanced_split_payload(
    group_labels: List[str],
    *,
    split_sizes: List[float],
    random_state: int,
) -> List[Dict[str, List[int]]]:
    n_rows = len(group_labels)
    counts = _target_counts(n_rows, split_sizes)
    rng = np.random.default_rng(random_state)

    groups: Dict[str, List[int]] = {}
    for idx, label in enumerate(group_labels):
        groups.setdefault(label or "UNGROUPED", []).append(idx)

    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    assigned = {"train": [], "val": [], "test": []}
    current = {"train": 0, "val": 0, "test": 0}

    for _, indices in group_items:
        best_split = None
        best_score = None
        for split_name in ("train", "val", "test"):
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
    allocated = set(assigned["train"]) | set(assigned["val"]) | set(assigned["test"])
    remainder = sorted(all_indices - allocated)
    if remainder:
        assigned["train"].extend(remainder)

    for split_name in assigned:
        assigned[split_name] = sorted(set(int(i) for i in assigned[split_name]))

    return [assigned]


def _group_validation_split(
    remaining_indices: List[int],
    *,
    group_labels: List[str],
    validation_fraction: float,
    random_state: int,
) -> tuple[List[int], List[int]]:
    rng = np.random.default_rng(random_state)
    groups: Dict[str, List[int]] = {}
    for idx in remaining_indices:
        groups.setdefault(group_labels[idx] or "UNGROUPED", []).append(idx)

    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)
    val_target = max(1, int(round(len(remaining_indices) * validation_fraction)))
    val: List[int] = []
    train: List[int] = []
    for _, indices in group_items:
        if len(val) < val_target:
            val.extend(indices)
        else:
            train.extend(indices)
    if not train and val:
        train.append(val.pop())
    return sorted(set(train)), sorted(set(val))


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


def _group_labels_for_family(
    df: pd.DataFrame,
    *,
    split_family: str,
    smiles_column: Optional[str],
    feature_columns: Optional[List[str]],
    random_state: int,
) -> Optional[List[str]]:
    if split_family == "random":
        return None
    if split_family == "scaffold":
        if not smiles_column or smiles_column not in df.columns:
            raise InvalidPredictionInputError(
                "Scaffold split requires a valid smiles column in the training dataset."
            )
        return [_murcko_scaffold_smiles(smiles) for smiles in df[smiles_column].tolist()]
    if split_family in {"cluster", "cluster_kmeans", "kmeans"}:
        return _feature_cluster_labels(
            df,
            feature_columns=feature_columns,
            random_state=random_state,
        )
    raise InvalidPredictionInputError(f"Unsupported split_family for k-fold generation: {split_family}")


def build_tabular_kfold_split_payloads(
    *,
    df: pd.DataFrame,
    split_family: str,
    n_folds: int,
    random_state: int,
    validation_fraction: float = 0.1,
    smiles_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
) -> List[Dict[str, List[int]]]:
    """Build train/val/test payloads for random, scaffold, or cluster k-fold CV."""
    n_rows = len(df)
    if n_rows < max(10, n_folds):
        raise InvalidPredictionInputError("K-fold split generation requires enough rows for all folds.")
    if n_folds < 2:
        raise InvalidPredictionInputError("n_folds must be >= 2.")
    if not (0.0 < validation_fraction < 0.5):
        raise InvalidPredictionInputError("validation_fraction must be in (0, 0.5).")

    normalized_family = "cluster" if split_family in {"cluster_kmeans", "kmeans"} else split_family
    indices = np.arange(n_rows)
    group_labels = _group_labels_for_family(
        df,
        split_family=normalized_family,
        smiles_column=smiles_column,
        feature_columns=feature_columns,
        random_state=random_state,
    )

    payloads: List[Dict[str, List[int]]] = []
    if group_labels is None:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_iter = splitter.split(indices)
        for fold_train_val_idx, fold_test_idx in fold_iter:
            train_idx, val_idx = train_test_split(
                fold_train_val_idx,
                test_size=max(1, int(round(len(fold_train_val_idx) * validation_fraction))),
                random_state=random_state,
                shuffle=True,
            )
            payloads.append(
                {
                    "train": sorted(int(i) for i in train_idx),
                    "val": sorted(int(i) for i in val_idx),
                    "test": sorted(int(i) for i in fold_test_idx),
                }
            )
        return payloads

    splitter = GroupKFold(n_splits=n_folds)
    for fold_train_val_idx, fold_test_idx in splitter.split(indices, groups=group_labels):
        train_idx, val_idx = _group_validation_split(
            [int(i) for i in fold_train_val_idx],
            group_labels=group_labels,
            validation_fraction=validation_fraction,
            random_state=random_state,
        )
        payloads.append(
            {
                "train": train_idx,
                "val": val_idx,
                "test": sorted(int(i) for i in fold_test_idx),
            }
        )
    return payloads


def materialize_tabular_split_payloads(
    *,
    protocol_policy: Dict[str, object],
    train_csv: str,
    smiles_column: Optional[str],
    feature_columns: Optional[List[str]],
) -> Dict[str, object]:
    """Attach explicit k-fold split payloads to a resolved protocol policy."""
    split_runs = list(protocol_policy.get("split_runs") or [])
    if not any(isinstance(run, dict) and run.get("requires_split_payload") for run in split_runs):
        return protocol_policy

    with S3.open(train_csv, "r") as fh:
        df = pd.read_csv(fh)

    payload_cache: Dict[tuple[str, int, int], List[Dict[str, List[int]]]] = {}
    materialized_runs: List[Dict[str, object]] = []
    for raw_run in split_runs:
        if not isinstance(raw_run, dict):
            materialized_runs.append(raw_run)
            continue
        run = dict(raw_run)
        if run.get("requires_split_payload"):
            family = str(run.get("split_family") or "random")
            n_folds = int(run.get("n_folds") or 5)
            fold_index = int(run.get("fold_index") or 1)
            split_seed = int(run.get("split_seed") or run.get("seed") or 42)
            cache_key = (family, n_folds, split_seed)
            if cache_key not in payload_cache:
                payload_cache[cache_key] = build_tabular_kfold_split_payloads(
                    df=df,
                    split_family=family,
                    n_folds=n_folds,
                    random_state=split_seed,
                    smiles_column=smiles_column,
                    feature_columns=feature_columns,
                )
            payloads = payload_cache[cache_key]
            if fold_index < 1 or fold_index > len(payloads):
                raise InvalidPredictionInputError(
                    f"Fold index {fold_index} is outside generated payload range 1..{len(payloads)}."
                )
            run["split_payload"] = [payloads[fold_index - 1]]
        materialized_runs.append(run)

    materialized_policy = dict(protocol_policy)
    materialized_policy["split_runs"] = materialized_runs
    seed_policy = dict(materialized_policy.get("seed_policy") or {})
    seed_policy["split_runs"] = materialized_runs
    materialized_policy["seed_policy"] = seed_policy
    return materialized_policy


def build_tabular_split_payload(
    *,
    df: pd.DataFrame,
    split_type: str,
    split_sizes: List[float],
    random_state: int,
    smiles_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
) -> List[Dict[str, List[int]]]:
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
        relative_val_size = counts["val"] / float(counts["train"] + counts["val"])
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val_size,
            random_state=random_state,
            shuffle=True,
        )
        return [
            {
                "train": sorted(int(i) for i in train_idx),
                "val": sorted(int(i) for i in val_idx),
                "test": sorted(int(i) for i in test_idx),
            }
        ]

    if split_type == "scaffold_balanced":
        if not smiles_column or smiles_column not in df.columns:
            raise InvalidPredictionInputError(
                "Scaffold split requires a valid smiles column in the training dataset."
            )
        scaffolds = [_murcko_scaffold_smiles(smiles) for smiles in df[smiles_column].tolist()]
        return _group_balanced_split_payload(
            scaffolds,
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
            split_sizes=split_sizes,
            random_state=random_state,
        )

    raise InvalidPredictionInputError(
        f"Unsupported split_type for tabular split generation: {split_type}"
    )
