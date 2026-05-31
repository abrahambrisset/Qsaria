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
from sklearn.model_selection import train_test_split
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

    for split_name in ("train", "val", "test"):
        if assigned[split_name]:
            continue
        donor = max(
            (name for name, indices in assigned.items() if name != split_name and len(indices) > 1),
            key=lambda name: (len(assigned[name]), name),
            default=None,
        )
        if donor is None:
            raise InvalidPredictionInputError(
                "Could not build non-empty train/val/test splits from grouped assignments."
            )
        moved_index = sorted(assigned[donor])[-1]
        assigned[donor].remove(moved_index)
        assigned[split_name].append(moved_index)

    for split_name in assigned:
        assigned[split_name] = sorted({int(i) for i in assigned[split_name]})

    return [assigned]


def build_tabular_split_payload(
    *,
    df: pd.DataFrame,
    split_type: str,
    split_sizes: List[float],
    random_state: int,
    smiles_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    stratify_column: Optional[str] = None,
) -> List[Dict[str, List[int]]]:
    n_rows = len(df)
    if n_rows < 10:
        raise InvalidPredictionInputError("Split generation requires at least 10 rows.")

    counts = _target_counts(n_rows, split_sizes)
    indices = np.arange(n_rows)

    if split_type == "random":
        if stratify_column and stratify_column in df.columns:
            stratify_values = df[stratify_column].reset_index(drop=True)
            class_counts = stratify_values.value_counts(dropna=False)
            n_classes = int(len(class_counts))
            can_stratify = (
                n_classes > 1
                and int(class_counts.min()) >= 3
                and counts["train"] >= n_classes
                and counts["val"] >= n_classes
                and counts["test"] >= n_classes
            )
            if can_stratify:
                try:
                    train_val_idx, test_idx = train_test_split(
                        indices,
                        test_size=counts["test"],
                        random_state=random_state,
                        shuffle=True,
                        stratify=stratify_values,
                    )
                    relative_val_size = counts["val"] / float(counts["train"] + counts["val"])
                    train_idx, val_idx = train_test_split(
                        train_val_idx,
                        test_size=relative_val_size,
                        random_state=random_state,
                        shuffle=True,
                        stratify=stratify_values.iloc[train_val_idx],
                    )
                    return [
                        {
                            "train": sorted(int(i) for i in train_idx),
                            "val": sorted(int(i) for i in val_idx),
                            "test": sorted(int(i) for i in test_idx),
                        }
                    ]
                except ValueError:
                    pass

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
        if not feature_columns:
            raise InvalidPredictionInputError(
                "KMeans split requires explicit numeric feature columns."
            )
        missing = [column for column in feature_columns if column not in df.columns]
        if missing:
            raise InvalidPredictionInputError(f"KMeans split is missing feature columns: {missing}")
        matrix = (
            df[feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        scaled = StandardScaler().fit_transform(matrix)
        n_clusters = max(3, min(20, int(math.sqrt(max(n_rows, 1) / 2.0))))
        n_clusters = min(n_clusters, n_rows)
        labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(
            scaled
        )
        return _group_balanced_split_payload(
            [f"cluster_{label}" for label in labels.tolist()],
            split_sizes=split_sizes,
            random_state=random_state,
        )

    raise InvalidPredictionInputError(
        f"Unsupported split_type for tabular split generation: {split_type}"
    )
