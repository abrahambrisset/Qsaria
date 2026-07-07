#!/usr/bin/env python
# coding: utf-8
"""
Applicability-domain builder for QSAR training outputs.

This module intentionally builds a hybrid AD artifact:
- a compact but complete fingerprint reference store for the train split
- a human-readable index with statistics, thresholds, prototypes, and coverage hints
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or ""
    except Exception:
        return ""


def _fingerprint_generator(radius: int = 2, nbits: int = 2048):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)


def _bitvect_to_array(fp, nbits: int) -> np.ndarray:
    arr = np.zeros((nbits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _compute_nn_similarity_stats(fps: List[Any]) -> Dict[str, float]:
    if len(fps) < 2:
        return {
            "nn_median": 1.0,
            "nn_p25": 1.0,
            "nn_p10": 1.0,
            "nn_mean": 1.0,
        }

    nn_scores: List[float] = []
    for idx, fp in enumerate(fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        best_other = max(
            (float(score) for j, score in enumerate(sims) if j != idx),
            default=0.0,
        )
        nn_scores.append(best_other)

    values = np.asarray(nn_scores, dtype=float)
    return {
        "nn_mean": float(np.mean(values)),
        "nn_median": float(np.median(values)),
        "nn_p25": float(np.percentile(values, 25)),
        "nn_p10": float(np.percentile(values, 10)),
    }


def _select_prototypes(
    smiles: List[str],
    scaffolds: List[str],
    fps: List[Any],
    *,
    max_prototypes: int = 100,
) -> List[Dict[str, Any]]:
    if not smiles:
        return []

    scaffold_to_indices: Dict[str, List[int]] = {}
    for idx, scaffold in enumerate(scaffolds):
        scaffold_to_indices.setdefault(scaffold or "NO_SCAFFOLD", []).append(idx)

    ordered_scaffolds = sorted(
        scaffold_to_indices.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    selected_indices: List[int] = []
    # First pass: dominant scaffold representative
    for _, indices in ordered_scaffolds:
        selected_indices.append(indices[0])
        if len(selected_indices) >= max_prototypes:
            break

    # Second pass: greedy diversity from remaining pool
    remaining = [idx for idx in range(len(smiles)) if idx not in selected_indices]
    while remaining and len(selected_indices) < max_prototypes:
        if not selected_indices:
            selected_indices.append(remaining.pop(0))
            continue
        best_idx = None
        best_score = None
        for idx in remaining[:500]:
            sims = [DataStructs.TanimotoSimilarity(fps[idx], fps[sel]) for sel in selected_indices]
            score = max(sims) if sims else 0.0
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [
        {
            "reference_index": int(idx),
            "smiles": smiles[idx],
            "scaffold": scaffolds[idx],
        }
        for idx in selected_indices
    ]


def build_applicability_domain_from_training_data(
    *,
    dataset: pd.DataFrame,
    train_indices: List[int],
    smiles_column: str,
    output_dir: str,
    model_id: str,
    radius: int = 2,
    nbits: int = 2048,
    max_prototypes: int = 100,
) -> Dict[str, Any]:
    if not train_indices:
        raise ValueError("Cannot build applicability domain without train indices.")
    if smiles_column not in dataset.columns:
        raise ValueError(f"SMILES column not found in dataset: {smiles_column}")

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    train_df = dataset.iloc[train_indices].reset_index(drop=True).copy()
    generator = _fingerprint_generator(radius=radius, nbits=nbits)

    kept_smiles: List[str] = []
    kept_source_indices: List[int] = []
    scaffolds: List[str] = []
    fps = []
    fp_arrays = []

    for local_idx, smiles in enumerate(train_df[smiles_column].tolist()):
        mol = _mol_from_smiles(smiles)
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        fps.append(fp)
        fp_arrays.append(_bitvect_to_array(fp, nbits))
        kept_smiles.append(smiles)
        kept_source_indices.append(int(train_indices[local_idx]))
        scaffolds.append(_murcko_scaffold_smiles(mol))

    if not fps:
        raise ValueError("Applicability domain could not be built: no valid training molecules.")

    fp_matrix = np.vstack(fp_arrays).astype(np.uint8)
    nn_stats = _compute_nn_similarity_stats(fps)

    thresholds = {
        "in_domain_min": nn_stats["nn_p25"],
        "edge_domain_min": nn_stats["nn_p10"],
    }

    scaffold_counts = pd.Series(scaffolds).value_counts()
    dominant_scaffolds = [
        {"scaffold": str(scaffold), "count": int(count)}
        for scaffold, count in scaffold_counts.head(10).items()
    ]

    prototypes = _select_prototypes(
        kept_smiles,
        scaffolds,
        fps,
        max_prototypes=max_prototypes,
    )

    reference_store_path = output_path / "reference_fingerprints.npz"
    np.savez_compressed(
        reference_store_path,
        fingerprints=fp_matrix,
        train_indices=np.asarray(kept_source_indices, dtype=np.int32),
        smiles=np.asarray(kept_smiles, dtype=object),
        scaffolds=np.asarray(scaffolds, dtype=object),
    )

    manifest = {
        "model_id": model_id,
        "reference_set": "train_split",
        "num_molecules": int(len(kept_smiles)),
        "fingerprint_type": "morgan",
        "radius": radius,
        "nbits": nbits,
        "storage_format": "npz",
        "includes_smiles": True,
        "includes_scaffold_labels": True,
        "prototype_count": len(prototypes),
    }
    manifest_path = output_path / "reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    ad_index = {
        "model_id": model_id,
        "method": "hybrid_morgan_domain",
        "reference_set": "train_split",
        "train_size": int(len(kept_smiles)),
        "fingerprint": {
            "type": "morgan",
            "radius": radius,
            "nbits": nbits,
        },
        "similarity_stats": nn_stats,
        "thresholds": thresholds,
        "prototype_count": len(prototypes),
        "prototypes": prototypes,
        "dominant_scaffolds": dominant_scaffolds,
        "coverage_summary": {
            "num_unique_scaffolds": int(scaffold_counts.size),
            "largest_scaffold_fraction": (
                float(scaffold_counts.iloc[0] / len(kept_smiles))
                if len(scaffold_counts) > 0
                else 0.0
            ),
        },
    }
    ad_index_path = output_path / "applicability_domain.json"
    ad_index_path.write_text(json.dumps(ad_index, indent=2) + "\n")

    return {
        "method": ad_index["method"],
        "reference_store_path": str(reference_store_path),
        "reference_manifest_path": str(manifest_path),
        "applicability_domain_path": str(ad_index_path),
        "train_size": int(len(kept_smiles)),
        "prototype_count": len(prototypes),
        "thresholds": thresholds,
        "similarity_stats": nn_stats,
        "coverage_summary": ad_index["coverage_summary"],
    }
