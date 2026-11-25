#!/usr/bin/env python
# coding: utf-8
"""Chemical train/test leakage detection.

Row-index disjointness is not enough for QSAR: the same structure written two
different ways, or a close analogue, can sit on both sides of a split and
inflate every reported metric. This module compares the two sets chemically.

Similarity is deliberately the same definition used by
``cs_copilot.tools.activity_cliffs.service``: count Morgan fingerprints
(radius 2, 2048 bits) compared with RDKit Tanimoto, so "similar" means the same
thing across the platform.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

logger = logging.getLogger(__name__)

LEAKAGE_FINGERPRINT_RADIUS = 2
LEAKAGE_FINGERPRINT_SIZE = 2048

DEFAULT_LEAKAGE_THRESHOLDS: Tuple[float, ...] = (0.95, 0.99)

#: Nearest-neighbour similarity at or above which a test compound is treated as
#: a near-duplicate for the purpose of grading severity.
SEVERITY_SIMILARITY_THRESHOLD = 0.99
#: Fraction of valid test compounds that are exact or near duplicates, mapped to
#: a severity grade. ``none`` below ``LOW``, then ``low`` / ``moderate`` / ``high``.
SEVERITY_LOW_FRACTION = 0.01
SEVERITY_MODERATE_FRACTION = 0.05
SEVERITY_HIGH_FRACTION = 0.20

EXAMPLE_CAP = 20

#: Maximum number of pairwise Tanimoto comparisons done exhaustively. Above
#: this, both sides are randomly sub-sampled (with a fixed seed) and the report
#: says so in ``warnings`` and ``sampling``. Exact-duplicate detection is a set
#: intersection and is always exhaustive.
MAX_EXHAUSTIVE_COMPARISONS = 20_000 * 20_000
LEAKAGE_SAMPLING_SEED = 1234


def _canonicalize(value: Any) -> Optional[str]:
    """Return the RDKit canonical SMILES of ``value``, or ``None`` if unparsable."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        mol = Chem.MolFromSmiles(text)
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:  # pragma: no cover - RDKit raises only on exotic input
        return None


def _canonicalize_all(values: Iterable[Any]) -> Tuple[List[str], int]:
    """Canonicalise ``values``, returning the parsable SMILES and the invalid count."""
    canonical: List[str] = []
    n_invalid = 0
    for value in values:
        smiles = _canonicalize(value)
        if smiles is None:
            n_invalid += 1
        else:
            canonical.append(smiles)
    return canonical, n_invalid


def _subsample(items: Sequence[str], size: int) -> List[str]:
    """Return ``size`` items drawn without replacement using a fixed seed."""
    rng = np.random.default_rng(LEAKAGE_SAMPLING_SEED)
    picked = np.sort(rng.choice(len(items), size=size, replace=False))
    return [items[int(index)] for index in picked]


def _grade_severity(leaked_fraction: float, exact_duplicates: int) -> str:
    """Map a leaked-test fraction to ``none`` / ``low`` / ``moderate`` / ``high``."""
    if leaked_fraction >= SEVERITY_HIGH_FRACTION:
        return "high"
    if leaked_fraction >= SEVERITY_MODERATE_FRACTION:
        return "moderate"
    if leaked_fraction >= SEVERITY_LOW_FRACTION or exact_duplicates > 0:
        return "low"
    return "none"


def _normalize_thresholds(thresholds: Iterable[float]) -> List[float]:
    """Validate and sort similarity thresholds."""
    values = sorted({float(threshold) for threshold in thresholds})
    if not values:
        raise ValueError("At least one similarity threshold is required.")
    for value in values:
        if not 0.0 < value <= 1.0:
            raise ValueError(f"Similarity thresholds must lie in (0, 1]; got {value}.")
    return values


def compute_leakage_report(
    train_smiles: Iterable[Any],
    test_smiles: Iterable[Any],
    thresholds: Iterable[float] = DEFAULT_LEAKAGE_THRESHOLDS,
) -> Dict[str, Any]:
    """Detect chemical leakage between a training and a test set.

    Both sets are canonicalised with RDKit before comparison, so ``"OCC"`` in the
    test set is reported as an *exact* duplicate of ``"CCO"`` in the training
    set. Unparsable SMILES are counted and skipped, never raised on.

    Parameters
    ----------
    train_smiles, test_smiles:
        SMILES strings. Anything that does not parse is counted as invalid.
    thresholds:
        Nearest-neighbour Tanimoto cut-offs reported in ``near_duplicates``.
        Defaults to ``(0.95, 0.99)``.

    Returns
    -------
    dict
        ``n_train`` / ``n_test`` are input row counts; ``n_invalid_*`` the
        unparsable subset of those. ``exact_duplicates`` counts *valid test rows*
        whose canonical SMILES also occurs in the training set (so internal test
        duplicates are counted once each), and ``exact_duplicate_examples`` holds
        up to 20 of the shared canonical structures.
        ``near_duplicates[str(threshold)]`` counts valid test rows whose nearest
        training neighbour reaches that Tanimoto, exact duplicates included.
        ``near_duplicate_examples`` holds up to 20 non-identical pairs above the
        lowest threshold, strongest first. The ``*_nn_similarity`` statistics and
        ``nn_similarity_percentiles`` summarise the per-test-compound nearest
        training neighbour and are ``None`` when either side has no valid
        molecule. ``sampling`` is ``None`` unless the comparison was sub-sampled.

    Notes
    -----
    Severity grades the fraction of valid test compounds that are exact
    duplicates or sit at or above a nearest-neighbour Tanimoto of
    ``SEVERITY_SIMILARITY_THRESHOLD`` (0.99): ``none`` below 1%, ``low`` from 1%,
    ``moderate`` from 5%, ``high`` from 20%. A single exact duplicate forces at
    least ``low`` and always sets ``leakage_suspected``.
    """
    cutoffs = _normalize_thresholds(thresholds)
    train_values = list(train_smiles)
    test_values = list(test_smiles)

    train_canonical, n_invalid_train = _canonicalize_all(train_values)
    test_canonical, n_invalid_test = _canonicalize_all(test_values)

    warnings: List[str] = []
    report: Dict[str, Any] = {
        "n_train": len(train_values),
        "n_test": len(test_values),
        "n_invalid_train": n_invalid_train,
        "n_invalid_test": n_invalid_test,
        "exact_duplicates": 0,
        "exact_duplicate_examples": [],
        "near_duplicates": {f"{cutoff:g}": 0 for cutoff in cutoffs},
        "near_duplicate_examples": [],
        "max_nn_similarity": None,
        "median_nn_similarity": None,
        "mean_nn_similarity": None,
        "nn_similarity_percentiles": None,
        "leakage_suspected": False,
        "severity": "none",
        "sampling": None,
        "warnings": warnings,
    }
    if n_invalid_train:
        warnings.append(f"{n_invalid_train} training SMILES could not be parsed and were skipped.")
    if n_invalid_test:
        warnings.append(f"{n_invalid_test} test SMILES could not be parsed and were skipped.")

    if not train_canonical or not test_canonical:
        warnings.append(
            "Leakage statistics unavailable: "
            f"{len(train_canonical)} valid training and {len(test_canonical)} valid test molecules."
        )
        return report

    # Exact duplicates: always exhaustive, it is only a set intersection.
    train_unique = sorted(set(train_canonical))
    train_lookup = set(train_unique)
    shared = sorted({smiles for smiles in test_canonical if smiles in train_lookup})
    exact_duplicates = sum(1 for smiles in test_canonical if smiles in train_lookup)
    report["exact_duplicates"] = exact_duplicates
    report["exact_duplicate_examples"] = shared[:EXAMPLE_CAP]

    train_compared: Sequence[str] = train_unique
    test_compared: Sequence[str] = test_canonical
    comparisons = len(train_compared) * len(test_compared)
    if comparisons > MAX_EXHAUSTIVE_COMPARISONS:
        scale = math.sqrt(MAX_EXHAUSTIVE_COMPARISONS / comparisons)
        train_compared = _subsample(train_unique, max(1, int(len(train_unique) * scale)))
        test_compared = _subsample(test_canonical, max(1, int(len(test_canonical) * scale)))
        report["sampling"] = {
            "n_train_compared": len(train_compared),
            "n_test_compared": len(test_compared),
            "seed": LEAKAGE_SAMPLING_SEED,
        }
        message = (
            f"Similarity search sub-sampled: {comparisons} pairs exceed the "
            f"{MAX_EXHAUSTIVE_COMPARISONS} comparison cap, so near-duplicate counts and "
            f"similarity statistics come from {len(test_compared)} test against "
            f"{len(train_compared)} unique training molecules, not the full sets. "
            "Exact-duplicate counts are still exhaustive."
        )
        warnings.append(message)
        logger.warning("%s", message)

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=LEAKAGE_FINGERPRINT_RADIUS,
        fpSize=LEAKAGE_FINGERPRINT_SIZE,
    )
    train_fps = [
        generator.GetCountFingerprint(Chem.MolFromSmiles(smiles)) for smiles in train_compared
    ]
    nn_similarity = np.empty(len(test_compared), dtype=float)
    nn_train_smiles: List[str] = []
    for position, smiles in enumerate(test_compared):
        fingerprint = generator.GetCountFingerprint(Chem.MolFromSmiles(smiles))
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, train_fps)
        best = int(np.argmax(similarities))
        nn_similarity[position] = float(similarities[best])
        nn_train_smiles.append(train_compared[best])

    report["max_nn_similarity"] = float(np.max(nn_similarity))
    report["median_nn_similarity"] = float(np.median(nn_similarity))
    report["mean_nn_similarity"] = float(np.mean(nn_similarity))
    report["nn_similarity_percentiles"] = {
        "p50": float(np.percentile(nn_similarity, 50)),
        "p90": float(np.percentile(nn_similarity, 90)),
        "p95": float(np.percentile(nn_similarity, 95)),
        "p99": float(np.percentile(nn_similarity, 99)),
    }
    report["near_duplicates"] = {
        f"{cutoff:g}": int(np.count_nonzero(nn_similarity >= cutoff)) for cutoff in cutoffs
    }

    lowest = cutoffs[0]
    candidates = [
        {
            "test_smiles": test_compared[position],
            "train_smiles": nn_train_smiles[position],
            "similarity": float(nn_similarity[position]),
        }
        for position in range(len(test_compared))
        if nn_similarity[position] >= lowest and test_compared[position] not in train_lookup
    ]
    candidates.sort(key=lambda item: item["similarity"], reverse=True)
    report["near_duplicate_examples"] = candidates[:EXAMPLE_CAP]

    leaked = np.count_nonzero(nn_similarity >= SEVERITY_SIMILARITY_THRESHOLD)
    leaked_fraction = max(
        float(leaked) / float(len(test_compared)),
        float(exact_duplicates) / float(len(test_canonical)),
    )
    report["severity"] = _grade_severity(leaked_fraction, exact_duplicates)
    report["leakage_suspected"] = exact_duplicates > 0 or report["severity"] != "none"

    if exact_duplicates:
        warnings.append(
            f"{exact_duplicates} test compound(s) are exact structural duplicates of "
            "training compounds; reported test performance is optimistic."
        )
    if leaked:
        warnings.append(
            f"{int(leaked)} test compound(s) have a training neighbour at Tanimoto "
            f">= {SEVERITY_SIMILARITY_THRESHOLD}."
        )
    return report
