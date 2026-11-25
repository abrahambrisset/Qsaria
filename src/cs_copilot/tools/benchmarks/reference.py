#!/usr/bin/env python
# coding: utf-8
# =====================================================================================
# WARNING - SYNTHETIC PIPELINE FIXTURE, NOT EXPERIMENTAL DATA.
#
# THE TARGETS IN THIS MODULE AND IN THE CSV FILES IT WRITES ARE COMPUTED RDKit
# DESCRIPTORS (Crippen MolLogP, TPSA). THEY ARE **NOT** MEASURED PROPERTIES.
# NOTHING HERE WAS OBSERVED IN A LABORATORY. THE STRUCTURES ARE REAL AND VALID,
# BUT THE LABELS ARE A DETERMINISTIC FUNCTION OF THE STRUCTURE, SO ANY MODEL
# TRAINED ON THEM IS ONLY RE-LEARNING AN RDKit FORMULA.
#
# THIS DATASET EXISTS SOLELY TO EXERCISE THE QSAR PIPELINE END TO END IN CI AND IN
# THE EVAL HARNESS: featurisation, splitting, training, metrics, reporting.
# IT MUST NEVER BE USED TO SUPPORT A SCIENTIFIC CLAIM ABOUT MODEL QUALITY,
# METHOD COMPARISON, PROSPECTIVE PERFORMANCE, OR CHEMICAL SERIES BEHAVIOUR.
# A good R^2 here means the featuriser can reproduce an additive atom-contribution
# sum. It says nothing about predicting any real-world endpoint.
# =====================================================================================
"""Deterministic reference dataset for end-to-end pipeline tests.

WARNING - the targets are COMPUTED RDKit DESCRIPTORS, not experimental measurements.
``clogp`` is :func:`rdkit.Chem.Crippen.MolLogP` and ``tpsa_high`` is a threshold on
:func:`rdkit.Chem.Descriptors.TPSA`. No value in this dataset was measured. The dataset
exists only so CI and the eval harness can train a real model on a real-sized table of
real molecules; it MUST NEVER be used to support a scientific claim about model quality.

Structures are enumerated combinatorially from fixed lists of real medicinal-chemistry
fragments (``scaffold + linker + tail`` concatenated as SMILES), validated and
canonicalised through RDKit, deduplicated, filtered to a drug-like-ish molecular-weight
window, and ordered by a SHA-256 digest of the canonical SMILES. There is no random
number generator anywhere in this module, so the output is reproducible byte for byte.

The one external coupling is the RDKit version: ``MolLogP`` and ``TPSA`` are RDKit
implementations, so an RDKit upgrade may shift the written values. If the
reproducibility test in ``tests/unit/test_benchmark_reference.py`` fails after a
dependency bump, regenerate the committed CSVs with :func:`write_reference_datasets`
and commit the diff.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, Literal, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

__all__ = [
    "MIN_SPLITTABLE_ROWS",
    "REFERENCE_CLASSIFICATION_PATH",
    "REFERENCE_REGRESSION_PATH",
    "TPSA_HIGH_THRESHOLD",
    "build_reference_dataset",
    "write_reference_datasets",
]

Task = Literal["regression", "classification"]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _REPO_ROOT / "tests" / "data"

#: Committed regression fixture (target ``clogp``). Present in a source checkout only.
REFERENCE_REGRESSION_PATH = _DATA_DIR / "reference_regression.csv"
#: Committed classification fixture (target ``tpsa_high``). Source checkout only.
REFERENCE_CLASSIFICATION_PATH = _DATA_DIR / "reference_classification.csv"

#: TPSA cut-off (A^2) separating the two ``tpsa_high`` classes. 60 A^2 is the usual
#: CNS-penetration rule-of-thumb cut and, unlike the stricter 75 A^2 line, it splits
#: this fragment space near 40/60 instead of leaving one class barely above 25%.
TPSA_HIGH_THRESHOLD = 60.0

#: ``build_qsar_split_payload`` rejects anything smaller, so every fixture must clear it.
MIN_SPLITTABLE_ROWS = 10

#: Molecular-weight window (Da) keeping the enumeration drug-like-ish.
_MW_RANGE = (120.0, 500.0)

#: Decimal places written for ``clogp``; also the rounding applied in memory, so the
#: returned frame and the committed file agree exactly.
_CLOGP_DECIMALS = 4

# Real fragments, written as SMILES prefixes with one open attachment point: each entry
# concatenated with a linker and a tail yields a complete, valid SMILES string.
_SCAFFOLDS: Tuple[Tuple[str, str], ...] = (
    ("benzene", "c1ccc(cc1)"),
    ("pyridin-3-yl", "c1cnc(cc1)"),
    ("naphthalen-2-yl", "c1ccc2cc(ccc2c1)"),
    ("indol-3-yl", "c1ccc2[nH]cc(c2c1)"),
    ("quinolin-3-yl", "c1ccc2ncc(cc2c1)"),
    ("morpholin-4-yl", "C1CN(CCO1)"),
    ("piperidin-4-yl", "C1CCN(CC1)"),
    ("piperazin-1-yl", "C1CN(CCN1)"),
    ("pyrimidin-2-yl", "c1cnc(nc1)"),
    ("thiophen-2-yl", "c1csc(c1)"),
    ("cyclohexyl", "C1CCC(CC1)"),
    ("furan-2-yl", "c1coc(c1)"),
    ("benzimidazol-2-yl", "c1ccc2[nH]c(nc2c1)"),
    ("tetrahydrofuran-3-yl", "C1COC(C1)"),
    ("benzofuran-2-yl", "c1ccc2oc(cc2c1)"),
    ("pyrrolidin-3-yl", "C1CNC(C1)"),
)

_LINKERS: Tuple[Tuple[str, str], ...] = (
    ("direct-bond", ""),
    ("methylene", "C"),
    ("carboxamide", "C(=O)N"),
    ("reverse-carboxamide", "NC(=O)"),
    ("oxymethylene", "OC"),
    ("sulfonamide", "S(=O)(=O)N"),
    ("ethylene", "CC"),
    ("aminomethylene", "CN"),
    ("urea", "NC(=O)N"),
    ("carbamate", "NC(=O)O"),
)

_TAILS: Tuple[Tuple[str, str], ...] = (
    ("hydrogen", ""),
    ("methyl", "C"),
    ("phenyl", "c1ccccc1"),
    ("pyridin-4-yl", "c1ccncc1"),
    ("hydroxy", "O"),
    ("amino", "N"),
    ("nitrile", "C#N"),
    ("primary-amide", "C(=O)N"),
    ("methoxy", "OC"),
    ("carboxy", "C(=O)O"),
    ("4-fluorophenyl", "c1ccc(F)cc1"),
    ("morpholin-4-yl", "N1CCOCC1"),
    ("sulfamoyl", "S(=O)(=O)N"),
    ("2-hydroxyethyl", "CCO"),
    ("acetamide", "CC(=O)N"),
    ("piperazin-1-yl", "N1CCNCC1"),
)


def _order_key(smiles: str) -> Tuple[str, str]:
    """Stable, structure-independent sort key for one canonical SMILES."""
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest(), smiles


@lru_cache(maxsize=1)
def _enumerate_pool() -> Tuple[str, ...]:
    """Enumerate every valid fragment combination as canonical SMILES.

    Returns
    -------
    tuple of str
        Deduplicated canonical SMILES ordered by ``sha256`` of the SMILES. The digest
        order is deterministic across runs and platforms (unlike :func:`hash`) and is
        uncorrelated with structure, so truncating the pool to ``n`` rows does not
        return an alphabetical, and therefore chemically skewed, slice.
    """
    unique: set[str] = set()
    for _, scaffold in _SCAFFOLDS:
        for _, linker in _LINKERS:
            for _, tail in _TAILS:
                mol = Chem.MolFromSmiles(scaffold + linker + tail)
                if mol is None:
                    continue
                if not _MW_RANGE[0] <= Descriptors.MolWt(mol) <= _MW_RANGE[1]:
                    continue
                unique.add(Chem.MolToSmiles(mol))
    return tuple(sorted(unique, key=_order_key))


def _scaffold_smiles(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES, or ``NO_SCAFFOLD`` for an acyclic molecule."""
    return MurckoScaffold.MurckoScaffoldSmiles(smiles) or "NO_SCAFFOLD"


def build_reference_dataset(task: Task = "regression", n: int = 400) -> pd.DataFrame:
    """Build the deterministic reference dataset.

    WARNING - the target column is a COMPUTED RDKit DESCRIPTOR, never a measurement.

    Parameters
    ----------
    task
        ``"regression"`` adds a ``clogp`` column (Crippen MolLogP, rounded to four
        decimals). ``"classification"`` adds a ``tpsa_high`` column, ``1`` when TPSA is
        strictly above :data:`TPSA_HIGH_THRESHOLD` and ``0`` otherwise.
    n
        Number of rows. Must be at least :data:`MIN_SPLITTABLE_ROWS` and at most the
        size of the enumerated pool.

    Returns
    -------
    pandas.DataFrame
        Columns ``compound_id``, ``smiles``, ``scaffold`` and the task's target.

    Raises
    ------
    ValueError
        If ``task`` is unknown, or ``n`` is outside the supported range.
    """
    if task not in ("regression", "classification"):
        raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")

    pool = _enumerate_pool()
    if not MIN_SPLITTABLE_ROWS <= n <= len(pool):
        raise ValueError(
            f"n must be between {MIN_SPLITTABLE_ROWS} and {len(pool)} (pool size), got {n}"
        )

    smiles = list(pool[:n])
    frame = pd.DataFrame(
        {
            "compound_id": [f"REF-{index:04d}" for index in range(1, n + 1)],
            "smiles": smiles,
            "scaffold": [_scaffold_smiles(item) for item in smiles],
        }
    )

    mols = [Chem.MolFromSmiles(item) for item in smiles]
    if task == "regression":
        values = [round(Crippen.MolLogP(mol), _CLOGP_DECIMALS) for mol in mols]
        frame["clogp"] = values
    else:
        frame["tpsa_high"] = [int(Descriptors.TPSA(mol) > TPSA_HIGH_THRESHOLD) for mol in mols]
    return frame


def write_reference_datasets(out_dir: Path | str, n: int = 400) -> Dict[str, Path]:
    """Write both reference CSVs into ``out_dir``.

    Writing is byte-reproducible: a fixed float format, no index column and an explicit
    ``\\n`` terminator, so regenerating over a committed file produces no diff.

    Parameters
    ----------
    out_dir
        Destination directory; created if missing.
    n
        Row count forwarded to :func:`build_reference_dataset`.

    Returns
    -------
    dict
        ``{"regression": path, "classification": path}``.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for task, filename in (
        ("regression", REFERENCE_REGRESSION_PATH.name),
        ("classification", REFERENCE_CLASSIFICATION_PATH.name),
    ):
        path = directory / filename
        build_reference_dataset(task=task, n=n).to_csv(
            path,
            index=False,
            float_format=f"%.{_CLOGP_DECIMALS}f",
            lineterminator="\n",
        )
        written[task] = path
    return written


if __name__ == "__main__":  # pragma: no cover - regeneration entry point
    for name, target in write_reference_datasets(_DATA_DIR).items():
        print(f"{name}: {target}")
