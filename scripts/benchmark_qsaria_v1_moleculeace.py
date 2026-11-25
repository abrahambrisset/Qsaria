#!/usr/bin/env python3
"""Run fixed Qsaria V1 and baseline fits on corrected MoleculeACE partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tomllib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from threadpoolctl import threadpool_limits

from cs_copilot.tools.prediction.leakage import audit_molecular_overlap
from scripts.benchmark_qsaria_v1 import ROOT, fingerprints, predict_models, sha256, structures

PINNED_SOURCES = (
    "scripts/benchmark_qsaria_v1.py",
    "src/cs_copilot/tools/chemistry/standardize.py",
    "src/cs_copilot/tools/prediction/leakage.py",
    "src/cs_copilot/tools/prediction/lightgbm_backend.py",
    "src/cs_copilot/tools/prediction/qsar_contracts.py",
    "src/cs_copilot/tools/prediction/qsar_splitters.py",
    "src/cs_copilot/tools/prediction/qsar_training_policy.py",
    "uv.lock",
)


def read_official_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["smiles", "y", "split", "cliff_mol"])
    if set(frame.split) != {"train", "test"}:
        raise ValueError(f"Expected official train and test assignments in {path.name}.")
    if not frame.cliff_mol.isin([0, 1]).all():
        raise ValueError(f"Invalid official cliff_mol flags in {path.name}.")
    values = pd.to_numeric(frame.y, errors="coerce")
    if not np.isfinite(values).all():
        raise ValueError(f"Invalid official activity values in {path.name}.")
    return (
        frame.assign(y=values)
        .rename(columns={"smiles": "Drug", "y": "Y"})
        .reset_index(names="row_id")
    )


def score_cohort(frame, prediction_columns):
    cliff = frame.loc[frame.cliff_mol.eq(1)]
    return {
        "n": len(frame),
        "n_cliff": len(cliff),
        "by_seed": {
            seed: {
                model: {
                    "rmse": (
                        float(root_mean_squared_error(frame.Y, frame[column]))
                        if len(frame)
                        else None
                    ),
                    "rmse_cliff": (
                        float(root_mean_squared_error(cliff.Y, cliff[column]))
                        if len(cliff)
                        else None
                    ),
                }
                for model, column in columns.items()
            }
            for seed, columns in prediction_columns.items()
        },
    }


def check_sources(revision: str) -> dict[str, str]:
    checksums = {}
    for relative in PINNED_SOURCES:
        pinned = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT)
        expected = hashlib.sha256(pinned).hexdigest()
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Working source {relative} differs from Qsaria revision {revision}.")
        checksums[relative] = expected
    checksums["scripts/benchmark_qsaria_v1_moleculeace.py"] = sha256(Path(__file__))
    return checksums


def evaluate_dataset(path: Path, output: Path, protocol: dict) -> dict:
    frame = structures(read_official_data(path))
    source_fit, source_test = frame.loc[frame.split.eq("train")], frame.loc[frame.split.eq("test")]
    fit, test = source_fit.loc[source_fit.usable], source_test.loc[source_test.usable]
    if fit.empty or test.empty:
        raise ValueError(f"No model-ready rows in an official partition of {path.name}.")
    fit_x = fingerprints(fit, protocol["radius"], protocol["bits"])
    test_x = fingerprints(test, protocol["radius"], protocol["bits"])
    predictions, times = predict_models(
        output,
        "regression",
        fit_x,
        fit.Y.to_numpy(),
        test_x,
        protocol["trees"],
        protocol["seeds"],
    )
    metadata = test[["row_id", "identity"]].copy()
    metadata["identity_seen_in_train"] = test.identity.isin(fit.identity)
    prediction_columns = {}
    for seed, models in predictions.items():
        prediction_columns[seed] = {model: f"{model}_{seed}" for model in models}
        for model, values in models.items():
            metadata[prediction_columns[seed][model]] = values
    saved = output / "predictions_unlabeled.csv"
    metadata.to_csv(saved, index=False)
    scored = pd.read_csv(saved).merge(
        test[["row_id", "Y", "cliff_mol"]], on="row_id", sort=False, validate="one_to_one"
    )
    scored.to_csv(output / "predictions.csv", index=False)
    source_counts = {}
    for name, source in (("train", source_fit), ("test", source_test)):
        source_counts[name] = {
            "n": len(source),
            "n_cliff": int(source.cliff_mol.sum()),
            "n_rejected": int((~source.usable).sum()),
            "n_cliff_rejected": int((source.cliff_mol.eq(1) & ~source.usable).sum()),
        }
    exact_overlap = scored.identity_seen_in_train
    return {
        "official_source_counts": source_counts,
        "raw_smiles_exact_overlap_test_rows": int(test.Drug.isin(fit.Drug).sum()),
        "official_cliff_test_rows_with_training_identity": int(
            (scored.cliff_mol.eq(1) & exact_overlap).sum()
        ),
        "repeated_identity_rows": {
            "train": int(fit.identity.duplicated().sum()),
            "test": int(test.identity.duplicated().sum()),
        },
        "identity_audit": audit_molecular_overlap(fit.Drug, test.Drug),
        "cohorts": {
            "all_scored": score_cohort(scored, prediction_columns),
            "identity_unseen": score_cohort(scored.loc[~exact_overlap], prediction_columns),
        },
        "fit_predict_seconds_by_seed": times,
        "unlabeled_predictions_sha256": sha256(saved),
        "scored_predictions_sha256": sha256(output / "predictions.csv"),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = tomllib.loads(args.config.read_text())
    if protocol["workers"] != 1:
        parser.error("This protocol requires one CPU worker.")
    source_checksums = check_sources(protocol["qsaria_source_revision"])
    datasets = {name: args.datasets / f"{name}.csv" for name in protocol["datasets"]}
    for name, path in datasets.items():
        if sha256(path) != protocol["datasets"][name]["sha256"]:
            parser.error(f"MoleculeACE data checksum mismatch for {name}.")
    output = args.output.resolve()
    output.mkdir(parents=True)
    manifest = {
        "protocol": protocol,
        "config_sha256": sha256(args.config),
        "source_sha256": source_checksums,
        "manifest_scope": "Frozen protocol and checksummed data and source inputs",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            package: version(package)
            for package in ("rdkit", "lightgbm", "scikit-learn", "numpy", "pandas", "threadpoolctl")
        },
    }
    write_json(output / "manifest.json", manifest)
    print(f"Frozen input manifest SHA-256: {sha256(output / 'manifest.json')}", flush=True)
    results = {}
    with threadpool_limits(limits=1):
        for name, path in datasets.items():
            directory = output / name
            directory.mkdir()
            results[name] = evaluate_dataset(path, directory, protocol)
            print(f"{name}: final metrics saved", flush=True)
    write_json(output / "moleculeace.json", {"manifest": manifest, "datasets": results})


if __name__ == "__main__":
    main()
