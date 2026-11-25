#!/usr/bin/env python3
"""Run the frozen Qsaria V1 benchmark on the published TDC ADMET partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import tomllib
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from cs_copilot.tools.chemistry.standardize import standardize_smiles
from cs_copilot.tools.prediction.applicability_domain import (
    fit_bounding_box_domain,
    score_bounding_box_domain,
)
from cs_copilot.tools.prediction.backend import PredictionModelRecord, PredictionTaskSpec
from cs_copilot.tools.prediction.lightgbm_backend import LightGBMBackend
from cs_copilot.tools.prediction.qsar_contracts import LightGBMRunRequest

ROOT = Path(__file__).resolve().parents[1]
METRICS = {"mae": mean_absolute_error, "auroc": roc_auc_score}
MODELS = ("lightgbm", "dummy", "random_forest")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_partition(archive: Path, dataset: str, partition: str, columns=None) -> pd.DataFrame:
    with ZipFile(archive) as bundle:
        with bundle.open(f"admet_group/{dataset}/{partition}.csv") as source:
            frame = pd.read_csv(source, usecols=columns)
    return frame.rename_axis("row_id").reset_index()


def structures(frame: pd.DataFrame) -> pd.DataFrame:
    identities = frame.Drug.map(standardize_smiles)
    molecules = identities.map(lambda value: Chem.MolFromSmiles(value) if value else None)
    usable = molecules.map(lambda molecule: molecule is not None and molecule.GetNumAtoms() > 0)
    return frame.assign(identity=identities, molecule=molecules, usable=usable)


def development_indices(frame: pd.DataFrame, task: str, seed: int):
    if task == "classification":
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train, validation = next(splitter.split(frame, frame.Y, groups=frame.identity))
    if not set(frame.iloc[train].identity).isdisjoint(frame.iloc[validation].identity):
        raise RuntimeError("Development split contains a repeated identity across partitions.")
    return train, validation


def score(metric: str, truth: np.ndarray, predictions: np.ndarray) -> float:
    return float(METRICS[metric](truth, predictions))


def paired_delta(truth, model, reference, identities, metric, repeats, seed):
    group_codes = pd.factorize(identities, sort=True)[0]
    row_order = np.argsort(group_codes, kind="stable")
    groups = np.split(row_order, np.flatnonzero(np.diff(group_codes[row_order])) + 1)
    random = np.random.default_rng(seed)
    differences = []
    for sample in random.integers(len(groups), size=(repeats, len(groups))):
        indices = np.concatenate([groups[position] for position in sample])
        if metric == "auroc" and len(np.unique(truth[indices])) < 2:
            continue
        differences.append(
            score(metric, truth[indices], model[indices])
            - score(metric, truth[indices], reference[indices])
        )
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "delta": score(metric, truth, model) - score(metric, truth, reference),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "valid": len(differences),
    }


def fingerprints(frame: pd.DataFrame, radius: int, bits: int) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    values = np.stack([generator.GetFingerprintAsNumPy(molecule) for molecule in frame.molecule])
    return pd.DataFrame(values, columns=[f"fp_{index:04d}" for index in range(bits)])


def predict_lightgbm(
    directory: Path, task: str, train_csv: Path, evaluation_csv: Path, columns, trees, seed
):
    backend = LightGBMBackend()
    result = backend.train_model(
        LightGBMRunRequest(
            train_csv=str(train_csv),
            output_dir=str(directory / "model"),
            task_type=task,
            target_columns=["Y"],
            feature_columns=list(columns),
            final_refit=True,
            split_sizes=[1.0],
            device_type="cpu",
            n_estimators=trees,
            n_jobs=1,
            random_state=seed,
        )
    )
    record = PredictionModelRecord(
        "benchmark",
        "lightgbm",
        result["model_path"],
        PredictionTaskSpec(task_type=task, target_columns=["Y"]),
    )
    prediction_path = directory / "qsaria_predictions.csv"
    backend.predict_from_csv(str(evaluation_csv), record, str(prediction_path))
    predictions = pd.read_csv(prediction_path)
    return predictions[
        "positive_probability" if task == "classification" else "prediction"
    ].to_numpy()


def predict_models(directory, task, fit_x, fit_y, evaluation_x, trees, seeds):
    train_csv = directory / "fit_features.csv"
    evaluation_csv = directory / "evaluation_features.csv"
    fit_x.assign(Y=fit_y).to_csv(train_csv, index=False)
    evaluation_x.to_csv(evaluation_csv, index=False)
    classification = task == "classification"
    dummy_type = DummyClassifier if classification else DummyRegressor
    forest_type = RandomForestClassifier if classification else RandomForestRegressor
    predictions = {}
    timings = {}
    for seed in seeds:
        run = directory / str(seed)
        run.mkdir()
        start = time.monotonic()
        lightgbm = predict_lightgbm(
            run, task, train_csv, evaluation_csv, fit_x.columns, trees, seed
        )
        seconds = {"lightgbm": round(time.monotonic() - start, 3)}
        scores = {"lightgbm": lightgbm}
        for name, model in (
            ("dummy", dummy_type()),
            ("random_forest", forest_type(n_estimators=trees, random_state=seed, n_jobs=1)),
        ):
            start = time.monotonic()
            model.fit(fit_x, fit_y)
            scores[name] = (
                model.predict_proba(evaluation_x)[:, 1]
                if classification
                else model.predict(evaluation_x)
            )
            seconds[name] = round(time.monotonic() - start, 3)
        predictions[seed] = scores
        timings[seed] = seconds
    return predictions, timings


def evaluate_dataset(archive, directory, dataset, setting, config, phase):
    started = time.monotonic()
    task, metric = setting["task"], setting["metric"]
    development = structures(read_partition(archive, dataset, "train_val"))
    valid = development.loc[development.usable]
    if phase == "development":
        train, validation = development_indices(valid, task, config["split_seed"])
        fit, evaluation = valid.iloc[train], valid.iloc[validation]
        source_evaluation_rows = len(evaluation)
    else:
        fit = valid
        public_test = structures(read_partition(archive, dataset, "test", ["Drug_ID", "Drug"]))
        source_evaluation_rows = len(public_test)
        evaluation = public_test.loc[public_test.usable]
        public_test.loc[~public_test.usable, ["row_id", "Drug_ID", "Drug", "identity"]].to_csv(
            directory / "rejected_test.csv", index=False
        )

    development.loc[~development.usable, ["row_id", "Drug_ID", "Drug", "identity"]].to_csv(
        directory / "rejected_development.csv", index=False
    )
    saved = evaluation[["row_id", "Drug_ID", "identity"]].copy()
    saved["identity_seen_in_fit"] = saved.identity.isin(fit.identity)
    fit_x = fingerprints(fit, config["radius"], config["bits"])
    evaluation_x = fingerprints(evaluation, config["radius"], config["bits"])
    domain = fit_bounding_box_domain(
        feature_frame=fit_x,
        feature_columns=list(fit_x.columns),
        output_dir=directory / "domain",
        model_id=dataset,
        feature_space="morgan_binary",
    )
    domain_scores = score_bounding_box_domain(
        feature_frame=evaluation_x, applicability_domain=domain
    )
    if not domain_scores["available"]:
        raise RuntimeError(f"Cannot score the bounding-box domain for {dataset}.")
    saved["bounding_box_status"] = domain_scores["scores"].ad_status.to_numpy()
    predictions, timings = predict_models(
        directory,
        task,
        fit_x,
        fit.Y.to_numpy(),
        evaluation_x,
        config["trees"],
        config[phase]["seeds"],
    )

    if phase == "development":
        saved["Y"] = evaluation.Y
    else:
        labels = read_partition(archive, dataset, "test", ["Y"]).set_index("row_id")
        saved["Y"] = saved.row_id.map(labels.Y)
    for seed, values in predictions.items():
        for model, model_predictions in values.items():
            saved[f"{model}_{seed}"] = model_predictions
    predictions_path = directory / "predictions.csv"
    saved.to_csv(predictions_path, index=False)
    saved = pd.read_csv(predictions_path)

    cohorts = {"all_scored": saved}
    if saved.identity_seen_in_fit.any():
        cohorts["identity_unseen"] = saved.loc[~saved.identity_seen_in_fit]
    evaluations = {}
    first_seed = config[phase]["seeds"][0]
    for name, cohort in cohorts.items():
        truth = cohort.Y.to_numpy()
        evaluations[name] = {
            "n": len(cohort),
            "scores_by_model_seed": {
                seed: {
                    model: score(metric, truth, cohort[f"{model}_{seed}"].to_numpy())
                    for model in MODELS
                }
                for seed in config[phase]["seeds"]
            },
            "lightgbm_minus_random_forest_first_seed": paired_delta(
                truth,
                cohort[f"lightgbm_{first_seed}"].to_numpy(),
                cohort[f"random_forest_{first_seed}"].to_numpy(),
                cohort.identity.to_numpy(),
                metric,
                config["bootstrap_repeats"],
                config["bootstrap_seed"],
            ),
        }
    domain_evaluation = {}
    for status, group in saved.groupby("bounding_box_status"):
        domain_evaluation[status] = {
            "n": len(group),
            "lightgbm_first_seed": (
                score(metric, group.Y.to_numpy(), group[f"lightgbm_{first_seed}"].to_numpy())
                if metric != "auroc" or group.Y.nunique() == 2
                else None
            ),
        }
    result = {
        "dataset": dataset,
        "metric": metric,
        "partition": phase,
        "development_source_rows": len(development),
        "development_rejected_rows": len(development) - len(valid),
        "development_repeated_identity_rows": int(valid.identity.duplicated().sum()),
        "fit_rows": len(fit),
        "evaluation_source_rows": source_evaluation_rows,
        "evaluated_rows": len(saved),
        "evaluation_seen_identity_rows": int(saved.identity_seen_in_fit.sum()),
        "cohorts": evaluations,
        "bounding_box_by_status": domain_evaluation,
        "seconds_by_model_seed": timings,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "test"), default="development")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    archive_hash = sha256(args.archive)
    if archive_hash != config["archive_sha256"]:
        parser.error(f"TDC archive checksum mismatch ({archive_hash})")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.phase == "test":
        paths = ["src", str(Path(__file__).resolve()), str(args.config.resolve()), "uv.lock"]
        differences = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal", "--", *paths],
            cwd=ROOT,
            text=True,
        )
        if differences:
            parser.error(
                f"Commit the frozen benchmark inputs before test evaluation:\n{differences.rstrip()}"
            )
    output = args.output.resolve()
    output.mkdir(parents=True)
    manifest = {
        "revision": revision,
        "lock_sha256": sha256(ROOT / "uv.lock"),
        "config_sha256": sha256(args.config),
        "archive_sha256": archive_hash,
        "phase": args.phase,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name) for name in ("rdkit", "lightgbm", "scikit-learn", "numpy", "pandas")
        },
        "protocol": config,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for dataset, setting in config["datasets"].items():
        directory = output / dataset
        directory.mkdir()
        summary = evaluate_dataset(args.archive, directory, dataset, setting, config, args.phase)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
