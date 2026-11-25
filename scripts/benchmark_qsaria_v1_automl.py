#!/usr/bin/env python3
"""Compare matched Qsaria LightGBM and classical model selection on TDC ADMET."""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import time
import tomllib
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from benchmark_qsaria_v1 import (
    ROOT,
    development_indices,
    fingerprints,
    paired_delta,
    read_partition,
    score,
    sha256,
    structures,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from threadpoolctl import threadpool_limits

from cs_copilot.tools.prediction.backend import PredictionModelRecord, PredictionTaskSpec
from cs_copilot.tools.prediction.leakage import audit_molecular_overlap
from cs_copilot.tools.prediction.lightgbm_backend import LightGBMBackend
from cs_copilot.tools.prediction.qsar_contracts import LightGBMRunRequest

QSARIA = "qsaria_lightgbm"
CLASSICAL = "classical_automl"
ARMS = (QSARIA, CLASSICAL)
ENSEMBLES = {
    "regression": {"random_forest": RandomForestRegressor, "extra_trees": ExtraTreesRegressor},
    "classification": {
        "random_forest": RandomForestClassifier,
        "extra_trees": ExtraTreesClassifier,
    },
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def protocol_checksums(config_path, archive_sha256):
    return {
        "archive_sha256": archive_sha256,
        "config_sha256": sha256(config_path),
        "lock_sha256": sha256(ROOT / "uv.lock"),
        "source_sha256": {
            path: sha256(ROOT / path)
            for path in (
                "scripts/benchmark_qsaria_v1.py",
                "scripts/benchmark_qsaria_v1_automl.py",
                "src/cs_copilot/tools/chemistry/standardize.py",
                "src/cs_copilot/tools/prediction/leakage.py",
                "src/cs_copilot/tools/prediction/lightgbm_backend.py",
                "src/cs_copilot/tools/prediction/qsar_contracts.py",
            )
        },
    }


def development_rows(archive, dataset):
    frame = structures(read_partition(archive, dataset, "train_val"))
    return frame.loc[frame.usable]


def fit_predict(directory, arm, parameters, setting, config, seed, fit_x, fit_y, evaluation_x):
    task = setting["task"]
    classification = task == "classification"
    budget = {
        "n_estimators": config["trees"],
        "n_jobs": config["workers"],
        "random_state": seed,
    }
    started = time.perf_counter()
    if arm == QSARIA:
        backend = LightGBMBackend()
        trained = backend.train_model(
            LightGBMRunRequest(
                train_csv=str(directory / "fit_features.csv"),
                output_dir=str(directory / "model"),
                task_type=task,
                target_columns=["Y"],
                feature_columns=list(fit_x.columns),
                final_refit=True,
                split_sizes=[1.0],
                device_type="cpu",
                deterministic=True,
                force_col_wise=True,
                early_stopping_rounds=0,
                **parameters,
                **budget,
            )
        )
        record = PredictionModelRecord(
            "automl",
            "lightgbm",
            trained["model_path"],
            PredictionTaskSpec(task_type=task, target_columns=["Y"]),
        )
        prediction_path = directory / "qsaria_predictions.csv"
        backend.predict_from_csv(
            str(directory / "evaluation_features.csv"), record, str(prediction_path)
        )
        column = "positive_probability" if classification else "prediction"
        predictions = pd.read_csv(prediction_path)[column].to_numpy()
        seconds = time.perf_counter() - started
        with Path(trained["model_path"]).open("rb") as handle:
            actual_trees = pickle.load(handle)["model"].n_estimators_
    else:
        model_type = ENSEMBLES[task][parameters["family"]]
        model = model_type(max_features=parameters["max_features"], **budget)
        model.fit(fit_x, fit_y)
        predictions = (
            model.predict_proba(evaluation_x)[:, 1]
            if classification
            else model.predict(evaluation_x)
        )
        seconds = time.perf_counter() - started
        actual_trees = len(model.estimators_)
    return predictions, {
        "fit_predict_seconds": round(seconds, 3),
        "actual_trees": int(actual_trees),
    }


def feature_matrices(directory, fit, evaluation, config):
    fit_x = fingerprints(fit, config["radius"], config["bits"])
    evaluation_x = fingerprints(evaluation, config["radius"], config["bits"])
    fit_x.assign(Y=fit.Y.to_numpy()).to_csv(directory / "fit_features.csv", index=False)
    evaluation_x.to_csv(directory / "evaluation_features.csv", index=False)
    return fit_x, evaluation_x


def select_recipes(args, config, checksums):
    winners = {}
    development = {}
    for dataset, setting in config["datasets"].items():
        directory = args.output / dataset
        directory.mkdir()
        valid = development_rows(args.archive, dataset)
        train, validation = development_indices(valid, setting["task"], config["split_seed"])
        fit, held_out = valid.iloc[train], valid.iloc[validation]
        fit_x, held_out_x = feature_matrices(directory, fit, held_out, config)
        metric = setting["metric"]
        direction = 1 if metric == "mae" else -1
        winners[dataset] = {}
        attempts = {}
        validation_predictions = held_out[["row_id", "Drug_ID", "identity", "Y"]].copy()
        for arm in ARMS:
            attempts[arm] = {}
            for name, parameters in config["candidates"][arm].items():
                predictions, timings = fit_predict(
                    directory,
                    arm,
                    parameters,
                    setting,
                    config,
                    config["selection_seed"],
                    fit_x,
                    fit.Y.to_numpy(),
                    held_out_x,
                )
                validation_predictions[f"{arm}_{name}"] = predictions
                attempts[arm][name] = {
                    **timings,
                    "validation_score": score(metric, held_out.Y.to_numpy(), predictions),
                }
            winners[dataset][arm] = min(
                attempts[arm], key=lambda name: direction * attempts[arm][name]["validation_score"]
            )
        validation_predictions.to_csv(directory / "validation_predictions.csv", index=False)
        development[dataset] = {
            "metric": metric,
            "fit_rows": len(fit),
            "validation_rows": len(held_out),
            "overlap_audit": audit_molecular_overlap(fit.Drug, held_out.Drug),
            "candidates": attempts,
        }
        print(f"{dataset} development winners: {winners[dataset]}", flush=True)
    selection = args.output / "selection.json"
    write_json(selection, {"checksums": checksums, "winners": winners})
    selection_sha256 = sha256(selection)
    write_json(
        args.output / "development.json",
        {"selection_sha256": selection_sha256, "datasets": development},
    )
    print(f"Selection SHA-256: {selection_sha256}")


def evaluate_dataset(directory, archive, dataset, setting, config, winners):
    fit = development_rows(archive, dataset)
    source = structures(read_partition(archive, dataset, "test", ["Drug_ID", "Drug"]))
    evaluation = source.loc[source.usable]
    fit_x, evaluation_x = feature_matrices(directory, fit, evaluation, config)
    saved = evaluation[["row_id", "Drug_ID", "identity"]].copy()
    saved["identity_seen_in_fit"] = saved.identity.isin(fit.identity)
    overlap_audit = audit_molecular_overlap(fit.Drug, evaluation.Drug)
    timings = {}
    for seed in config["test_seeds"]:
        timings[str(seed)] = {}
        for arm in ARMS:
            parameters = config["candidates"][arm][winners[arm]]
            predictions, run_timing = fit_predict(
                directory,
                arm,
                parameters,
                setting,
                config,
                seed,
                fit_x,
                fit.Y.to_numpy(),
                evaluation_x,
            )
            saved[f"{arm}_{seed}"] = predictions
            timings[str(seed)][arm] = run_timing
    unlabeled_path = directory / "predictions_unlabeled.csv"
    saved.to_csv(unlabeled_path, index=False)
    saved = pd.read_csv(unlabeled_path)
    labels = read_partition(archive, dataset, "test", ["Y"]).set_index("row_id")
    saved["Y"] = saved.row_id.map(labels.Y)
    saved.to_csv(directory / "predictions.csv", index=False)

    metric = setting["metric"]
    cohorts = {}
    for name, cohort in {
        "all_scored": saved,
        "identity_unseen": saved.loc[~saved.identity_seen_in_fit],
    }.items():
        truth = cohort.Y.to_numpy()
        seed_results = {}
        for seed in config["test_seeds"]:
            model = cohort[f"{QSARIA}_{seed}"].to_numpy()
            reference = cohort[f"{CLASSICAL}_{seed}"].to_numpy()
            seed_results[str(seed)] = {
                QSARIA: score(metric, truth, model),
                CLASSICAL: score(metric, truth, reference),
                "qsaria_minus_classical": paired_delta(
                    truth,
                    model,
                    reference,
                    cohort.identity.to_numpy(),
                    metric,
                    config["bootstrap_repeats"],
                    config["bootstrap_seed"],
                ),
            }
        cohorts[name] = {"n": len(cohort), "by_seed": seed_results}
    return {
        "metric": metric,
        "fit_rows": len(fit),
        "test_source_rows": len(source),
        "test_rejected_rows": len(source) - len(evaluation),
        "test_identity_seen_rows": int(saved.identity_seen_in_fit.sum()),
        "overlap_audit": overlap_audit,
        "cohorts": cohorts,
        "refit_predict_by_seed": timings,
        "unlabeled_predictions_sha256": sha256(unlabeled_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("select").add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--selection", type=Path, required=True)
    evaluate.add_argument("--selection-sha256", required=True)
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text())
    counts = [len(config["candidates"][arm]) for arm in ARMS]
    if not counts[0] or len(set(counts)) != 1 or config["workers"] != 1:
        parser.error("The protocol requires equal nonempty candidate counts and one CPU worker.")
    archive_sha256 = sha256(args.archive)
    if archive_sha256 != config["archive_sha256"]:
        parser.error(f"TDC archive checksum mismatch ({archive_sha256}).")
    checksums = protocol_checksums(args.config, archive_sha256)
    if args.phase == "evaluate":
        selection_sha256 = sha256(args.selection)
        if selection_sha256 != args.selection_sha256:
            parser.error(f"Selection checksum mismatch ({selection_sha256}).")
        selection = json.loads(args.selection.read_text())
        if selection["checksums"] != checksums:
            parser.error("The selection and evaluation inputs have different checksums.")
        winners = selection["winners"]

    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    with threadpool_limits(limits=1):
        if args.phase == "select":
            select_recipes(args, config, checksums)
            return
        datasets = {}
        for dataset, setting in config["datasets"].items():
            directory = args.output / dataset
            directory.mkdir()
            datasets[dataset] = evaluate_dataset(
                directory, args.archive, dataset, setting, config, winners[dataset]
            )
            print(f"{dataset} test results saved", flush=True)
    write_json(
        args.output / "evaluation.json",
        {
            "selection_sha256": selection_sha256,
            "selection": selection,
            "protocol": config,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in (
                    "rdkit",
                    "lightgbm",
                    "scikit-learn",
                    "numpy",
                    "pandas",
                    "threadpoolctl",
                )
            },
            "datasets": datasets,
        },
    )


if __name__ == "__main__":
    main()
