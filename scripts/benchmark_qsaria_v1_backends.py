"""Evaluate Chemprop and a CPU-bounded TabICL comparison on V1 benchmark splits."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import tomllib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from cs_copilot.tools.prediction.backend import PredictionModelRecord, PredictionTaskSpec
from cs_copilot.tools.prediction.chemprop_adapter import (
    normalize_chemprop_classification_predictions,
)
from cs_copilot.tools.prediction.qsar_contracts import TabICLRunRequest
from cs_copilot.tools.prediction.tabicl_backend import TabICLBackend
from scripts.benchmark_qsaria_v1 import (
    ROOT,
    fingerprints,
    predict_lightgbm,
    read_partition,
    score,
    sha256,
    structures,
)
from scripts.benchmark_qsaria_v1_moleculeace import read_official_data

CONFIG = ROOT / "benchmarks/qsaria-v1-backends.toml"
TDC_CONFIG = ROOT / "benchmarks/qsaria-v1.toml"
ACE_CONFIG = ROOT / "benchmarks/qsaria-v1-moleculeace.toml"


def run_chemprop_cli(directory, stage, args):
    cli = Path(sys.executable).with_name("chemprop")
    with (directory / f"{stage}.log").open("w") as handle:
        subprocess.run(
            [str(cli), stage, *args], check=True, stdout=handle, stderr=subprocess.STDOUT
        )


def predict_chemprop(directory, fit, evaluation, task, config, seed):
    training_csv = directory / "fit.csv"
    test_csv = directory / "evaluation.csv"
    fit[["identity", "Y"]].to_csv(training_csv, index=False)
    evaluation[["identity"]].to_csv(test_csv, index=False)
    output_dir = directory / "model"
    common = ["--accelerator", "cpu", "--devices", "1", "--num-workers", "0"]
    options = []
    for name, value in config.items():
        options.extend([f"--{name.replace('_', '-')}", str(value)])
    run_chemprop_cli(
        directory,
        "train",
        [
            "--data-path",
            str(training_csv),
            "--output-dir",
            str(output_dir),
            "--task-type",
            task,
            "--smiles-columns",
            "identity",
            "--target-columns",
            "Y",
            "--split-sizes",
            "1.0",
            "0.0",
            "0.0",
            "--data-seed",
            str(seed),
            "--pytorch-seed",
            str(seed),
            "--remove-checkpoints",
            *common,
            *options,
        ],
    )
    output_csv = directory / "native_predictions.csv"
    run_chemprop_cli(
        directory,
        "predict",
        [
            "--test-path",
            str(test_csv),
            "--preds-path",
            str(output_csv),
            "--model-paths",
            str(output_dir / "model_0/best.pt"),
            "--smiles-columns",
            "identity",
            *common,
        ],
    )
    predictions = pd.read_csv(output_csv)
    if task == "classification":
        predictions = normalize_chemprop_classification_predictions(
            predictions, task=PredictionTaskSpec(task, target_columns=["Y"])
        )
        return predictions.positive_probability.to_numpy()
    return predictions.Y.to_numpy()


def select_context(fit, task, config):
    if len(fit) <= config["context_rows"]:
        return fit
    context, _ = train_test_split(
        fit,
        train_size=config["context_rows"],
        random_state=config["context_seed"],
        stratify=fit.Y if task == "classification" else None,
    )
    return context.sort_index()


def predict_tabicl(
    directory, training_csv, evaluation_csv, columns, task, config, checkpoint, seed
):
    backend = TabICLBackend()
    info = backend.train_model(
        TabICLRunRequest(
            train_csv=str(training_csv),
            output_dir=str(directory / "model"),
            task_type=task,
            target_columns=["Y"],
            feature_columns=list(columns),
            final_refit=True,
            split_sizes=[1.0],
            device="cpu",
            n_jobs=1,
            n_estimators=config["n_estimators"],
            batch_size=1,
            use_amp=False,
            use_fa3=False,
            checkpoint_dir=str(checkpoint.parent),
            checkpoint_version=checkpoint.name,
            random_state=seed,
            applicability_domain_methods=["bounding_box"],
        )
    )
    record = PredictionModelRecord(
        "benchmark",
        "tabicl",
        info["model_path"],
        PredictionTaskSpec(task, target_columns=["Y"]),
        inference_profile={"feature_columns": list(columns), "class_labels": info["class_labels"]},
    )
    csv_path = directory / "tabicl_predictions.csv"
    backend.predict_from_csv(str(evaluation_csv), record, str(csv_path))
    column = "positive_probability" if task == "classification" else "prediction"
    return pd.read_csv(csv_path)[column].to_numpy()


def endpoint_inputs(name, archive, moleculeace, tdc):
    if name in tdc["datasets"]:
        setting = tdc["datasets"][name]
        training = read_partition(archive, name, "train_val")
        evaluation = read_partition(archive, name, "test", ["Drug"])
        return training, evaluation, setting["task"], setting["metric"]
    source = read_official_data(moleculeace / f"{name}.csv")
    training = source.loc[source.split.eq("train"), ["row_id", "Drug", "Y"]]
    evaluation = source.loc[source.split.eq("test"), ["row_id", "Drug"]]
    return training, evaluation, "regression", "rmse"


def metrics(frame, metric, prediction_columns):
    cohorts = {"all_scored": frame}
    if frame.identity_seen_in_fit.any():
        cohorts["identity_unseen"] = frame.loc[~frame.identity_seen_in_fit]
    if "cliff_mol" in frame:
        cohorts["activity_cliff"] = frame.loc[frame.cliff_mol.eq(1)]
    result = {}
    for name, cohort in cohorts.items():
        values = {}
        for column in prediction_columns:
            if metric == "rmse":
                values[column] = float(root_mean_squared_error(cohort.Y, cohort[column]))
            else:
                values[column] = score(metric, cohort.Y, cohort[column])
        result[name] = {"n": len(cohort), "scores": values}
    return result


def run_endpoint(name, directory, args, settings, tdc):
    train, test, task, metric = endpoint_inputs(name, args.archive, args.moleculeace, tdc)
    fit = structures(train).loc[lambda frame: frame.usable]
    evaluation = structures(test).loc[lambda frame: frame.usable]
    tabicl = settings["tabicl"]
    if args.backend == "tabicl":
        fit = select_context(fit, task, tabicl)
        fit_x = fingerprints(fit, tabicl["radius"], tabicl["bits"])
        eval_x = fingerprints(evaluation, tabicl["radius"], tabicl["bits"])
        fit_csv = directory / "fit_features.csv"
        eval_csv = directory / "evaluation_features.csv"
        fit_x.assign(Y=fit.Y.to_numpy()).to_csv(fit_csv, index=False)
        eval_x.to_csv(eval_csv, index=False)
        checkpoint = args.checkpoints / tabicl["checkpoints"][task]["file"]

    fit[["row_id", "identity"]].to_csv(directory / "fit_rows.csv", index=False)
    saved = evaluation[["row_id", "identity"]].copy()
    saved["identity_seen_in_fit"] = saved.identity.isin(fit.identity)
    timings = {}
    prediction_columns = []
    for seed in settings["seeds"]:
        run = directory / str(seed)
        run.mkdir()
        started = time.monotonic()
        if args.backend == "chemprop":
            outputs = {
                "chemprop": predict_chemprop(run, fit, evaluation, task, settings["chemprop"], seed)
            }
        else:
            outputs = {
                "tabicl": predict_tabicl(
                    run, fit_csv, eval_csv, fit_x.columns, task, tabicl, checkpoint, seed
                )
            }
            matched = run / "matched_lightgbm"
            matched.mkdir()
            outputs["matched_lightgbm"] = predict_lightgbm(
                matched, task, fit_csv, eval_csv, fit_x.columns, tabicl["lightgbm_trees"], seed
            )
        timings[seed] = round(time.monotonic() - started, 3)
        for model, values in outputs.items():
            if values.shape != (len(saved),) or not np.isfinite(values).all():
                raise ValueError(f"Invalid predictions for {name}, {model}, seed {seed}.")
            column = f"{model}_{seed}"
            saved[column] = values
            prediction_columns.append(column)
        print(f"{name}, {args.backend}, seed {seed}: {timings[seed]} s", flush=True)

    saved.to_csv(directory / "predictions_unlabeled.csv", index=False)
    if name in tdc["datasets"]:
        labels = read_partition(args.archive, name, "test", ["Y"])
        columns = ["row_id", "Y"]
    else:
        labels = read_official_data(args.moleculeace / f"{name}.csv")
        columns = ["row_id", "Y", "cliff_mol"]
    scored = saved.merge(labels[columns], how="left", on="row_id", validate="one_to_one")
    predictions_csv = directory / "predictions.csv"
    scored.to_csv(predictions_csv, index=False)
    return {
        "task": task,
        "metric": metric,
        "fit_rows": len(fit),
        "official_train_rows": len(train),
        "official_test_rows": len(test),
        "test_rows": len(scored),
        "identity_overlap_rows": int(scored.identity_seen_in_fit.sum()),
        "metrics": metrics(pd.read_csv(predictions_csv), metric, prediction_columns),
        "fit_rows_sha256": sha256(directory / "fit_rows.csv"),
        "predictions_sha256": sha256(predictions_csv),
        "seconds_by_seed": timings,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["chemprop", "tabicl"], required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--moleculeace", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tdc = tomllib.loads(TDC_CONFIG.read_text())
    ace = tomllib.loads(ACE_CONFIG.read_text())
    settings = tomllib.loads(CONFIG.read_text())
    sources = {"TDC ADMET": (args.archive, tdc["archive_sha256"])}
    for name, data in ace["datasets"].items():
        sources[name] = (args.moleculeace / f"{name}.csv", data["sha256"])
    if args.backend == "tabicl":
        if args.checkpoints is None:
            parser.error("--checkpoints is required for TabICL.")
        args.checkpoints = args.checkpoints.resolve()
        for task, checkpoint in settings["tabicl"]["checkpoints"].items():
            sources[f"TabICL {task}"] = (
                args.checkpoints / checkpoint["file"],
                checkpoint["sha256"],
            )
    for name, (path, expected) in sources.items():
        if sha256(path) != expected:
            raise ValueError(f"Unexpected checksum for {name}: {path}")

    output = args.output.resolve()
    output.mkdir(parents=True)
    manifest = {
        "backend": args.backend,
        "settings": settings,
        "sha256": {name: checksum for name, (_, checksum) in sources.items()},
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in (
                CONFIG,
                TDC_CONFIG,
                ACE_CONFIG,
                Path(__file__).resolve(),
                ROOT / "uv.lock",
                ROOT / "scripts/benchmark_qsaria_v1.py",
                ROOT / "scripts/benchmark_qsaria_v1_moleculeace.py",
            )
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            package: version(package)
            for package in (
                "lightgbm",
                "chemprop",
                "tabicl",
                "rdkit",
                "torch",
                "pandas",
                "scikit-learn",
            )
        },
        "optimization": False,
    }
    write_json(output / "manifest.json", manifest)
    print(f"Input manifest SHA-256: {sha256(output / 'manifest.json')}", flush=True)
    results = {}
    with threadpool_limits(limits=1):
        for name in (*tdc["datasets"], *ace["datasets"]):
            directory = output / name
            directory.mkdir()
            results[name] = run_endpoint(name, directory, args, settings, tdc)
            write_json(directory / "scores.json", results[name])
    write_json(output / "results.json", {"manifest": manifest, "datasets": results})


if __name__ == "__main__":
    main()
