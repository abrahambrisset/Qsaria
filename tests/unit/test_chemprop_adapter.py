from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from cs_copilot.tools.prediction.backend import (
    InvalidPredictionInputError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from cs_copilot.tools.prediction.chemprop_adapter import (
    materialize_chemprop_inputs,
    normalize_chemprop_classification_predictions,
)
from cs_copilot.tools.prediction.chemprop_backend import (
    DEFAULT_CHEMPROP_FINGERPRINT_FFN_BLOCK_INDEX,
    ChempropBackend,
)
from cs_copilot.tools.prediction.chemprop_toolkit import ChempropToolkit


def _task() -> PredictionTaskSpec:
    return PredictionTaskSpec(
        task_type="regression",
        smiles_columns=["smiles"],
        target_columns=["pEC50"],
    )


def test_chemprop_adapter_writes_minimal_aligned_inputs(tmp_path):
    source = tmp_path / "curated.csv"
    pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "smiles": ["CCO", "CCC", "CCN"],
            "pEC50": ["5.1", "6.2", "4.9"],
            "metadata": ["a", "b", "c"],
        }
    ).to_csv(source, index=False)

    result = materialize_chemprop_inputs(
        source_csv=str(source),
        output_dir=tmp_path / "chemprop_inputs",
        task=_task(),
        split_payload=[{"train": [0], "val": [1], "test": [2]}],
        split_label="random_seed_1",
        seed=1,
    )

    clean = pd.read_csv(result["chemprop_training_input_csv"])
    assert list(clean.columns) == ["smiles", "pEC50"]
    assert result["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert result["row_count"] == 3


def test_chemprop_cli_forwards_observed_epoch_progress(monkeypatch):
    backend = ChempropBackend()
    observed = []
    monkeypatch.setattr(backend, "_ensure_available", lambda: None)
    monkeypatch.setattr(backend, "_find_cli_path", lambda: None)

    backend._run_cli(
        [
            sys.executable,
            "-c",
            "print('Epoch 1/3', flush=True); print('Epoch 2/3', flush=True)",
        ],
        progress_callback=observed.append,
    )

    assert [(item["epoch"], item["total_epochs"]) for item in observed] == [(1, 3), (2, 3)]


def test_chemprop_cli_forwards_observed_ray_tune_candidate_progress(monkeypatch):
    """Ray's own status rows drive the candidate counter, not trial artifacts."""

    backend = ChempropBackend()
    observed = []
    monkeypatch.setattr(backend, "_ensure_available", lambda: None)
    monkeypatch.setattr(backend, "_find_cli_path", lambda: None)

    backend._run_cli(
        [
            sys.executable,
            "-c",
            "print('│ train_func_deadbeef │ TERMINATED │', flush=True); "
            "print('train_func_cafebabe      RUNNING', flush=True); "
            "print('Epoch 2/10', flush=True)",
        ],
        total_trials=25,
        progress_callback=observed.append,
    )

    ray_status = next(item for item in observed if item["event"] == "ray_tune_trial_status")
    assert ray_status == {
        "event": "ray_tune_trial_status",
        "trial_id": "train_func_cafebabe",
        "trial_status": "RUNNING",
        "candidate_index": 2,
        "total_trials": 25,
        "completed_trials": 1,
        "failed_trials": 0,
    }
    assert (observed[-1]["event"], observed[-1]["epoch"], observed[-1]["total_epochs"]) == (
        "epoch",
        2,
        10,
    )


def test_chemprop_adapter_accepts_train_test_without_hidden_validation(tmp_path):
    source = tmp_path / "curated.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCN"],
            "pEC50": [5.1, 6.2, 4.9],
        }
    ).to_csv(source, index=False)

    result = materialize_chemprop_inputs(
        source_csv=str(source),
        output_dir=tmp_path / "chemprop_inputs",
        task=_task(),
        split_payload=[{"train": [0, 1], "test": [2]}],
        split_label="random_80_20",
        seed=1,
    )

    splits = json.loads((tmp_path / "chemprop_inputs" / "chemprop_splits.json").read_text())
    assert splits == [{"train": [0, 1], "test": [2]}]
    assert result["split_counts"] == {"train": 2, "test": 1}


def test_chemprop_adapter_accepts_train_only_final_refit_payload(tmp_path):
    source = tmp_path / "curated.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCN"],
            "pEC50": [5.1, 6.2, 4.9],
        }
    ).to_csv(source, index=False)

    result = materialize_chemprop_inputs(
        source_csv=str(source),
        output_dir=tmp_path / "chemprop_inputs",
        task=_task(),
        split_payload=[{"train": [0, 1, 2]}],
        split_label="final_refit",
        seed=1,
    )

    splits = json.loads((tmp_path / "chemprop_inputs" / "chemprop_splits.json").read_text())
    assert splits == [{"train": [0, 1, 2]}]
    assert result["split_counts"] == {"train": 3}


def test_chemprop_adapter_rejects_bad_training_inputs(tmp_path):
    source = tmp_path / "bad.csv"
    pd.DataFrame({"smiles": ["CCO", ""], "pEC50": [5.0, None]}).to_csv(source, index=False)

    with pytest.raises(InvalidPredictionInputError, match="empty SMILES"):
        materialize_chemprop_inputs(
            source_csv=str(source),
            output_dir=tmp_path / "chemprop_inputs",
            task=_task(),
            split_payload=[{"train": [0], "val": [0], "test": [1]}],
            split_label="bad",
            seed=1,
        )


def test_chemprop_adapter_rejects_missing_or_non_numeric_targets(tmp_path):
    missing = tmp_path / "missing.csv"
    pd.DataFrame({"smiles": ["CCO"]}).to_csv(missing, index=False)
    with pytest.raises(InvalidPredictionInputError, match="missing columns"):
        materialize_chemprop_inputs(
            source_csv=str(missing),
            output_dir=tmp_path / "missing_inputs",
            task=_task(),
            split_payload=[{"train": [0], "val": [0], "test": [0]}],
            split_label="missing",
            seed=1,
        )

    non_numeric = tmp_path / "non_numeric.csv"
    pd.DataFrame({"smiles": ["CCO", "CCC", "CCN"], "pEC50": [5.0, "bad", 4.0]}).to_csv(
        non_numeric,
        index=False,
    )
    with pytest.raises(InvalidPredictionInputError, match="non-numeric"):
        materialize_chemprop_inputs(
            source_csv=str(non_numeric),
            output_dir=tmp_path / "non_numeric_inputs",
            task=_task(),
            split_payload=[{"train": [0], "val": [1], "test": [2]}],
            split_label="non_numeric",
            seed=1,
        )


def test_chemprop_adapter_accepts_native_multiclass_labels(tmp_path):
    source = tmp_path / "multiclass.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCN"],
            "profile": ["CYP2C9", "CYP2D6", "CYP3A4"],
        }
    ).to_csv(source, index=False)
    task = PredictionTaskSpec(
        task_type="multiclass_classification",
        smiles_columns=["smiles"],
        target_columns=["profile"],
    )

    result = materialize_chemprop_inputs(
        source_csv=str(source),
        output_dir=tmp_path / "inputs",
        task=task,
        split_payload=[{"train": [0], "val": [1], "test": [2]}],
        split_label="random",
        seed=7,
    )

    clean = pd.read_csv(result["chemprop_training_input_csv"])
    assert clean["profile"].tolist() == [0, 1, 2]
    metadata = result["classification_targets"]["profile"]
    assert metadata["task_kind"] == "multiclass_classification"
    assert metadata["class_count"] == 3
    assert metadata["class_labels"] == ["CYP2C9", "CYP2D6", "CYP3A4"]
    assert "positive_class_label" not in metadata


def test_chemprop_adapter_rejects_multitarget_multiclass_with_different_class_counts(tmp_path):
    source = tmp_path / "multitarget_multiclass.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCN", "CCCl"],
            "target_a": ["a", "b", "c", "a"],
            "target_b": ["a", "b", "c", "d"],
        }
    ).to_csv(source, index=False)
    task = PredictionTaskSpec(
        task_type="multiclass_classification",
        smiles_columns=["smiles"],
        target_columns=["target_a", "target_b"],
    )

    with pytest.raises(InvalidPredictionInputError, match="same number of classes"):
        materialize_chemprop_inputs(
            source_csv=str(source),
            output_dir=tmp_path / "inputs",
            task=task,
            split_payload=[{"train": [0, 1], "val": [2], "test": [3]}],
            split_label="random",
            seed=7,
        )


def test_chemprop_adapter_accepts_multitarget_multiclass_with_shared_class_count(tmp_path):
    source = tmp_path / "multitarget_multiclass.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCN"],
            "target_a": ["a", "b", "c"],
            "target_b": ["x", "y", "z"],
        }
    ).to_csv(source, index=False)

    result = materialize_chemprop_inputs(
        source_csv=str(source),
        output_dir=tmp_path / "inputs",
        task=PredictionTaskSpec(
            task_type="multiclass_classification",
            smiles_columns=["smiles"],
            target_columns=["target_a", "target_b"],
        ),
        split_payload=[{"train": [0], "val": [1], "test": [2]}],
        split_label="random",
        seed=7,
    )

    assert {metadata["class_count"] for metadata in result["classification_targets"].values()} == {
        3
    }


def test_chemprop_multiclass_predictions_use_canonical_probability_columns():
    task = PredictionTaskSpec(
        task_type="multiclass_classification",
        smiles_columns=["smiles"],
        target_columns=["profile"],
    )
    native = pd.DataFrame(
        {
            "smiles": ["CCO", "CCC"],
            "profile": [2, 0],
            "profile_prob": ["[0.1, 0.2, 0.7]", "[0.8, 0.1, 0.1]"],
        }
    )

    normalized = normalize_chemprop_classification_predictions(
        native,
        task=task,
        classification_targets={
            "profile": {
                "class_labels": ["CYP2C9", "CYP2D6", "CYP3A4"],
                "class_count": 3,
            }
        },
    )

    assert normalized["prediction"].tolist() == ["CYP3A4", "CYP2C9"]
    assert normalized["prediction_class_code"].tolist() == [2, 0]
    assert normalized["probability_cyp2c9"].tolist() == [0.1, 0.8]
    assert normalized["probability_cyp3a4"].tolist() == [0.7, 0.1]
    assert "positive_probability" not in normalized.columns


def test_chemprop_backend_prefers_native_splits_file(monkeypatch, tmp_path):
    train_csv = tmp_path / "chemprop_training_input.csv"
    train_csv.write_text("smiles,pEC50\nCCO,5.0\nCCC,6.0\nCCN,4.0\n")
    splits_file = tmp_path / "chemprop_splits.json"
    splits_file.write_text('[{"train":[0],"val":[1],"test":[2]}]\n')
    captured = {}
    backend = ChempropBackend()

    monkeypatch.setattr(backend, "is_available", lambda: True)

    def fake_run_cli(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(backend, "_run_cli", fake_run_cli)

    backend.train_model(
        train_csv=str(train_csv),
        output_dir=str(tmp_path / "out"),
        task=_task(),
        extra_args={
            "splits_file": str(splits_file),
            "split_type": "random",
            "split_sizes": [0.8, 0.1, 0.1],
            "data_seed": 42,
            "validation_strategy": {"type": "repeated_holdout"},
        },
    )

    args = captured["args"]
    assert "--splits-file" in args
    assert str(splits_file) in args
    assert "--split-type" not in args
    assert "--split-sizes" not in args
    assert "--data-seed" not in args
    assert "--validation-strategy" not in args


def test_chemprop_hpopt_uses_an_isolated_local_ray_runtime(monkeypatch, tmp_path):
    train_csv = tmp_path / "chemprop_training_input.csv"
    train_csv.write_text("smiles,pEC50\nCCO,5.0\nCCC,6.0\nCCN,4.0\n")
    captured = {}
    backend = ChempropBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)

    def fake_run_cli(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["working_dir_contents"] = list(kwargs["cwd"].iterdir())
        (kwargs["output_dir"] / "best_config.toml").write_text("depth = [4]\n")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(backend, "_run_cli", fake_run_cli)
    output_dir = tmp_path / "hpopt_output"

    result = backend.hpopt_model(
        train_csv=str(train_csv),
        output_dir=str(output_dir),
        task=_task(),
        extra_args={
            "raytune_num_samples": 25,
            "epochs": 30,
            "raytune_use_gpu": True,
            "raytune_num_gpus": 1,
            "accelerator": "gpu",
            "devices": 1,
        },
    )

    args = captured["args"]
    assert args[args.index("--raytune-num-samples") + 1] == "25"
    assert "--raytune-use-gpu" in args
    assert args[args.index("--raytune-num-gpus") + 1] == "1"
    assert args[args.index("--accelerator") + 1] == "gpu"
    assert args[args.index("--devices") + 1] == "1"
    assert Path(args[args.index("--output-dir") + 1]) == output_dir
    ray_runtime_dir = Path(args[args.index("--raytune-temp-dir") + 1])
    ray_temp_root = ray_runtime_dir.parent
    assert ray_runtime_dir.name == "ray"
    assert ray_temp_root.parent == Path(tempfile.gettempdir())
    assert ray_temp_root.name.startswith("r-")
    assert captured["kwargs"]["env_overrides"]["RAY_ADDRESS"] == "local"
    assert captured["kwargs"]["cwd"] == ray_temp_root / "work"
    assert captured["working_dir_contents"] == []
    assert captured["kwargs"]["total_epochs"] == 30
    assert captured["kwargs"]["total_trials"] == 25
    assert result["best_config_path"] == str(output_dir / "best_config.toml")


def test_chemprop_backend_uses_native_multiclass_cli(monkeypatch, tmp_path):
    train_csv = tmp_path / "multiclass.csv"
    train_csv.write_text("smiles,profile\nCCO,0\nCCC,1\nCCN,2\n")
    captured = {}
    backend = ChempropBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)

    def fake_run_cli(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(backend, "_run_cli", fake_run_cli)
    backend.train_model(
        train_csv=str(train_csv),
        output_dir=str(tmp_path / "out"),
        task=PredictionTaskSpec(
            task_type="multiclass_classification",
            smiles_columns=["smiles"],
            target_columns=["profile"],
        ),
        extra_args={"multiclass_num_classes": 3},
    )

    args = captured["args"]
    assert args[args.index("--task-type") + 1] == "multiclass"
    assert args[args.index("--multiclass-num-classes") + 1] == "3"


def test_chemprop_backend_normalizes_multiclass_prediction_output(monkeypatch, tmp_path):
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("smiles\nCCO\nCCC\n")
    model_path = tmp_path / "best.pt"
    model_path.write_text("mock")
    preds_path = tmp_path / "predictions.csv"
    backend = ChempropBackend()

    def fake_run_cli(args, **kwargs):
        preds_path.write_text(
            'smiles,profile,profile_prob\nCCO,2,"[0.1, 0.2, 0.7]"\n' 'CCC,0,"[0.8, 0.1, 0.1]"\n'
        )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(backend, "_run_cli", fake_run_cli)
    record = PredictionModelRecord(
        model_id="multiclass",
        backend_name="chemprop",
        model_path=str(model_path),
        task=PredictionTaskSpec(
            task_type="multiclass_classification",
            smiles_columns=["smiles"],
            target_columns=["profile"],
        ),
        inference_profile={
            "classification_targets": {
                "profile": {
                    "class_labels": ["CYP2C9", "CYP2D6", "CYP3A4"],
                    "class_count": 3,
                }
            }
        },
    )

    result = backend.predict_from_csv(
        input_csv=str(input_csv),
        model_record=record,
        preds_path=str(preds_path),
    )

    predictions = pd.read_csv(preds_path)
    assert result["prediction_format"] == "qsaria_classification_canonical"
    assert predictions["prediction"].tolist() == ["CYP3A4", "CYP2C9"]
    assert predictions["probability_cyp2d6"].tolist() == [0.2, 0.1]


def test_chemprop_multiclass_replicates_average_probabilities_before_argmax(tmp_path):
    train_csv = tmp_path / "training.csv"
    train_csv.write_text("smiles,profile\nCCO,0\nCCC,1\nCCN,2\n")
    output_dir = tmp_path / "run"
    inputs_dir = output_dir / "chemprop_inputs"
    inputs_dir.mkdir(parents=True)
    splits_file = inputs_dir / "chemprop_splits.json"
    splits_file.write_text('[{"train":[0],"val":[1],"test":[2]}]\n')
    (inputs_dir / "chemprop_input_manifest.json").write_text(
        json.dumps(
            {
                "classification_targets": {
                    "profile": {
                        "class_labels": ["A", "B", "C"],
                        "class_count": 3,
                    }
                }
            }
        )
        + "\n"
    )
    replicate_predictions = (
        (0, "[0.6, 0.3, 0.1]"),
        (2, "[0.1, 0.2, 0.7]"),
    )
    for replicate_index, (predicted_class, probabilities) in enumerate(replicate_predictions):
        replicate_dir = output_dir / f"replicate_{replicate_index}" / "model_0"
        replicate_dir.mkdir(parents=True)
        (replicate_dir / "test_predictions.csv").write_text(
            f'smiles,profile,profile_prob\nCCN,{predicted_class},"{probabilities}"\n'
        )

    result = ChempropToolkit()._write_normalized_test_predictions(
        train_csv=str(train_csv),
        output_dir=output_dir,
        task=PredictionTaskSpec(
            task_type="multiclass_classification",
            smiles_columns=["smiles"],
            target_columns=["profile"],
        ),
        splits_file=str(splits_file),
    )

    predictions = pd.read_csv(result["test_predictions_path"])
    assert predictions["prediction"].tolist() == ["C"]
    assert predictions["prediction_class_code"].tolist() == [2]
    assert predictions["probability_a"].tolist() == pytest.approx([0.35])
    assert predictions["probability_c"].tolist() == pytest.approx([0.4])
    assert result["prediction_aggregation"] == "mean_aligned_replicates"


def test_chemprop_fingerprint_uses_official_default_ffn_block(monkeypatch, tmp_path):
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("smiles\nCCO\n")
    model_path = tmp_path / "best.pt"
    model_path.write_text("mock")
    output_csv = tmp_path / "fingerprints.csv"
    captured = {}
    backend = ChempropBackend()

    monkeypatch.setattr(backend, "is_available", lambda: True)

    def fake_run_cli(args, **kwargs):
        captured["args"] = args
        output_csv.with_stem(f"{output_csv.stem}_0").write_text("fp_0,fp_1\n0.1,0.2\n")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(backend, "_run_cli", fake_run_cli)

    result = backend.fingerprint_from_csv(
        input_csv=str(input_csv),
        model_path=str(model_path),
        output_csv=str(output_csv),
    )

    args = captured["args"]
    assert DEFAULT_CHEMPROP_FINGERPRINT_FFN_BLOCK_INDEX == -1
    assert args[args.index("--ffn-block-index") + 1] == "-1"
    assert result["feature_columns"] == ["chemprop_fp_0", "chemprop_fp_1"]
