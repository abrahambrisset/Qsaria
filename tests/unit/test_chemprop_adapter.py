from __future__ import annotations

import json
import subprocess

import pandas as pd
import pytest

from cs_copilot.tools.prediction.backend import InvalidPredictionInputError, PredictionTaskSpec
from cs_copilot.tools.prediction.chemprop_adapter import materialize_chemprop_inputs
from cs_copilot.tools.prediction.chemprop_backend import ChempropBackend


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
