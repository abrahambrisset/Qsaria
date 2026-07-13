import json
import os
import subprocess

import pandas as pd
import pytest

from cs_copilot.tools.prediction.backend import PredictionTaskSpec
from cs_copilot.tools.prediction.chemprop_adapter import materialize_chemprop_inputs
from cs_copilot.tools.prediction.chemprop_backend import ChempropBackend
from cs_copilot.tools.prediction.chemprop_toolkit import ChempropToolkit
from cs_copilot.tools.prediction.hyperparameter_tuning import (
    HyperparameterTuningError,
    LightGBMOptunaAdapter,
    _running_mean,
    build_tuning_progress_plot,
    describe_backend_hyperparameters,
    normalize_tuning_config,
)
from cs_copilot.tools.prediction.lightgbm_toolkit import _ad_status_series
from cs_copilot.tools.prediction.qsar_training_policy import resolve_validation_protocol
from cs_copilot.tools.prediction.tabular_representations import (
    default_tabular_representation_for_protocol,
)


def _chemprop_task() -> PredictionTaskSpec:
    return PredictionTaskSpec(
        task_type="regression",
        smiles_columns=["smiles"],
        target_columns=["target"],
    )


def test_backend_hyperparameter_contracts_expose_supported_engines_and_parameters():
    lightgbm = describe_backend_hyperparameters("lightgbm")
    chemprop = describe_backend_hyperparameters("chemprop")
    tabicl = describe_backend_hyperparameters("tabicl")

    assert lightgbm["default_engine"] == "optuna_tpe"
    assert {
        item["name"] for item in lightgbm["parameters"] if item["default_tuning"]
    } == {"n_estimators", "learning_rate", "max_depth", "num_leaves"}
    assert chemprop["default_engine"] == "chemprop_hpopt_hyperopt"
    assert {
        item["name"] for item in chemprop["parameters"] if item["default_tuning"]
    } == {
        "depth",
        "message_hidden_dim",
        "ffn_hidden_dim",
        "ffn_num_layers",
        "dropout",
    }
    assert tabicl["supports_hyperparameter_tuning"] is False


def test_standard_protocol_is_one_fixed_random_holdout_and_rdkit_is_tabular_default():
    protocol = resolve_validation_protocol(
        requested_protocol=None,
        training_profile="local_light",
        base_seed=123,
    )

    assert protocol["protocol"] == "standard_qsar"
    assert len(protocol["split_runs"]) == 1
    assert protocol["split_runs"][0]["backend_split_type"] == "random"
    assert protocol["split_runs"][0]["split_sizes"] == [0.8, 0.1, 0.1]
    assert default_tabular_representation_for_protocol("standard_qsar") == "rdkit_all"


def test_lightgbm_direct_value_is_excluded_from_the_search():
    config = normalize_tuning_config(
        {"n_trials": 4, "parameters": ["n_estimators", "learning_rate", "max_depth"]},
        backend_name="lightgbm",
        task_type="regression",
        eligible=True,
        fixed_parameters={"learning_rate"},
    )

    assert config is not None
    assert config.parameters == ("n_estimators", "max_depth")
    assert config.objective.metric == "rmse"
    assert config.objective.subset == "in_domain"


def test_lightgbm_tuning_extracts_ad_status_from_dataframe_without_boolean_coercion():
    statuses = _ad_status_series(
        {"scores": pd.DataFrame({"ad_status": ["in_domain", "out_of_domain"]})}
    )

    assert statuses is not None
    assert statuses.tolist() == ["in_domain", "out_of_domain"]
    assert _ad_status_series({"scores": None}) is None


def test_tuning_progress_plot_persists_trial_metrics_and_incumbent(tmp_path):
    summary = {
        "engine": "optuna_tpe",
        "objective": {
            "metric": "rmse",
            "subset": "in_domain",
            "direction": "minimize",
        },
        "trials": [
            {
                "number": 0,
                "state": "complete",
                "objective": 1.3,
                "metrics": {"all": {"r2": 0.4, "rmse": 1.5, "mae": 1.1}, "in_domain": {"rmse": 1.3}},
            },
            {
                "number": 1,
                "state": "complete",
                "objective": 1.0,
                "metrics": {"all": {"r2": 0.6, "rmse": 1.2, "mae": 0.9}, "in_domain": {"rmse": 1.0}},
            },
        ],
    }

    plot_path = build_tuning_progress_plot(summary, output_dir=tmp_path / "plots")

    assert plot_path is not None
    assert plot_path.endswith("hyperparameter_tuning_progress.png")
    assert (tmp_path / "plots" / "hyperparameter_tuning_progress.png").exists()


def test_running_mean_preserves_trial_alignment_and_ignores_missing_scores():
    assert _running_mean([1.0, None, 3.0, 2.0]) == [1.0, 1.0, 2.0, 2.0]


def test_invalid_tuning_requests_fail_before_training():
    with pytest.raises(HyperparameterTuningError, match="does not support"):
        normalize_tuning_config(
            {"n_trials": 20},
            backend_name="tabicl",
            task_type="regression",
            eligible=True,
        )
    with pytest.raises(HyperparameterTuningError, match="requires a single holdout"):
        normalize_tuning_config(
            {"n_trials": 20},
            backend_name="lightgbm",
            task_type="regression",
            eligible=False,
        )
    with pytest.raises(HyperparameterTuningError, match="Unknown Qsaria hyperparameter"):
        normalize_tuning_config(
            {"parameters": ["unknown_parameter"]},
            backend_name="lightgbm",
            task_type="regression",
            eligible=True,
        )
    with pytest.raises(HyperparameterTuningError, match="objective.subset must be all"):
        normalize_tuning_config(
            {"objective": {"subset": "in_domain"}},
            backend_name="chemprop",
            task_type="regression",
            eligible=True,
        )


def test_optuna_adapter_uses_full_trials_and_respects_depth_leaf_constraint():
    config = normalize_tuning_config(
        {"n_trials": 4, "seed": 17},
        backend_name="lightgbm",
        task_type="regression",
        eligible=True,
    )
    assert config is not None
    observed = []

    def evaluate(parameters):
        observed.append(dict(parameters))
        return {
            "objective": float(parameters["n_estimators"]),
            "metrics": {"in_domain": {"rmse": float(parameters["n_estimators"])}},
            "diagnostics": {},
        }

    summary = LightGBMOptunaAdapter().run(
        config=config,
        fixed_parameters={},
        evaluate=evaluate,
    )

    assert summary.requested_trials == 4
    assert summary.completed_trials == 4
    assert summary.failed_trials == 0
    assert len(observed) == 4
    assert all(item["num_leaves"] <= 2 ** item["max_depth"] for item in observed)


def test_chemprop_tuning_materialization_has_no_test_rows(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("smiles,target\nCCO,1.0\nCCN,2.0\n")

    materialized = materialize_chemprop_inputs(
        source_csv=str(source),
        output_dir=tmp_path / "inputs",
        task=_chemprop_task(),
        split_payload=[{"train": [0], "val": [1], "test": []}],
        split_label="hyperparameter_selection",
        seed=7,
        allow_empty_test=True,
    )

    assert materialized["row_count"] == 2
    assert materialized["split_counts"] == {"train": 1, "val": 1, "test": 0}


def test_chemprop_cluster_holdout_builds_transient_molecular_descriptors(tmp_path):
    molecules = [
        "CCO",
        "CCN",
        "CCC",
        "CCCl",
        "CCBr",
        "CCS",
        "CCCO",
        "CCCN",
        "CCCF",
        "CCCC",
        "CCOC",
        "CCNC",
    ]
    source = tmp_path / "cluster.csv"
    source.write_text(
        "smiles,target\n"
        + "\n".join(f"{smiles},{index / 10.0}" for index, smiles in enumerate(molecules))
        + "\n"
    )
    toolkit = ChempropToolkit(register_tools=False)

    split = toolkit._build_split_payload(
        train_csv=str(source),
        task=_chemprop_task(),
        split_type="kmeans",
        split_sizes=[0.8, 0.1, 0.1],
        seed=3,
    )[0]

    assert {"train", "val", "test"}.issubset(split)
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == len(molecules)


def test_chemprop_backend_keeps_supported_architecture_and_schedule_arguments(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("smiles,target\nCCO,1.0\nCCN,2.0\nCCC,3.0\n")
    backend = ChempropBackend()
    captured = {}
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(
        backend,
        "_run_cli",
        lambda args, **kwargs: captured.setdefault(
            "completed", subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        ),
    )

    backend.train_model(
        train_csv=str(source),
        output_dir=str(tmp_path / "output"),
        task=_chemprop_task(),
        extra_args={
            "depth": 4,
            "dropout": 0.1,
            "init_lr": 0.0001,
            "max_lr": 0.001,
            "final_lr": 0.00001,
            "warmup_epochs": 2,
        },
    )

    args = captured["completed"].args
    for flag in (
        "--depth",
        "--dropout",
        "--init-lr",
        "--max-lr",
        "--final-lr",
        "--warmup-epochs",
    ):
        assert flag in args


def test_chemprop_hpopt_receives_train_validation_only_and_keeps_only_summary(tmp_path):
    class FakeChempropBackend:
        def __init__(self):
            self.train_rows = None
            self.splits = None

        def hpopt_model(self, *, train_csv, output_dir, task, extra_args):
            self.train_rows = pd.read_csv(train_csv)
            self.splits = json.loads(open(extra_args["splits_file"]).read())
            assert extra_args["raytune_num_samples"] == 3
            assert extra_args["raytune_search_algorithm"] == "hyperopt"
            assert extra_args["raytune_trial_scheduler"] == "FIFO"
            assert extra_args["tracking_metric"] == "val_loss"
            assert extra_args["search_parameter_keywords"] == ["dropout"]
            return {"best_params": {"dropout": 0.1}}

    backend = FakeChempropBackend()
    toolkit = ChempropToolkit(backend=backend, register_tools=False)
    config = normalize_tuning_config(
        {"n_trials": 3, "parameters": ["depth", "dropout"]},
        backend_name="chemprop",
        task_type="regression",
        eligible=True,
        fixed_parameters={"depth"},
    )
    assert config is not None
    source = tmp_path / "source.csv"
    source.write_text("smiles,target\nCCO,1.0\nCCN,2.0\nCCC,3.0\n")

    summary = toolkit._run_chemprop_hpopt(
        source_df=pd.read_csv(source),
        task=_chemprop_task(),
        split_payload=[{"train": [0], "val": [1], "test": [2]}],
        config=config,
        fixed_parameters={"depth": 4},
        train_args={},
        output_dir=tmp_path / "final_run",
        seed=9,
    )

    assert len(backend.train_rows) == 2
    assert backend.splits == [{"train": [0], "val": [1], "test": []}]
    assert summary["best_trial"]["params"] == {"depth": 4, "dropout": 0.1}
    assert (tmp_path / "final_run" / "hyperparameter_tuning_summary.json").exists()


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("QSARIA_CHEMPROP_HPO_LIVE") != "1",
    reason="Set QSARIA_CHEMPROP_HPO_LIVE=1 to run the native Chemprop/Ray smoke test.",
)
def test_chemprop_hpopt_live_smoke(tmp_path):
    """Exercise Chemprop 2.2.x native hpopt without ever exposing the test rows."""
    import importlib.metadata

    version = importlib.metadata.version("chemprop")
    assert version.startswith("2.2.")
    backend = ChempropBackend()
    if not backend.is_available():
        pytest.skip("Chemprop CLI is not available in this environment.")

    molecules = [
        "CCO",
        "CCN",
        "CCC",
        "CCCl",
        "CCBr",
        "CCS",
        "CCCO",
        "CCCN",
        "CCCF",
        "CCCC",
        "CCOC",
        "CCNC",
    ]
    source = tmp_path / "live.csv"
    source.write_text(
        "smiles,target\n"
        + "\n".join(f"{smiles},{index / 10.0}" for index, smiles in enumerate(molecules))
        + "\n"
    )
    toolkit = ChempropToolkit(backend=backend, register_tools=False)
    config = normalize_tuning_config(
        {"n_trials": 1},
        backend_name="chemprop",
        task_type="regression",
        eligible=True,
    )
    assert config is not None

    summary = toolkit._run_chemprop_hpopt(
        source_df=pd.read_csv(source),
        task=_chemprop_task(),
        split_payload=[{"train": list(range(8)), "val": [8, 9], "test": [10, 11]}],
        config=config,
        fixed_parameters={},
        train_args={"epochs": 1, "num_workers": 0},
        output_dir=tmp_path / "final_run",
        seed=13,
    )

    assert summary["status"] == "completed"
    assert summary["selection_protocol"]["test_rows_provided_to_hpopt"] == 0
