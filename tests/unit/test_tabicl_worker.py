from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cs_copilot.tools.prediction.backend import PredictionExecutionError
from cs_copilot.tools.prediction.tabicl_toolkit import TabICLToolkit


def _fake_agent() -> SimpleNamespace:
    return SimpleNamespace(
        session_state={
            "prediction_models": {
                "registered": {},
                "last_prediction": {},
                "prediction_history": [],
                "catalog_recommendations": {},
                "training_runs": [],
                "active_training_run": None,
            }
        }
    )


def test_write_worker_job_creates_clean_payload(tmp_path):
    toolkit = TabICLToolkit()
    job_dir = tmp_path / "worker_job"
    job_dir.mkdir()
    (job_dir / "result.json").write_text("{}")
    payload = {"train_csv": "dataset.csv", "target_columns": ["Y"]}

    job_path = toolkit._write_worker_job(job_dir=job_dir, payload=payload)

    assert job_path.exists()
    assert json.loads(job_path.read_text()) == payload
    assert not (job_dir / "result.json").exists()


def test_run_training_worker_reads_result_json(tmp_path, monkeypatch):
    toolkit = TabICLToolkit()
    job_dir = tmp_path / "worker_job"
    job_dir.mkdir()
    job_path = job_dir / "job.json"
    job_path.write_text(json.dumps({"train_csv": "dataset.csv"}))
    result_path = job_dir / "result.json"
    marker_path = tmp_path / ".training_in_progress"
    expected = {"model_path": "model.pkl", "split_results": []}

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            marker_path.write_text(json.dumps({"progress_message": "TabICL training progress: run 1/2 - random"}))
            result_path.write_text(json.dumps(expected))

        def poll(self):
            return 0

        @property
        def pid(self):
            return 12345

    monkeypatch.setattr("cs_copilot.tools.prediction.tabicl_toolkit.subprocess.Popen", FakeProcess)

    result = toolkit._run_training_worker(
        job_path=job_path,
        worker_log_path=job_dir / "worker.log",
        active_marker_path=marker_path,
    )

    assert result["model_path"] == expected["model_path"]
    assert "worker_duration_seconds" in result


def test_train_tabicl_model_wraps_worker_errors(tmp_path, monkeypatch):
    toolkit = TabICLToolkit()
    agent = _fake_agent()
    output_dir = tmp_path / "training_output"

    monkeypatch.setattr(
        toolkit,
        "_run_training_worker",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("worker exploded")),
    )

    with pytest.raises(PredictionExecutionError, match="worker exploded"):
        toolkit.train_tabicl_model(
            train_csv=str(tmp_path / "dataset.csv"),
            task_type="regression",
            output_dir=str(output_dir),
            target_columns=["Y"],
            validation_protocol="standard_qsar",
            agent=agent,
        )

    assert agent.session_state["prediction_models"]["active_training_run"] is None


def test_train_tabicl_model_syncs_training_runs_from_worker_result(tmp_path, monkeypatch):
    toolkit = TabICLToolkit()
    agent = _fake_agent()
    output_dir = tmp_path / "training_output"

    monkeypatch.setattr(
        toolkit,
        "_apply_training_profile",
        lambda extra_args: {
            "compute_environment": {"execution_env": "apptainer_local"},
            "training_profile": "heavy_validation",
            "profile_reason": "test",
            "validation_protocol": "standard_qsar",
            "extra_args": dict(extra_args or {}),
        },
    )
    monkeypatch.setattr(
        toolkit,
        "_resolve_validation_protocol",
        lambda **kwargs: {
            "protocol": "standard_qsar",
            "reason": "test",
            "seed_policy": {
                "mode": "user_provided_or_replay",
                "model_seed": 42,
                "split_runs": [
                    {"label": "random_seed_42", "seed": 42, "backend_split_type": "random", "primary": True},
                    {"label": "scaffold", "seed": 43, "backend_split_type": "scaffold_balanced", "primary": False},
                ],
            },
            "split_runs": [
                {"label": "random_seed_42", "seed": 42, "backend_split_type": "random"},
                {"label": "scaffold", "seed": 43, "backend_split_type": "scaffold_balanced"},
            ],
        },
    )
    captured_job_payload = {}

    def fake_run_training_worker(**kwargs):
        captured_job_payload.update(json.loads(Path(kwargs["job_path"]).read_text()))
        return {
            "model_path": str(output_dir / "model_0" / "best.pkl"),
            "summary_path": str(output_dir / "cs_copilot_training_summary.json"),
            "split_results": [
                {
                    "strategy_label": "random",
                    "strategy": "random",
                    "strategy_family": "random",
                    "output_dir": str(output_dir / "random_split"),
                    "seed": 42,
                },
                {
                    "strategy_label": "scaffold",
                    "strategy": "scaffold",
                    "strategy_family": "scaffold",
                    "output_dir": str(output_dir / "scaffold_split"),
                    "seed": 42,
                },
            ],
        }

    monkeypatch.setattr(
        toolkit,
        "_run_training_worker",
        fake_run_training_worker,
    )

    result = toolkit.train_tabicl_model(
        train_csv=str(tmp_path / "dataset.csv"),
        task_type="regression",
        output_dir=str(output_dir),
        target_columns=["Y"],
        validation_protocol="standard_qsar",
        agent=agent,
    )

    assert result["model_path"].endswith("best.pkl")
    training_runs = agent.session_state["prediction_models"]["training_runs"]
    assert len(training_runs) == 1
    assert training_runs[0]["validation_protocol"] == "standard_qsar"
    assert captured_job_payload["extra_args"]["validation_protocol"] == "standard_qsar"
    assert training_runs[0]["split_runs"][0]["label"] == "random"
    assert training_runs[0]["seed_policy"]["model_seed"] == 42
