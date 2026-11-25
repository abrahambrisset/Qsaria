"""Deterministic coordinator scenarios against the production stdio MCP server."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from scripts.call_qsaria_v1_mcp import ROOT, payload, server_parameters

FIXTURE = ROOT / "benchmarks/fixtures/qsaria-v1-synthetic.csv"
CURATE = "qsaria_curation_curate_qsar_dataset"
TRAIN = "qsaria_training_train_lightgbm_model"
REGISTER = "qsaria_registry_register_model"
REQUEST = {
    "target_columns": ["clogp"],
    "task_type": "regression",
    "representation": {"kind": "generated", "name": "morgan_only"},
    "backend": {"device": "cpu", "n_estimators": 12, "num_leaves": 6, "min_child_samples": 2},
    "validation": {"kind": "single_random_holdout", "seed": 42},
    "compute": {"profile": "local_light"},
    "outlier_analysis": {"enabled": False},
    "applicability_domain": {"methods": ["bounding_box"]},
}


async def call(session, tool, *, error=None, **arguments):
    response = await session.call_tool(tool, arguments)
    result = payload(response)
    if error is not None:
        assert response.isError, f"{tool} accepted a request that should be refused"
        assert error in str(result), result
    else:
        assert not response.isError, f"{tool}: {result}"
    return result


async def handoff(session, experiment_id, agent, *, artifact_ids, model_ids=(), error=None):
    return await call(
        session,
        "qsaria_record_handoff",
        experiment_id=experiment_id,
        handoff={
            "schema_version": "1.0",
            "experiment_id": experiment_id,
            "agent": agent,
            "status": "completed",
            "summary": f"Recorded {agent} evidence for the reference experiment.",
            "facts": {},
            "artifact_ids": list(artifact_ids),
            "model_ids": list(model_ids),
            "warnings": [],
            "blockers": [],
            "recommended_next_action": "",
        },
        error=error,
    )


def artifact_entry(artifacts, experiment_root, returned_path):
    relative_path = Path(returned_path).resolve().relative_to(experiment_root).as_posix()
    entry = next(item for item in artifacts if item["path"] == relative_path)
    assert entry["exists"] is True
    return entry


def recheck_training(result, experiment_root, prediction_csv):
    """Use the saved index map and predictions as the source for the test statistics."""
    assert result["backend_name"] == "lightgbm"
    assert result["validation_protocol"] == "single_random_holdout"
    assert result["requested_contract"]["tuning"]["enabled"] is False
    [run] = result["split_results"]
    assert run["strategy_label"] == "random_seed_42"
    assert run["seed"] == 42

    paths = {
        key: Path(result[key]).resolve().relative_to(experiment_root)
        for key in ("candidate_train_csv", "summary_path", "model_path")
    }
    assert (experiment_root / paths["model_path"]).stat().st_size > 0
    summary = json.loads((experiment_root / paths["summary_path"]).read_text())
    assert summary["effective_train_args"]["device_type"] == "cpu"
    assert summary["effective_train_args"]["n_estimators"] == 12
    assert not summary.get("hyperparameter_tuning")

    split_path = Path(run["splits_path"]).resolve().relative_to(experiment_root)
    [split] = json.loads((experiment_root / split_path).read_text())
    indices = {name: split[name] for name in ("train", "val", "test")}
    assert [len(rows) for rows in indices.values()] == [32, 4, 4]
    assert sorted(sum(indices.values(), [])) == list(range(40))
    assert split["metadata"]["random_state"] == run["seed"]
    split_hash = hashlib.sha256(
        json.dumps(indices, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    assert split["metadata"]["split_hash"] == split_hash

    with (experiment_root / paths["candidate_train_csv"]).open(newline="") as file:
        targets = [float(row["clogp"]) for row in csv.DictReader(file)]
    predictions = list(csv.DictReader(prediction_csv.splitlines()))
    assert [float(row["y_true"]) for row in predictions] == [targets[i] for i in indices["test"]]
    errors = [float(row["y_pred"]) - float(row["y_true"]) for row in predictions]
    metrics = {
        "n": len(errors),
        "mae": fmean(abs(value) for value in errors),
        "rmse": math.sqrt(fmean(value**2 for value in errors)),
    }
    assert run["test_count"] == metrics["n"]
    for key, value in metrics.items():
        assert run["metrics"]["test"][key] == pytest.approx(value, rel=1e-10)
        assert summary["metrics"]["test"][key] == pytest.approx(value, rel=1e-10)
    return metrics, split_hash


def server_events(experiment_root):
    manifests = experiment_root.glob("workflows/manifests/qsaria/*.json")
    events = (json.loads(path.read_text()) for path in manifests)
    return sorted(events, key=lambda event: event["started_at"])


async def run_scenario(session, workspace, *, recover_missing_handoff):
    experiment_id = "exp_ci_handoff_recovery" if recover_missing_handoff else "exp_ci_nominal"
    experiment_root = workspace / "storage" / "sessions" / experiment_id
    scope = {"experiment_id": experiment_id}
    created = await call(
        session,
        "qsaria_create_experiment",
        **scope,
        user_request="Curate the local fixture and fit a 12-tree LightGBM model for this session.",
        report_language="en",
        metadata={"persistence_policy": "session_only"},
    )
    assert created["experiment_id"] == experiment_id
    curated = await call(
        session,
        CURATE,
        **scope,
        dataset_path=str(FIXTURE),
        task_type="regression",
        smiles_column="smiles",
        target_columns=["clogp"],
    )
    assert curated["ready_for_qsar"] is True and curated["rows_out"] == 40
    artifacts = await call(session, "qsaria_list_artifacts", **scope)
    dataset = artifact_entry(artifacts, experiment_root, curated["curated_dataset_path"])
    train_args = {"train_csv": curated["curated_dataset_path"], "request": REQUEST}

    if recover_missing_handoff:
        await call(
            session, TRAIN, **scope, **train_args, error="must record its structured handoff"
        )
        await handoff(
            session,
            experiment_id,
            "qsaria_curation",
            artifact_ids=["art_unknown_fixture"],
            error="unknown artifact_ids",
        )
        state = await call(session, "qsaria_get_experiment_state", **scope)
        assert state["experiment_id"] == experiment_id and state["handoffs"] == []
        assert [event["tool_name"] for event in server_events(experiment_root)] == [CURATE]

    cur_handoff = await handoff(
        session, experiment_id, "qsaria_curation", artifact_ids=[dataset["artifact_id"]]
    )
    result = await call(session, TRAIN, **scope, **train_args)
    artifacts = await call(session, "qsaria_list_artifacts", **scope)
    predictions = artifact_entry(artifacts, experiment_root, result["test_predictions_path"])
    split = artifact_entry(artifacts, experiment_root, result["splits_path"])
    model = artifact_entry(artifacts, experiment_root, result["model_path"])
    preview = await call(
        session, "qsaria_get_artifact", **scope, artifact_id=predictions["artifact_id"]
    )
    assert preview["source"] == TRAIN and preview["preview_truncated"] is False
    assert preview["preview"] == (experiment_root / predictions["path"]).read_text()
    metrics, split_hash = recheck_training(result, experiment_root, preview["preview"])

    model_id = result["recommended_registry_payload"]["model_id"]
    train_handoff = await handoff(
        session,
        experiment_id,
        "qsaria_training",
        artifact_ids=[predictions["artifact_id"], split["artifact_id"], model["artifact_id"]],
        model_ids=[model_id],
    )
    registered = await call(session, REGISTER, **scope, **result["recommended_registry_payload"])
    assert registered["model_id"] == model_id
    await handoff(session, experiment_id, "qsaria_registry", artifact_ids=[], model_ids=[model_id])

    context = await call(session, "qsaria_report_build_context", **scope)
    assert context["experiment_id"] == experiment_id
    facts = next(p["report_facts"] for p in context["reporting_packets"] if p["tool_name"] == TRAIN)
    assert facts["evaluation"]["scope"] == "internal"
    [fold] = facts["validation_studies"]["folds"]
    assert fold["label"] == "random_seed_42"
    for key, value in metrics.items():
        assert fold["metrics"]["test"][key] == pytest.approx(value, rel=1e-10)

    report = {
        "model_id": model_id,
        "split_hash": split_hash,
        "prediction_artifact_id": predictions["artifact_id"],
        "test": metrics,
    }
    saved = await call(
        session, "qsaria_report_save", **scope, report_format="json", report_content=report
    )
    await handoff(session, experiment_id, "qsaria_report", artifact_ids=[saved["artifact_id"]])
    final = await call(session, "qsaria_complete_experiment", **scope)
    assert final["status"] == "completed"
    assert model_id in final["model_ids"]
    assert final["report"]["artifact_id"] == saved["artifact_id"]
    report_artifact = await call(
        session, "qsaria_get_artifact", **scope, artifact_id=saved["artifact_id"]
    )
    assert json.loads(report_artifact["preview"]) == report
    events = server_events(experiment_root)
    assert [(event["tool_name"], event["status"]) for event in events] == [
        (CURATE, "success"),
        (TRAIN, "success"),
        (REGISTER, "success"),
    ]
    assert events[0]["completed_at"] <= cur_handoff["recorded_at"] <= events[1]["started_at"]
    assert events[1]["completed_at"] <= train_handoff["recorded_at"] <= events[2]["started_at"]
    assert model_id in events[2]["model_ids"]
    assert {split["artifact_id"], predictions["artifact_id"]} <= set(events[1]["artifact_ids"])
    return split_hash, metrics


def test_reference_qsaria_mcp_scenarios():
    async def run(workspace):
        async with (
            asyncio.timeout(120),
            stdio_client(server_parameters(workspace)) as (reader, writer),
            ClientSession(reader, writer) as session,
        ):
            await session.initialize()
            bootstrap = await call(session, "qsaria_bootstrap")
            assert bootstrap["status"] == "ok" and bootstrap["llm_policy"] == "disabled"
            nominal = await run_scenario(session, workspace, recover_missing_handoff=False)
            recovered = await run_scenario(session, workspace, recover_missing_handoff=True)
            assert nominal == recovered

    parent = ROOT / ".files/publication/agentic-ci"
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="mcp-", dir=parent) as folder:
        asyncio.run(run(Path(folder)))
