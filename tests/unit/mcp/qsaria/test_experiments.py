"""Persistence, isolation and reporting tests for Qsaria MCP experiments."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from cs_copilot.mcp.context import get_current_context
from cs_copilot.mcp.qsaria import (
    ExperimentAlreadyExistsError,
    ExperimentManager,
    ExperimentNotFoundError,
    ExperimentStateTransitionError,
    IncompatibleExperimentSchemaError,
    InvalidHandoffError,
    InvalidInputPathError,
)
from cs_copilot.mcp.qsaria.contracts import HANDOFF_SCHEMA_VERSION
from cs_copilot.mcp.qsaria.experiments import PUBLIC_STATE_REL_PATH, RUNTIME_STATE_REL_PATH
from cs_copilot.storage import S3
from cs_copilot.tools.prediction import catalog as catalog_module
from cs_copilot.tools.prediction.catalog import (
    PredictionModelCatalog,
    model_catalog_lock_path,
)


@pytest.fixture
def manager(tmp_path):
    return ExperimentManager(local_root=tmp_path / "storage")


def _state_path(manager, experiment_id, rel_path):
    return manager._session_root(experiment_id) / rel_path


def _handoff(experiment_id, *, agent, status, summary, **overrides):
    payload = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "agent": agent,
        "status": status,
        "summary": summary,
        "facts": {},
        "artifact_ids": [],
        "model_ids": [],
        "warnings": [],
        "blockers": (
            [summary]
            if status in {"retryable_error", "terminal_failure", "needs_user_input"}
            else []
        ),
        "recommended_next_action": "",
    }
    payload.update(overrides)
    return payload


def _persist_interrupted_call(
    manager,
    experiment_id,
    *,
    agent="qsaria_training",
    tool_name="qsaria_training_train_qsar_model",
    automatic_retry=False,
):
    with manager._locked(experiment_id):
        public_state, runtime_state = manager._load_states(experiment_id)
        public_state["status"] = "running"
        public_state["phase"] = tool_name
        runtime_state["active_call"] = {
            "call_id": "interrupted123",
            "agent": agent,
            "tool_name": tool_name,
            "started_at": public_state["updated_at"],
            "automatic_retry": automatic_retry,
        }
        manager._save_states(experiment_id, public_state, runtime_state)


def test_bootstrap_does_not_list_resume_or_create_experiments(manager):
    result = manager.bootstrap()

    assert result["llm_policy"] == "disabled"
    assert result["model"] is None
    assert result["auto_resume"] is False
    assert result["active_experiment_id"] is None
    assert result["experiments_listed"] is False
    assert result["contracts"]["report_facts"] == "2.0"
    assert result["compatibility"]["target"] == "external_mcp_coordinator_v1"
    assert result["compatibility"]["coordinator_contract"] == ("external_mcp_coordinator_v1")
    assert result["compatibility"]["supported_clients"] == [
        "codex_v1",
        "claude_code_v1",
    ]
    assert result["compatibility"]["agno_chainlit_runtime_changed"] is False
    assert result["compatibility"]["storage_concurrency"]["distributed_lock"] is True
    assert result["error_policy"]["retryable_error_max_automatic_retries"] == 1
    assert len(result["capabilities"]["sub_agents"]) == 5
    assert result["versions"]["qsaria"]
    assert not manager._local_root().exists()


def test_create_persists_separate_public_and_private_documents(manager):
    state = manager.create_experiment(
        "Train a solubility model",
        report_language="fr",
        metadata={"source": "unit-test"},
        experiment_id="exp_persist_123",
    )

    public_path = _state_path(manager, "exp_persist_123", PUBLIC_STATE_REL_PATH)
    runtime_path = _state_path(manager, "exp_persist_123", RUNTIME_STATE_REL_PATH)
    public_payload = json.loads(public_path.read_text())
    runtime_payload = json.loads(runtime_path.read_text())

    assert state == public_payload
    assert public_payload["request_summary"] == "Train a solubility model"
    assert public_payload["metadata"]["persistence_policy"] == "catalog"
    assert "session_state" not in public_payload
    assert runtime_payload["session_state"] == {}
    assert runtime_payload["experiment_id"] == public_payload["experiment_id"]
    assert (public_path.parent / ".experiment.lock").exists()
    assert not list(public_path.parent.glob("*.tmp"))


def test_create_accepts_only_explicit_persistence_policies(manager):
    session_only = manager.create_experiment(
        "Train without catalog persistence",
        metadata={"persistence_policy": "session_only"},
        experiment_id="exp_session_only_policy_123",
    )
    assert session_only["metadata"]["persistence_policy"] == "session_only"

    with pytest.raises(ValueError, match="metadata.persistence_policy"):
        manager.create_experiment(
            "Invalid persistence policy",
            metadata={"persistence_policy": "maybe"},
            experiment_id="exp_invalid_policy_123",
        )


def test_duplicate_and_unknown_experiments_fail_explicitly(manager):
    manager.create_experiment("first", experiment_id="exp_duplicate_123")

    with pytest.raises(ExperimentAlreadyExistsError):
        manager.create_experiment("second", experiment_id="exp_duplicate_123")
    with pytest.raises(ExperimentNotFoundError):
        manager.open_experiment("exp_missing_123")


def test_experiment_state_and_index_survive_manager_restart(manager):
    manager.create_experiment("restart", experiment_id="exp_restart_123")
    with manager.run(
        "exp_restart_123",
        agent_name="qsaria_training",
        tool_name="qsaria_training_train_qsar_model",
    ) as runtime:
        runtime.session_state["checkpoint"] = {"trial": 3}
        runtime.capture_result({"status": "completed", "model_id": "model-restart-123"})

    restarted = ExperimentManager(local_root=manager._local_root())

    assert restarted.open_experiment("exp_restart_123")["model_ids"] == ["model-restart-123"]
    assert restarted.list_experiments()[0]["experiment_id"] == "exp_restart_123"
    runtime_payload = json.loads(
        _state_path(restarted, "exp_restart_123", RUNTIME_STATE_REL_PATH).read_text()
    )
    assert runtime_payload["session_state"]["checkpoint"] == {"trial": 3}


def test_run_binds_model_free_context_and_restores_both_contextvars(manager):
    manager.create_experiment("curate", experiment_id="exp_context_123")
    previous_prefix = S3.current_prefix()
    previous_context = get_current_context()

    with manager.run(
        "exp_context_123",
        agent_name="qsaria_curation",
        tool_name="qsaria_curation_inspect_dataset_schema",
    ) as runtime:
        assert S3.current_prefix() == "sessions/exp_context_123"
        assert get_current_context() is runtime.context
        assert runtime.context.model is None
        assert runtime.context.llm is None
        assert runtime.context.llm_policy == "disabled"
        runtime.session_state["qsar_curation"] = {"rows": 42}
        runtime.capture_result({"status": "completed"})

    assert S3.current_prefix() == previous_prefix
    assert get_current_context() is previous_context
    runtime_payload = json.loads(
        _state_path(manager, "exp_context_123", RUNTIME_STATE_REL_PATH).read_text()
    )
    assert runtime_payload["session_state"]["qsar_curation"]["rows"] == 42
    assert runtime_payload["tool_events"][-1]["status"] == "success"
    assert manager.get_experiment_state("exp_context_123")["status"] == "active"


def test_successful_role_must_record_handoff_before_another_role_runs(manager):
    experiment_id = "exp_handoff_barrier_123"
    manager.create_experiment("curate then train", experiment_id=experiment_id)

    with manager.run(
        experiment_id,
        agent_name="qsaria_curation",
        tool_name="qsaria_curation_curate_qsar_dataset",
    ) as runtime:
        runtime.capture_result({"status": "completed"})

    # One mission may use several tools before its single handoff.
    with manager.run(
        experiment_id,
        agent_name="qsaria_curation",
        tool_name="qsaria_curation_write_curation_report",
    ) as runtime:
        runtime.capture_result({"status": "completed"})

    with pytest.raises(ExperimentStateTransitionError, match="must record its structured handoff"):
        with manager.run(
            experiment_id,
            agent_name="qsaria_training",
            tool_name="qsaria_training_train_lightgbm_model",
        ):
            pass
    with pytest.raises(ExperimentStateTransitionError, match="must record its structured handoff"):
        manager.record_handoff(
            experiment_id,
            _handoff(
                experiment_id,
                agent="qsaria_training",
                status="completed",
                summary="Training cannot skip curation evidence",
            ),
        )

    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_curation",
            status="completed",
            summary="Curation evidence recorded",
        ),
    )

    with manager.run(
        experiment_id,
        agent_name="qsaria_training",
        tool_name="qsaria_training_train_lightgbm_model",
    ) as runtime:
        runtime.capture_result({"status": "completed"})


def test_pending_success_handoff_blocks_reporting_and_completion(manager):
    experiment_id = "exp_pending_handoff_finalization_123"
    manager.create_experiment("train", experiment_id=experiment_id)
    with manager.run(
        experiment_id,
        agent_name="qsaria_training",
        tool_name="qsaria_training_train_lightgbm_model",
    ) as runtime:
        runtime.capture_result({"status": "completed"})

    with pytest.raises(ExperimentStateTransitionError, match="cannot build the final report"):
        manager.build_report_context(experiment_id)
    with pytest.raises(ExperimentStateTransitionError, match="cannot save the final report"):
        manager.save_report(experiment_id, "# Incomplete report\n")
    with pytest.raises(ExperimentStateTransitionError, match="cannot finalize an experiment"):
        manager.complete_experiment(experiment_id)

    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_training",
            status="completed",
            summary="Training evidence recorded",
        ),
    )
    assert manager.build_report_context(experiment_id)["experiment_id"] == experiment_id


def test_run_defaults_unknown_failures_to_terminal_and_does_not_mask_original(manager):
    manager.create_experiment("train", experiment_id="exp_failure_123")
    previous_prefix = S3.current_prefix()

    with pytest.raises(RuntimeError, match="backend unavailable"):
        with manager.run(
            "exp_failure_123",
            agent_name="qsaria_training",
            tool_name="qsaria_training_train_qsar_model",
        ) as runtime:
            runtime.session_state["checkpoint"] = {"phase": "training"}
            raise RuntimeError("backend unavailable")

    assert S3.current_prefix() == previous_prefix
    public = manager.get_experiment_state("exp_failure_123")
    runtime_payload = json.loads(
        _state_path(manager, "exp_failure_123", RUNTIME_STATE_REL_PATH).read_text()
    )
    assert public["status"] == "terminal_failure"
    assert "backend unavailable" in public["blockers"]
    assert runtime_payload["session_state"]["checkpoint"]["phase"] == "training"
    assert runtime_payload["tool_events"][-1]["status"] == "terminal_failure"


def test_result_extraction_registers_real_artifact_and_model_ids(manager):
    manager.create_experiment("infer", experiment_id="exp_artifacts_123")
    with manager.run(
        "exp_artifacts_123",
        agent_name="qsaria_inference",
        tool_name="qsaria_inference_predict_from_csv",
    ) as runtime:
        output_path = Path(runtime.artifact_path("predictions.csv"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("smiles,prediction\nCCO,1.2\n")
        runtime.capture_result(
            {
                "status": "completed",
                "preds_path": str(output_path),
                "model_id": "qsaria-model-123",
            }
        )

    state = manager.get_experiment_state("exp_artifacts_123")
    artifacts = manager.list_artifacts("exp_artifacts_123")
    assert state["model_ids"] == ["qsaria-model-123"]
    assert len(state["artifact_ids"]) == 1
    assert len(artifacts) == 1
    assert artifacts[0]["artifact_id"].startswith("art_")
    assert artifacts[0]["exists"] is True
    assert artifacts[0]["kind"] == "dataset"

    fetched = manager.get_artifact(
        "exp_artifacts_123", artifacts[0]["artifact_id"], max_preview_bytes=8
    )
    assert fetched["preview"] == "smiles,p"
    assert fetched["preview_truncated"] is True

    runtime_payload = json.loads(
        _state_path(manager, "exp_artifacts_123", RUNTIME_STATE_REL_PATH).read_text()
    )
    event = runtime_payload["tool_events"][-1]
    assert event["artifact_ids"] == [artifacts[0]["artifact_id"]]
    assert event["model_ids"] == ["qsaria-model-123"]


def test_artifact_reader_rejects_absolute_paths_outside_managed_roots(manager, tmp_path):
    manager.create_experiment("inspect", experiment_id="exp_external_path_123")
    external = tmp_path / "external-secret.txt"
    external.write_text("not an experiment artifact")
    with manager.run(
        "exp_external_path_123",
        agent_name="qsaria_curation",
        tool_name="qsaria_curation_inspect_dataset_schema",
    ) as runtime:
        runtime.capture_result({"source_dataset_path": str(external)})

    assert manager.list_artifacts("exp_external_path_123") == []
    runtime_payload = json.loads(
        _state_path(manager, "exp_external_path_123", RUNTIME_STATE_REL_PATH).read_text()
    )
    provenance = next(iter(runtime_payload["provenance_inputs"].values()))
    assert provenance["retrievable"] is False
    assert provenance["name"] == external.name
    assert str(external) not in json.dumps(provenance)


def test_deleted_artifact_is_refreshed_and_cannot_be_claimed_by_a_handoff(manager):
    experiment_id = "exp_deleted_artifact_123"
    manager.create_experiment("artifact freshness", experiment_id=experiment_id)
    with manager.run(
        experiment_id,
        agent_name="qsaria_training",
        tool_name="qsaria_training_train_qsar_model",
    ) as runtime:
        model_path = Path(runtime.artifact_path("model.pkl"))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"abc")
        runtime.capture_result({"model_path": str(model_path)})

    artifact = manager.list_artifacts(experiment_id)[0]
    assert artifact["exists"] is True
    assert artifact["size"] == 3
    model_path.unlink()

    refreshed = manager.list_artifacts(experiment_id)[0]
    assert refreshed["exists"] is False
    assert refreshed["size"] is None
    fetched = manager.get_artifact(
        experiment_id,
        artifact["artifact_id"],
        max_preview_bytes=0,
    )
    assert fetched["exists"] is False
    with pytest.raises(InvalidHandoffError, match="unavailable artifact_ids"):
        manager.record_handoff(
            experiment_id,
            _handoff(
                experiment_id,
                agent="qsaria_training",
                status="completed",
                summary="Must not claim deleted evidence",
                artifact_ids=[artifact["artifact_id"]],
            ),
        )


def test_handoff_cannot_promote_an_arbitrary_s3_uri_to_artifact(manager):
    experiment_id = "exp_s3_handoff_guard_123"
    manager.create_experiment("guard remote evidence", experiment_id=experiment_id)
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_curation",
            status="completed",
            summary="No managed output",
            facts={"artifact_path": "s3://foreign-bucket/private/secret.csv"},
        ),
    )

    assert manager.list_artifacts(experiment_id) == []
    runtime_payload = json.loads(
        _state_path(manager, experiment_id, RUNTIME_STATE_REL_PATH).read_text()
    )
    assert runtime_payload["artifacts"] == {}
    assert next(iter(runtime_payload["provenance_inputs"].values()))["scheme"] == "s3"


def test_handoff_uses_only_claimed_or_new_artifacts_and_rejects_unknown_ids(manager):
    manager.create_experiment("workflow", experiment_id="exp_handoff_123")
    with manager.run(
        "exp_handoff_123",
        agent_name="qsaria_curation",
        tool_name="qsaria_curation_curate_qsar_dataset",
    ) as runtime:
        first_path = Path(runtime.artifact_path("curated.csv"))
        first_path.parent.mkdir(parents=True, exist_ok=True)
        first_path.write_text("smiles,target\nCCO,1\n")
        runtime.capture_result({"curated_dataset_path": str(first_path)})
    first_id = manager.list_artifacts("exp_handoff_123")[0]["artifact_id"]

    recorded = manager.record_handoff(
        "exp_handoff_123",
        _handoff(
            "exp_handoff_123",
            agent="qsaria_curation",
            status="completed",
            summary="Curated",
            facts={"rows": 1},
            artifact_ids=[first_id],
            recommended_next_action="train",
        ),
    )
    assert recorded["artifact_ids"] == [first_id]

    with pytest.raises(InvalidHandoffError, match="unknown artifact_ids"):
        manager.record_handoff(
            "exp_handoff_123",
            _handoff(
                "exp_handoff_123",
                agent="qsaria_training",
                status="completed",
                summary="Invalid claim",
                artifact_ids=["art_does_not_exist"],
            ),
        )


def test_report_context_save_once_and_complete(manager):
    manager.create_experiment("train", experiment_id="exp_report_123")
    with manager.run(
        "exp_report_123",
        agent_name="qsaria_training",
        tool_name="qsaria_training_train_lightgbm_model",
    ) as runtime:
        runtime.capture_result({"status": "completed", "model_id": "model-report-123"})
    manager.record_handoff(
        "exp_report_123",
        _handoff(
            "exp_report_123",
            agent="qsaria_training",
            status="completed",
            summary="Training done",
            facts={
                "report_facts": {
                    "schema_version": "2.0",
                    "report_kind": "training",
                    "evaluation": {"scope": "internal"},
                },
                "report_tables": {
                    "selection_validation_metrics": {
                        "kind": "selection_validation_metrics",
                        "rows": [{"model": "m1", "r2": 0.8}],
                    }
                },
            },
            model_ids=["model-report-123"],
        ),
    )

    context = manager.build_report_context("exp_report_123")
    assert context["report_facts"][0]["schema_version"] == "2.0"
    assert context["report_tables"][0]["selection_validation_metrics"]["rows"][0]["r2"] == 0.8
    assert context["reporting_packets"][0]["agent"] == "qsaria_training"
    assert context["evidence_precedence"][0] == "structured_artifacts"

    report = manager.save_report("exp_report_123", "# Qsaria report\n\nResults.")
    assert report["artifact_id"].startswith("art_")
    assert report["path"] == "workflows/reports/qsaria_report.md"
    with pytest.raises(FileExistsError, match="already saved"):
        manager.save_report("exp_report_123", "# second report")
    manager.record_handoff(
        "exp_report_123",
        _handoff(
            "exp_report_123",
            agent="qsaria_report",
            status="completed",
            summary="Final report saved",
            artifact_ids=[report["artifact_id"]],
        ),
    )

    completed = manager.complete_experiment("exp_report_123", summary="Verified by coordinator")
    assert completed["status"] == "completed"
    assert completed["completion_summary"] == "Verified by coordinator"
    assert manager.get_artifact("exp_report_123", report["artifact_id"])["preview"].startswith(
        "# Qsaria report"
    )


def test_training_reporting_packets_survive_handoff_compaction_and_require_registry(manager):
    experiment_id = "exp_durable_reporting_123"
    manager.create_experiment("train and persist", experiment_id=experiment_id)
    with manager.run(
        experiment_id,
        agent_name="qsaria_training",
        tool_name="qsaria_training_train_lightgbm_model",
    ) as runtime:
        manifest = Path(
            runtime.artifact_path(
                "catalog_candidates_manifest.json",
                category="training",
            )
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text('{"candidate_registry_payloads": [{"model_id": "candidate"}]}')
        runtime.capture_result(
            {
                "status": "completed",
                "candidate_manifest_path": str(manifest),
                "persistence_plan": {
                    "persist_all_candidates": True,
                    "candidate_count": 1,
                    "candidate_manifest_path": str(manifest),
                },
                "reporting_handoff": {
                    "report_facts": {
                        "schema_version": "2.0",
                        "report_kind": "training",
                        "evaluation": {"scope": "external"},
                    },
                    "report_tables": {
                        "final_test_comparison": {
                            "kind": "final_test_comparison",
                            "rows": [{"source": "baseline", "rmse": 0.7}],
                        }
                    },
                },
            }
        )
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_training",
            status="completed",
            summary="Training handoff intentionally compacted",
            facts={"candidate_manifest_path": str(manifest)},
        ),
    )

    state = manager.get_experiment_state(experiment_id)
    assert state["persistence"]["status"] == "pending"
    assert state["persistence"]["required_tools"] == [
        "qsaria_registry_register_and_persist_candidates"
    ]
    with pytest.raises(ExperimentStateTransitionError, match="catalog persistence is pending"):
        manager.build_report_context(experiment_id)
    with pytest.raises(ExperimentStateTransitionError, match="catalog persistence must be"):
        with manager.run(
            experiment_id,
            agent_name="qsaria_inference",
            tool_name="qsaria_inference_predict_from_csv",
        ):
            pass

    with manager.run(
        experiment_id,
        agent_name="qsaria_registry",
        tool_name="qsaria_registry_register_and_persist_candidates",
    ) as runtime:
        runtime.capture_result(
            {
                "persisted": True,
                "candidate_count": 1,
                "candidates": [
                    {
                        "model_id": "catalog-model-123",
                        "model_root": "data/model_assets/internal/catalog-model-123",
                        "model_path": (
                            "data/model_assets/internal/catalog-model-123/model/best.pkl"
                        ),
                        "metadata_path": (
                            "data/model_assets/internal/catalog-model-123/metadata.json"
                        ),
                    }
                ],
            }
        )
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_registry",
            status="completed",
            summary="Candidate materialized in the durable catalog",
            model_ids=["catalog-model-123"],
        ),
    )

    state = manager.get_experiment_state(experiment_id)
    assert state["persistence"]["status"] == "completed"
    assert state["persistence"]["evidence"]["model_ids"] == ["catalog-model-123"]
    context = manager.build_report_context(experiment_id)
    assert context["reporting_packets"][0]["source"] == "structured_tool_result"
    assert context["report_facts"][0]["schema_version"] == "2.0"
    assert context["report_tables"][0]["final_test_comparison"]["rows"][0]["rmse"] == 0.7


def test_explicit_session_only_policy_skips_registry_barrier(manager):
    experiment_id = "exp_session_only_training_123"
    manager.create_experiment(
        "train without persistence",
        metadata={"persistence_policy": "session_only"},
        experiment_id=experiment_id,
    )
    with manager.run(
        experiment_id,
        agent_name="qsaria_training",
        tool_name="qsaria_training_train_lightgbm_model",
    ) as runtime:
        manifest = Path(
            runtime.artifact_path(
                "catalog_candidates_manifest.json",
                category="training",
            )
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text('{"candidate_registry_payloads": [{"model_id": "candidate"}]}')
        runtime.capture_result(
            {
                "candidate_manifest_path": str(manifest),
                "persistence_plan": {
                    "persist_all_candidates": True,
                    "candidate_count": 1,
                    "candidate_manifest_path": str(manifest),
                },
            }
        )
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_training",
            status="completed",
            summary="Session-only training complete",
        ),
    )

    state = manager.get_experiment_state(experiment_id)
    assert state["persistence"]["status"] == "skipped_by_user"
    assert manager.build_report_context(experiment_id)["experiment_id"] == experiment_id


def test_single_model_persistence_barrier_requires_register_then_persist(manager):
    experiment_id = "exp_single_model_persistence_123"
    manager.create_experiment("train one model", experiment_id=experiment_id)
    with manager.run(
        experiment_id,
        agent_name="qsaria_training",
        tool_name="qsaria_training_train_qsar_model",
    ) as runtime:
        runtime.capture_result(
            {
                "recommended_registry_payload": {
                    "model_id": "candidate-single-123",
                    "model_path": "workflows/training/model/best.pkl",
                }
            }
        )
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_training",
            status="completed",
            summary="Single candidate trained",
        ),
    )

    assert manager.get_experiment_state(experiment_id)["persistence"]["required_tools"] == [
        "qsaria_registry_register_model"
    ]
    with manager.run(
        experiment_id,
        agent_name="qsaria_registry",
        tool_name="qsaria_registry_register_model",
    ) as runtime:
        runtime.capture_result({"registered": True, "model_id": "candidate-single-123"})
    assert manager.get_experiment_state(experiment_id)["persistence"]["required_tools"] == [
        "qsaria_registry_persist_registered_model"
    ]

    with manager.run(
        experiment_id,
        agent_name="qsaria_registry",
        tool_name="qsaria_registry_persist_registered_model",
    ) as runtime:
        runtime.capture_result(
            {
                "persisted": True,
                "model_id": "catalog-single-123",
                "model_root": "data/model_assets/internal/catalog-single-123",
                "model_path": "data/model_assets/internal/catalog-single-123/model/best.pkl",
                "metadata_path": "data/model_assets/internal/catalog-single-123/metadata.json",
            }
        )
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_registry",
            status="completed",
            summary="Single candidate persisted",
            model_ids=["catalog-single-123"],
        ),
    )

    persistence = manager.get_experiment_state(experiment_id)["persistence"]
    assert persistence["status"] == "completed"
    assert persistence["evidence"]["model_root"].endswith("catalog-single-123")


def test_list_is_explicit_filterable_and_compact(manager):
    manager.create_experiment("one", experiment_id="exp_list_one")
    manager.create_experiment("two", experiment_id="exp_list_two")
    manager.record_handoff(
        "exp_list_one",
        _handoff(
            "exp_list_one",
            agent="qsaria_curation",
            status="completed",
            summary="Inspection complete",
        ),
    )
    manager.complete_experiment("exp_list_one")

    completed = manager.list_experiments(status="completed")
    assert [item["experiment_id"] for item in completed] == ["exp_list_one"]
    assert "handoffs" not in completed[0]
    assert manager.open_experiment("exp_list_two")["request_summary"] == "two"


def test_concurrent_experiments_keep_prefix_and_runtime_state_isolated(manager):
    ids = ["exp_thread_one", "exp_thread_two"]
    for experiment_id in ids:
        manager.create_experiment(experiment_id, experiment_id=experiment_id)

    def execute(experiment_id):
        with manager.run(
            experiment_id,
            agent_name="qsaria_registry",
            tool_name="qsaria_registry_list_catalog_models",
        ) as runtime:
            observed_prefix = S3.current_prefix()
            runtime.session_state["owner"] = experiment_id
            runtime.capture_result({"status": "completed"})
            return observed_prefix

    with ThreadPoolExecutor(max_workers=2) as pool:
        prefixes = list(pool.map(execute, ids))

    assert set(prefixes) == {"sessions/exp_thread_one", "sessions/exp_thread_two"}
    for experiment_id in ids:
        runtime_payload = json.loads(
            _state_path(manager, experiment_id, RUNTIME_STATE_REL_PATH).read_text()
        )
        assert runtime_payload["session_state"]["owner"] == experiment_id


def test_catalog_lock_is_shared_with_prediction_catalog(manager, monkeypatch, tmp_path):
    catalog_path = tmp_path / "catalog" / "model_catalog.json"
    monkeypatch.setattr(catalog_module, "DEFAULT_MODEL_CATALOG_PATH", catalog_path)
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", tmp_path / "internal")

    with manager.catalog_lock():
        assert model_catalog_lock_path(catalog_path).exists()
        # Catalog methods lock again while the MCP manager still owns the
        # outer lock, so this also protects the same-thread nesting contract.
        catalog = PredictionModelCatalog.load(str(catalog_path))
        catalog.save()

    assert not (manager._local_root() / "qsaria" / ".catalog.lock").exists()


def test_s3_metadata_mode_keeps_path_based_toolkit_outputs_local(manager, monkeypatch):
    monkeypatch.setattr(manager, "_uses_s3", lambda: True)

    resolved = manager.resolve_artifact_path(
        "exp_s3_layout_123",
        "workflows/training/output",
    )

    assert resolved == str(
        manager._local_root()
        / "sessions"
        / "exp_s3_layout_123"
        / "workflows"
        / "training"
        / "output"
    )
    assert not resolved.startswith("s3://")


def test_s3_metadata_mode_resolves_registered_relative_input_locally(manager, monkeypatch):
    experiment_id = manager.create_experiment("registered local input")["experiment_id"]
    with manager.run(
        experiment_id,
        agent_name="qsaria_curation",
        tool_name="qsaria_curation_curate_qsar_dataset",
    ) as runtime:
        curated = Path(runtime.artifact_path("curated_dataset.csv", category="curation"))
        curated.parent.mkdir(parents=True, exist_ok=True)
        curated.write_text("smiles,pEC50\nCCO,4.2\n", encoding="utf-8")
        runtime.capture_result(
            {
                "status": "completed",
                "curated_dataset_path": str(curated),
            }
        )

    artifact = next(
        item
        for item in manager.list_artifacts(experiment_id, kind="dataset")
        if item["path"].endswith("/curated_dataset.csv")
    )
    with manager._locked(experiment_id):
        _public_state, runtime_state = manager._load_states(experiment_id)

    monkeypatch.setattr(manager, "_uses_s3", lambda: True)
    resolved = manager.normalize_experiment_input_path(
        experiment_id,
        artifact["path"],
        runtime_state=runtime_state,
        parameter="train_csv",
    )

    assert resolved == str(curated.resolve())
    assert not resolved.startswith("s3://")


def test_incompatible_state_schema_is_rejected_without_rewrite(manager):
    experiment_id = "exp_future_schema_123"
    manager.create_experiment("future", experiment_id=experiment_id)
    public_path = _state_path(manager, experiment_id, PUBLIC_STATE_REL_PATH)
    payload = json.loads(public_path.read_text())
    payload["schema_version"] = "99.0"
    public_path.write_text(json.dumps(payload, indent=2) + "\n")
    before = public_path.read_bytes()

    with pytest.raises(IncompatibleExperimentSchemaError, match="unsupported public schema"):
        manager.open_experiment(experiment_id)

    assert public_path.read_bytes() == before


def test_terminal_and_completed_states_cannot_be_silently_reopened(manager):
    terminal_id = "exp_terminal_sticky_123"
    manager.create_experiment("terminal", experiment_id=terminal_id)
    with manager.run(
        terminal_id,
        agent_name="qsaria_curation",
        tool_name="qsaria_curation_curate_qsar_dataset",
    ) as runtime:
        runtime.capture_error("scientific blocker", status="terminal_failure")

    with pytest.raises(ExperimentStateTransitionError, match="does not allow"):
        with manager.run(
            terminal_id,
            agent_name="qsaria_training",
            tool_name="qsaria_training_train_qsar_model",
        ):
            pass
    with pytest.raises(ExperimentStateTransitionError, match="terminal experiments"):
        manager.record_handoff(
            terminal_id,
            _handoff(
                terminal_id,
                agent="qsaria_training",
                status="completed",
                summary="must not bypass terminal state",
            ),
        )
    terminal_handoff = manager.record_handoff(
        terminal_id,
        _handoff(
            terminal_id,
            agent="qsaria_curation",
            status="terminal_failure",
            summary="Curation stopped on a scientific blocker",
            blockers=["scientific blocker"],
        ),
    )
    assert terminal_handoff["status"] == "terminal_failure"
    assert manager.get_experiment_state(terminal_id)["status"] == "terminal_failure"
    with pytest.raises(ExperimentStateTransitionError, match="only one matching"):
        manager.record_handoff(
            terminal_id,
            _handoff(
                terminal_id,
                agent="qsaria_curation",
                status="terminal_failure",
                summary="duplicate terminal handoff",
            ),
        )
    finalized_terminal = manager.complete_experiment(
        terminal_id,
        status="terminal_failure",
        summary="Terminal evidence verified by coordinator",
    )
    assert finalized_terminal["phase"] == "completed"
    assert finalized_terminal["completed_at"]
    assert finalized_terminal["completion_summary"] == "Terminal evidence verified by coordinator"
    with pytest.raises(ExperimentStateTransitionError, match="cannot change"):
        manager.complete_experiment(terminal_id, status="completed")

    completed_id = "exp_completed_sticky_123"
    manager.create_experiment("completed", experiment_id=completed_id)
    manager.record_handoff(
        completed_id,
        _handoff(
            completed_id,
            agent="qsaria_curation",
            status="completed",
            summary="Completed evidence",
        ),
    )
    manager.complete_experiment(completed_id)
    with pytest.raises(ExperimentStateTransitionError, match="does not allow"):
        with manager.run(
            completed_id,
            agent_name="qsaria_inference",
            tool_name="qsaria_inference_predict_from_csv",
        ):
            pass


def test_s3_requires_explicit_single_writer_contract(monkeypatch):
    monkeypatch.setattr(ExperimentManager, "_uses_s3", lambda self: True)
    monkeypatch.delenv("QSARIA_S3_SINGLE_WRITER", raising=False)

    with pytest.raises(RuntimeError, match="QSARIA_S3_SINGLE_WRITER=true"):
        ExperimentManager()

    monkeypatch.setenv("QSARIA_S3_SINGLE_WRITER", "true")
    manager = ExperimentManager()
    assert manager.bootstrap()["compatibility"]["storage_concurrency"] == {
        "mode": "s3_single_writer",
        "distributed_lock": False,
        "single_writer_acknowledged": True,
    }


def test_agent_level_needs_user_input_resumes_with_the_same_role(manager):
    experiment_id = "exp_agent_pause_123"
    manager.create_experiment("ambiguous target", experiment_id=experiment_id)
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_curation",
            status="needs_user_input",
            summary="The target column is ambiguous",
            blockers=["Choose logS or measured_solubility"],
        ),
    )

    with pytest.raises(ExperimentStateTransitionError, match="only be resolved by role"):
        with manager.run(
            experiment_id,
            agent_name="qsaria_training",
            tool_name="qsaria_training_train_qsar_model",
        ):
            pass

    with manager.run(
        experiment_id,
        agent_name="qsaria_curation",
        tool_name="qsaria_curation_identify_qsar_columns",
    ) as runtime:
        runtime.capture_result({"status": "completed", "target_columns": ["logS"]})
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_curation",
            status="completed",
            summary="The user selected logS",
            facts={"target_columns": ["logS"]},
        ),
    )

    state = manager.get_experiment_state(experiment_id)
    assert state["status"] == "active"
    assert state["blockers"] == []


def test_partial_scientific_handoff_can_finish_with_one_report(manager):
    experiment_id = "exp_partial_report_123"
    manager.create_experiment("partial benchmark", experiment_id=experiment_id)
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_training",
            status="partial",
            summary="Two backends completed; one optional backend was unavailable",
            warnings=["Optional backend unavailable"],
        ),
    )
    assert manager.get_experiment_state(experiment_id)["status"] == "partial"

    report = manager.save_report(
        experiment_id,
        "# Partial Qsaria report\n\nVerified evidence.",
    )
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_report",
            status="completed",
            summary="Partial workflow report saved",
            artifact_ids=[report["artifact_id"]],
        ),
    )
    assert manager.get_experiment_state(experiment_id)["status"] == "partial"
    with pytest.raises(ExperimentStateTransitionError, match="partial scientific"):
        manager.complete_experiment(experiment_id)
    finalized = manager.complete_experiment(
        experiment_id,
        status="partial",
        summary="Partial evidence verified",
    )

    assert finalized["status"] == "partial"
    assert finalized["phase"] == "completed"
    assert sum(item["agent"] == "qsaria_report" for item in finalized["handoffs"]) == 1


@pytest.mark.parametrize("failure_status", ["retryable_error", "needs_user_input"])
def test_unresolved_failure_cannot_be_finalized_as_partial(manager, failure_status):
    experiment_id = f"exp_unresolved_{failure_status}"
    manager.create_experiment("pause", experiment_id=experiment_id)
    with manager.run(
        experiment_id,
        agent_name="qsaria_curation",
        tool_name="qsaria_curation_identify_qsar_columns",
    ) as runtime:
        runtime.capture_error("awaiting correction", status=failure_status)
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_curation",
            status=failure_status,
            summary="awaiting correction",
        ),
    )

    with pytest.raises(ExperimentStateTransitionError, match="cannot finalize unresolved"):
        manager.complete_experiment(experiment_id, status="partial")


@pytest.mark.parametrize("failure_status", ["retryable_error", "needs_user_input"])
def test_pending_failure_can_escalate_to_terminal_in_the_same_role(manager, failure_status):
    experiment_id = f"exp_abandon_{failure_status}"
    manager.create_experiment("abandon", experiment_id=experiment_id)
    with manager.run(
        experiment_id,
        agent_name="qsaria_curation",
        tool_name="qsaria_curation_identify_qsar_columns",
    ) as runtime:
        runtime.capture_error("cannot continue", status=failure_status)

    recorded = manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_curation",
            status="terminal_failure",
            summary="The coordinator explicitly abandons this path",
            blockers=["cannot continue", "explicitly abandoned"],
        ),
    )
    assert recorded["status"] == "terminal_failure"
    finalized = manager.complete_experiment(experiment_id, status="terminal_failure")
    assert finalized["status"] == "terminal_failure"


def test_interrupted_call_requires_handoff_before_the_single_retry(manager):
    experiment_id = "exp_interrupted_call_123"
    manager.create_experiment("recover", experiment_id=experiment_id)
    _persist_interrupted_call(manager, experiment_id)

    restarted = ExperimentManager(local_root=manager._local_root())
    recovered = restarted.get_experiment_state(experiment_id)
    assert recovered["status"] == "retryable_error"
    assert any("interrupted Qsaria call" in item for item in recovered["blockers"])
    runtime_payload = json.loads(
        _state_path(restarted, experiment_id, RUNTIME_STATE_REL_PATH).read_text()
    )
    assert "active_call" not in runtime_payload
    assert runtime_payload["pending_failure"]["source"] == "interrupted_call"
    assert runtime_payload["pending_failure"]["handoff_recorded"] is False

    with pytest.raises(ExperimentStateTransitionError, match="structured agent handoff"):
        with restarted.run(
            experiment_id,
            agent_name="qsaria_training",
            tool_name="qsaria_training_train_qsar_model",
        ):
            pass

    restarted.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_training",
            status="retryable_error",
            summary="The interrupted call can be retried once",
        ),
    )
    with restarted.run(
        experiment_id,
        agent_name="qsaria_training",
        tool_name="qsaria_training_train_qsar_model",
    ) as runtime:
        runtime.capture_result({"status": "completed"})
    assert restarted.get_experiment_state(experiment_id)["status"] == "active"


def test_crash_during_single_retry_recovers_as_terminal(manager):
    experiment_id = "exp_interrupted_retry_123"
    manager.create_experiment("recover exhausted retry", experiment_id=experiment_id)
    _persist_interrupted_call(manager, experiment_id, automatic_retry=True)

    restarted = ExperimentManager(local_root=manager._local_root())
    recovered = restarted.get_experiment_state(experiment_id)
    assert recovered["status"] == "terminal_failure"
    assert any("automatic retry is exhausted" in item for item in recovered["blockers"])
    runtime_payload = json.loads(
        _state_path(restarted, experiment_id, RUNTIME_STATE_REL_PATH).read_text()
    )
    assert runtime_payload["pending_failure"]["automatic_retries"] == 1
    assert runtime_payload["pending_failure"]["status"] == "terminal_failure"

    restarted.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_training",
            status="terminal_failure",
            summary="The only retry was interrupted",
        ),
    )
    restarted.complete_experiment(experiment_id, status="terminal_failure")


def test_report_handoff_requires_exact_saved_artifact(manager):
    experiment_id = "exp_report_claim_123"
    manager.create_experiment("report", experiment_id=experiment_id)
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_curation",
            status="completed",
            summary="Scientific evidence ready",
        ),
    )
    manager.save_report(experiment_id, "# Report\n")

    with pytest.raises(InvalidHandoffError, match="must claim exactly"):
        manager.record_handoff(
            experiment_id,
            _handoff(
                experiment_id,
                agent="qsaria_report",
                status="completed",
                summary="Missing report claim",
            ),
        )


def test_deleted_report_is_rejected_at_handoff_and_completion(manager):
    before_handoff_id = "exp_report_deleted_before_123"
    manager.create_experiment("report", experiment_id=before_handoff_id)
    manager.record_handoff(
        before_handoff_id,
        _handoff(
            before_handoff_id,
            agent="qsaria_curation",
            status="completed",
            summary="Evidence ready",
        ),
    )
    report = manager.save_report(before_handoff_id, "# Report\n")
    Path(manager.resolve_artifact_path(before_handoff_id, report["path"])).unlink()
    with pytest.raises(ExperimentStateTransitionError, match="missing or has an unexpected size"):
        manager.record_handoff(
            before_handoff_id,
            _handoff(
                before_handoff_id,
                agent="qsaria_report",
                status="completed",
                summary="Deleted report",
                artifact_ids=[report["artifact_id"]],
            ),
        )

    before_completion_id = "exp_report_deleted_after_123"
    manager.create_experiment("report", experiment_id=before_completion_id)
    manager.record_handoff(
        before_completion_id,
        _handoff(
            before_completion_id,
            agent="qsaria_curation",
            status="completed",
            summary="Evidence ready",
        ),
    )
    report = manager.save_report(before_completion_id, "# Report\n")
    manager.record_handoff(
        before_completion_id,
        _handoff(
            before_completion_id,
            agent="qsaria_report",
            status="completed",
            summary="Report verified",
            artifact_ids=[report["artifact_id"]],
        ),
    )
    Path(manager.resolve_artifact_path(before_completion_id, report["path"])).unlink()
    with pytest.raises(ExperimentStateTransitionError, match="missing or has an unexpected size"):
        manager.complete_experiment(before_completion_id)


def test_report_partial_is_sticky_and_report_pause_statuses_are_rejected(manager):
    partial_id = "exp_report_partial_123"
    manager.create_experiment("partial report", experiment_id=partial_id)
    manager.record_handoff(
        partial_id,
        _handoff(
            partial_id,
            agent="qsaria_curation",
            status="completed",
            summary="Evidence ready",
        ),
    )
    report = manager.save_report(partial_id, "# Partial report\n")
    manager.record_handoff(
        partial_id,
        _handoff(
            partial_id,
            agent="qsaria_report",
            status="partial",
            summary="Report completed with a limitation",
            artifact_ids=[report["artifact_id"]],
            warnings=["One optional section is unavailable"],
        ),
    )
    assert manager.get_experiment_state(partial_id)["status"] == "partial"
    with pytest.raises(ExperimentStateTransitionError, match="partial scientific"):
        manager.complete_experiment(partial_id)
    assert manager.complete_experiment(partial_id, status="partial")["status"] == "partial"

    for report_status in ("retryable_error", "needs_user_input"):
        experiment_id = f"exp_report_disallowed_{report_status}"
        manager.create_experiment("report pause", experiment_id=experiment_id)
        manager.record_handoff(
            experiment_id,
            _handoff(
                experiment_id,
                agent="qsaria_curation",
                status="completed",
                summary="Evidence ready",
            ),
        )
        with pytest.raises(InvalidHandoffError, match="cannot pause or retry"):
            manager.record_handoff(
                experiment_id,
                _handoff(
                    experiment_id,
                    agent="qsaria_report",
                    status=report_status,
                    summary="Report could not finish",
                ),
            )


@pytest.mark.parametrize("science_terminal", [False, True])
def test_terminal_report_handoff_can_finalize_with_or_without_prior_science_failure(
    manager,
    science_terminal,
):
    experiment_id = f"exp_terminal_report_{science_terminal}"
    manager.create_experiment("terminal report", experiment_id=experiment_id)
    if science_terminal:
        with manager.run(
            experiment_id,
            agent_name="qsaria_training",
            tool_name="qsaria_training_train_qsar_model",
        ) as runtime:
            runtime.capture_error("scientific failure", status="terminal_failure")
        manager.record_handoff(
            experiment_id,
            _handoff(
                experiment_id,
                agent="qsaria_training",
                status="terminal_failure",
                summary="Scientific failure",
            ),
        )
    else:
        manager.record_handoff(
            experiment_id,
            _handoff(
                experiment_id,
                agent="qsaria_curation",
                status="completed",
                summary="Evidence ready",
            ),
        )
    report = manager.save_report(experiment_id, "# Unvalidated report evidence\n")
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_report",
            status="terminal_failure",
            summary="Report verification failed",
            artifact_ids=[report["artifact_id"]],
        ),
    )
    finalized = manager.complete_experiment(experiment_id, status="terminal_failure")
    assert finalized["status"] == "terminal_failure"
    assert finalized["report"]["validation_status"] == "terminal_failure"


def test_report_transaction_recovers_after_bytes_are_written(manager, monkeypatch):
    experiment_id = "exp_report_recovery_123"
    manager.create_experiment("report crash", experiment_id=experiment_id)
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_curation",
            status="completed",
            summary="Evidence ready",
        ),
    )
    original_save = manager._save_states
    calls = 0

    def fail_final_state_save(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated crash after report bytes")
        return original_save(*args, **kwargs)

    monkeypatch.setattr(manager, "_save_states", fail_final_state_save)
    with pytest.raises(RuntimeError, match="simulated crash"):
        manager.save_report(experiment_id, "# Durable report\n")
    monkeypatch.setattr(manager, "_save_states", original_save)

    restarted = ExperimentManager(local_root=manager._local_root())
    recovered = restarted.get_experiment_state(experiment_id)
    assert recovered["report"]["content_sha256"]
    runtime_payload = json.loads(
        _state_path(restarted, experiment_id, RUNTIME_STATE_REL_PATH).read_text()
    )
    assert "pending_report" not in runtime_payload


def test_report_transaction_abandons_intent_when_no_bytes_exist(manager, monkeypatch):
    experiment_id = "exp_report_no_bytes_123"
    manager.create_experiment("report crash", experiment_id=experiment_id)
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_curation",
            status="completed",
            summary="Evidence ready",
        ),
    )
    original_write = manager._write_text

    def fail_report_write(experiment, rel_path, text):
        if rel_path.startswith("workflows/reports/"):
            raise RuntimeError("simulated crash before report bytes")
        return original_write(experiment, rel_path, text)

    monkeypatch.setattr(manager, "_write_text", fail_report_write)
    with pytest.raises(RuntimeError, match="before report bytes"):
        manager.save_report(experiment_id, "# Missing report\n")
    monkeypatch.setattr(manager, "_write_text", original_write)

    restarted = ExperimentManager(local_root=manager._local_root())
    assert restarted.get_experiment_state(experiment_id)["report"] is None
    runtime_payload = json.loads(
        _state_path(restarted, experiment_id, RUNTIME_STATE_REL_PATH).read_text()
    )
    assert "pending_report" not in runtime_payload
    assert restarted.save_report(experiment_id, "# Retried report\n")["artifact_id"]


def test_split_state_transaction_recovers_public_from_newer_runtime_snapshot(
    manager,
    monkeypatch,
):
    experiment_id = "exp_split_state_recovery_123"
    manager.create_experiment("state transaction", experiment_id=experiment_id)
    with manager._locked(experiment_id):
        public_state, runtime_state = manager._load_states(experiment_id)
        public_state["status"] = "active"
        public_state["phase"] = "recovered_phase"
        original_write_json = manager._write_json

        def fail_public_write(experiment, rel_path, payload):
            if rel_path == PUBLIC_STATE_REL_PATH:
                raise RuntimeError("simulated crash between state documents")
            return original_write_json(experiment, rel_path, payload)

        monkeypatch.setattr(manager, "_write_json", fail_public_write)
        with pytest.raises(RuntimeError, match="between state documents"):
            manager._save_states(experiment_id, public_state, runtime_state)
        monkeypatch.setattr(manager, "_write_json", original_write_json)

    restarted = ExperimentManager(local_root=manager._local_root())
    recovered = restarted.get_experiment_state(experiment_id)
    runtime_payload = json.loads(
        _state_path(restarted, experiment_id, RUNTIME_STATE_REL_PATH).read_text()
    )
    assert recovered["status"] == "active"
    assert recovered["phase"] == "recovered_phase"
    assert recovered["state_revision"] == runtime_payload["state_revision"]


def test_public_state_newer_than_runtime_fails_closed(manager):
    experiment_id = "exp_public_ahead_123"
    manager.create_experiment("state inconsistency", experiment_id=experiment_id)
    public_path = _state_path(manager, experiment_id, PUBLIC_STATE_REL_PATH)
    public_payload = json.loads(public_path.read_text())
    public_payload["state_revision"] += 1
    public_payload["status"] = "completed"
    public_path.write_text(json.dumps(public_payload, indent=2) + "\n")

    restarted = ExperimentManager(local_root=manager._local_root())
    with pytest.raises(ExperimentStateTransitionError, match="public experiment state is newer"):
        restarted.get_experiment_state(experiment_id)


def test_session_symlinks_cannot_cross_experiments_or_escape_report_writes(manager, tmp_path):
    target_id = "exp_symlink_target_123"
    manager.create_experiment("target", experiment_id=target_id)
    malicious_id = "exp_symlink_alias_123"
    alias_root = manager._local_root() / "sessions" / malicious_id
    alias_root.symlink_to(manager._session_root(target_id), target_is_directory=True)
    with pytest.raises(InvalidInputPathError, match="symbolic-link session root"):
        manager.create_experiment("alias", experiment_id=malicious_id)

    report_id = "exp_symlink_report_123"
    manager.create_experiment("report", experiment_id=report_id)
    manager.record_handoff(
        report_id,
        _handoff(
            report_id,
            agent="qsaria_curation",
            status="completed",
            summary="Evidence ready",
        ),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (manager._session_root(report_id) / "workflows").symlink_to(
        outside,
        target_is_directory=True,
    )
    with pytest.raises(InvalidInputPathError, match="symbolic-link component"):
        manager.save_report(report_id, "# Must remain isolated\n")
    assert list(outside.iterdir()) == []
