#!/usr/bin/env python
# coding: utf-8
"""Tests for deterministic Agno QSAR handoffs and report context compaction."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cs_copilot.agents.factories import QSARReportFactory, QSARTrainingFactory
from cs_copilot.agents.qsar_agent_handoffs import (
    QsarTrainingAgentHandoff,
    TrainingPersistenceEvidence,
    build_qsar_training_agent_handoff,
    compact_qsar_report_session_context,
    finalize_qsar_training_agent_output,
)


def _execution(name, payload, *, failed=False):
    result = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(tool_name=name, result=result, tool_call_error=failed)


def test_training_handoff_preserves_canonical_facts_tables_and_all_models(tmp_path):
    bundle = tmp_path / "training_bundle.zip"
    bundle.write_bytes(b"zip")
    summary = tmp_path / "training_summary.json"
    summary.write_text("{}")
    model_root_a = tmp_path / "model-a"
    model_root_b = tmp_path / "model-b"
    model_root_a.mkdir()
    model_root_b.mkdir()
    model_a = model_root_a / "best.pkl"
    model_b = model_root_b / "best.pkl"
    metadata_a = model_root_a / "metadata.json"
    metadata_b = model_root_b / "metadata.json"
    for path in (model_a, model_b, metadata_a, metadata_b):
        path.write_text("{}")

    training_payload = {
        "backend_name": "lightgbm",
        "persistence_plan": {"persist_all_candidates": True, "candidate_count": 2},
        "bundle_file_ref": str(bundle),
        "summary_path": str(summary),
        "resolved_plan": {"runtime_paths": {"bundle_path": str(tmp_path / "planned.zip")}},
        "reporting_handoff": {
            "report_facts": {
                "schema_version": "2.0",
                "report_kind": "training",
                "governance": {"catalog_model_policy": "outlier_variants_no_test_winner"},
            },
            "report_tables": {
                "final_test_comparison": {
                    "markdown": "| variant | R2 |\n|---|---:|\n| baseline | 0.5 |",
                }
            },
            "decision_summary": "This legacy narrative is not forwarded.",
        },
    }
    persisted = {
        "persisted": True,
        "candidate_count": 2,
        "candidates": [
            {
                "model_id": "model-a",
                "model_root": str(model_root_a),
                "model_path": str(model_a),
                "metadata_path": str(metadata_a),
            },
            {
                "model_id": "model-b",
                "model_root": str(model_root_b),
                "model_path": str(model_b),
                "metadata_path": str(metadata_b),
            },
        ],
    }

    handoff = build_qsar_training_agent_handoff(
        [
            _execution("train_standard_qsar_model", training_payload),
            _execution("register_and_persist_candidates", persisted),
        ]
    )

    assert handoff is not None
    assert handoff.status == "completed"
    assert handoff.report_facts == training_payload["reporting_handoff"]["report_facts"]
    assert handoff.report_tables == training_payload["reporting_handoff"]["report_tables"]
    assert [model.model_id for model in handoff.persistence.models] == ["model-a", "model-b"]
    assert handoff.persistence.status == "completed"
    assert str(tmp_path / "planned.zip") not in handoff.model_dump_json()
    assert "legacy narrative" not in handoff.model_dump_json()
    assert str(bundle) in {artifact.path for artifact in handoff.artifacts}


def test_training_post_hook_replaces_agent_narrative_and_records_shared_handoff(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text("{}")
    executions = [
        _execution(
            "train_lightgbm_model",
            {
                "summary_path": str(summary),
                "reporting_handoff": {
                    "report_facts": {"schema_version": "2.0", "report_kind": "training"},
                    "report_tables": {},
                },
            },
        )
    ]
    run_output = SimpleNamespace(
        tools=executions,
        content="A polished but lossy narrative",
        content_type="str",
    )
    session_state = {"prediction_models": {"registered": {"large": "payload"}}}

    finalize_qsar_training_agent_output(run_output, session_state)

    payload = json.loads(run_output.content)
    assert payload["agent"] == "qsar_training"
    assert payload["report_facts"]["report_kind"] == "training"
    assert "lossy narrative" not in run_output.content
    assert run_output.content_type == "QsarTrainingAgentHandoff"
    assert session_state["prediction_models"]["latest_training_handoff"] == payload


def test_training_handoff_preserves_exact_terminal_tool_failure():
    handoff = build_qsar_training_agent_handoff(
        [
            _execution(
                "train_lightgbm_model",
                "LightGBM failed with exact backend error",
                failed=True,
            )
        ]
    )

    assert handoff is not None
    assert handoff.status == "terminal_failure"
    assert handoff.blockers == ("LightGBM failed with exact backend error",)
    assert handoff.report_facts == {}


def test_report_context_excludes_registered_model_bulk_and_history():
    handoff = {
        "schema_version": "1.0",
        "agent": "qsar_training",
        "status": "completed",
        "report_facts": {"report_kind": "training"},
    }
    state = {
        "current_run_id": "run-1",
        "REPORT_LANGUAGE": "fr",
        "qsar_curation": {
            "last_request": {"large": "x" * 20_000},
            "last_result": {"rows_in": 100, "rows_out": 90},
            "history": [{"large": "x" * 20_000}],
        },
        "prediction_models": {
            "registered": {"model-a": {"metadata": "x" * 100_000}},
            "training_runs": [{"large": "x" * 20_000}],
            "latest_training_handoff": handoff,
            "last_prediction": {"model_id": "model-a", "row_count": 3},
            "prediction_history": [{"model_id": "model-a", "row_count": 3}],
        },
        "prediction_outputs": {"latest_summary": "/tmp/predictions.csv"},
    }

    compact_qsar_report_session_context(state)

    assert state["REPORT_LANGUAGE"] == "fr"
    assert state["qsar_curation"] == {"last_result": {"rows_in": 100, "rows_out": 90}}
    assert "registered" not in state["prediction_models"]
    assert "training_runs" not in state["prediction_models"]
    assert state["prediction_models"]["latest_training_handoff"] == handoff
    assert len(json.dumps(state)) < 5_000


def test_qsar_factories_install_only_the_role_specific_hooks():
    training = QSARTrainingFactory().get_agent_config()
    report = QSARReportFactory().get_agent_config()

    assert training.post_hooks == [finalize_qsar_training_agent_output]
    assert training.pre_hooks == []
    assert report.pre_hooks == [compact_qsar_report_session_context]
    assert report.post_hooks == []


def test_training_handoff_schema_forbids_unknown_fields_recursively():
    schema = QsarTrainingAgentHandoff.model_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["TrainingPersistenceEvidence"]["additionalProperties"] is False
    assert schema["$defs"]["PersistedModelEvidence"]["additionalProperties"] is False
    assert schema["$defs"]["TrainingArtifactEvidence"]["additionalProperties"] is False

    with pytest.raises(ValidationError, match="extra_forbidden"):
        QsarTrainingAgentHandoff(
            status="completed",
            summary="done",
            persistence=TrainingPersistenceEvidence(status="not_required"),
            invented_field="not allowed",
        )
