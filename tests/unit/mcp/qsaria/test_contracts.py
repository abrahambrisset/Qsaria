"""Contract tests for Qsaria experiment and sub-agent envelopes."""

from __future__ import annotations

from pathlib import Path

import pytest

from cs_copilot.mcp.qsaria.contracts import (
    HANDOFF_EXECUTION_MODES,
    HANDOFF_SCHEMA_VERSION,
    InvalidExperimentIdError,
    InvalidHandoffError,
    artifact_id_for,
    extract_artifact_paths,
    extract_model_ids,
    generate_experiment_id,
    normalize_handoff,
    validate_experiment_id,
)


def _valid_handoff(**overrides):
    payload = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "experiment_id": "exp_contract_123",
        "agent": "qsaria_training",
        "status": "completed",
        "summary": "ok",
        "facts": {},
        "artifact_ids": [],
        "model_ids": [],
        "warnings": [],
        "blockers": [],
        "recommended_next_action": "",
    }
    payload.update(overrides)
    return payload


def test_generated_experiment_id_is_valid_and_path_safe():
    experiment_id = generate_experiment_id()
    assert experiment_id.startswith("exp_")
    assert validate_experiment_id(experiment_id) == experiment_id
    assert "/" not in experiment_id


@pytest.mark.parametrize(
    "value",
    ["", "latest", "exp_x", "exp_../../secret", "exp_with/slash", " exp_ok_123 "],
)
def test_invalid_or_noncanonical_experiment_ids_are_rejected(value):
    with pytest.raises(InvalidExperimentIdError):
        validate_experiment_id(value)


def test_handoff_contract_is_normalized_without_losing_fact_first_payload():
    handoff = normalize_handoff(
        "exp_contract_123",
        _valid_handoff(
            summary="Training complete",
            facts={"report_facts": {"schema_version": "2.0", "metric": 0.91}},
            model_ids=["model-a", "model-a"],
            warnings=["small sample"],
        ),
    )

    assert handoff["schema_version"] == HANDOFF_SCHEMA_VERSION
    assert handoff["execution_mode"] == "project_agent"
    assert handoff["facts"]["report_facts"]["schema_version"] == "2.0"
    assert handoff["model_ids"] == ["model-a"]
    assert handoff["blockers"] == []


def test_handoff_records_coordinator_manual_tool_recovery() -> None:
    handoff = normalize_handoff(
        "exp_contract_123",
        _valid_handoff(execution_mode="coordinator_manual_tools"),
    )

    assert HANDOFF_EXECUTION_MODES == {
        "project_agent",
        "coordinator_manual_tools",
    }
    assert handoff["agent"] == "qsaria_training"
    assert handoff["execution_mode"] == "coordinator_manual_tools"


@pytest.mark.parametrize(
    "patch",
    [
        {"agent": "unknown"},
        {"status": "invented"},
        {"summary": ""},
        {"experiment_id": "exp_other_123"},
        {"experiment_id": ""},
        {"schema_version": "99.0"},
        {"facts": ["not", "an", "object"]},
        {"facts": None},
        {"artifact_ids": "artifact"},
        {"model_ids": [1]},
        {"warnings": None},
        {"blockers": "blocked"},
        {"recommended_next_action": 1},
        {"summary": 1},
        {"execution_mode": "source_patch"},
        {"execution_mode": 1},
    ],
)
def test_invalid_handoff_fields_are_rejected(patch):
    payload = _valid_handoff(**patch)
    with pytest.raises(InvalidHandoffError):
        normalize_handoff("exp_contract_123", payload)


def test_handoff_requires_the_complete_schema_and_failure_blockers():
    payload = _valid_handoff()
    payload.pop("artifact_ids")
    with pytest.raises(InvalidHandoffError, match="missing required field"):
        normalize_handoff("exp_contract_123", payload)

    with pytest.raises(InvalidHandoffError, match="requires at least one blocker"):
        normalize_handoff(
            "exp_contract_123",
            _valid_handoff(status="terminal_failure", blockers=[]),
        )


def test_nested_artifact_and_model_identifiers_are_extracted():
    payload = {
        "status": "completed",
        "artifacts": {
            "metrics_json": Path("workflows/exp_x/results/metrics.json"),
            "plot_artifacts": ["workflows/exp_x/results/parity.png"],
        },
        "registry": {
            "model_id": "model-primary",
            "registered_model_ids": ["model-primary", "model-secondary"],
        },
        "message": "not/a/path because the key is not an artifact field",
    }

    assert extract_artifact_paths(payload) == [
        "workflows/exp_x/results/metrics.json",
        "workflows/exp_x/results/parity.png",
    ]
    assert extract_model_ids(payload) == ["model-primary", "model-secondary"]
    assert artifact_id_for("exp_contract_123", "result.json").startswith("art_")
    assert artifact_id_for("exp_contract_123", "result.json") == artifact_id_for(
        "exp_contract_123", "result.json"
    )
