import json
from pathlib import Path

from cs_copilot.tools.prediction.qsar_progress import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    apply_progress_update,
    completed_phase_for_tool,
    format_elapsed_seconds,
    load_active_run_snapshot,
    phase_for_tool,
    render_status_card,
)


def test_status_card_is_compact_and_english_only():
    card = render_status_card(
        status=STATUS_RUNNING,
        elapsed_seconds=130,
        backend_name="lightgbm",
        phase="Hyperparameter optimization",
        detail="trial 12 of 50",
    )

    assert card == (
        "QSAR workflow\n"
        "Running · LightGBM\n"
        "Hyperparameter optimization — trial 12 of 50\n"
        "Elapsed: 2m 10s"
    )
    assert "Toujours actif" not in card
    assert "validation=" not in card


def test_status_card_keeps_compact_completed_and_failed_states():
    assert "Completed · Chemprop" in render_status_card(
        status=STATUS_COMPLETED,
        elapsed_seconds=3,
        backend_name="chemprop",
        phase="Writing training artifacts",
    )
    assert "Failed · TabICL" in render_status_card(
        status=STATUS_FAILED,
        elapsed_seconds=3,
        backend_name="tabicl",
        phase="Training model",
    )


def test_tool_phase_mapping_covers_qsaria_workflow_without_leaking_unknown_tools():
    assert phase_for_tool("curate_qsar_dataset") == "Curating dataset"
    assert phase_for_tool("train_lightgbm_model") == "Preparing LightGBM training"
    assert phase_for_tool("persist_registered_model") == "Persisting model artifacts"
    assert phase_for_tool("predict_from_csv") == "Generating predictions"
    assert completed_phase_for_tool("predict_from_csv") == "Predictions generated"
    assert phase_for_tool("prepare_training_dataset") is None
    assert phase_for_tool("not_a_qsaria_tool") is None


def test_all_training_backends_and_chainlit_use_the_single_canonical_progress_state():
    project_root = Path(__file__).resolve().parents[2]
    for relative_path in (
        "src/cs_copilot/tools/prediction/chemprop_toolkit.py",
        "src/cs_copilot/tools/prediction/lightgbm_toolkit.py",
        "src/cs_copilot/tools/prediction/tabicl_toolkit.py",
    ):
        source = (project_root / relative_path).read_text()
        assert 'prediction_state["active_training_run"]' in source
        assert 'session_state["qsar_training"]' not in source

    chainlit_source = (project_root / "chainlit_app.py").read_text()
    assert 'prediction_state.get("active_training_run")' in chainlit_source
    assert 'session_state.get("qsar_training")' not in chainlit_source


def test_active_marker_snapshot_overrides_stale_in_memory_progress(tmp_path):
    marker_path = tmp_path / ".training_in_progress"
    marker_path.write_text(
        json.dumps(
            {
                "phase": "Training model",
                "progress_message": "run 1 of 1",
                "train_rows": 80,
                "val_rows": 10,
                "test_rows": 10,
            }
        )
    )

    snapshot = load_active_run_snapshot(
        {
            "backend_name": "tabicl",
            "phase": "Preparing TabICL training",
            "active_marker_path": str(marker_path),
        }
    )

    assert snapshot["backend_name"] == "tabicl"
    assert snapshot["phase"] == "Training model"
    assert snapshot["train_rows"] == 80
    assert format_elapsed_seconds(3_661) == "1h 1m 1s"


def test_shared_progress_update_uses_the_common_safe_shape():
    record = {"backend_name": "lightgbm", "phase": "Preparing model training"}

    updated = apply_progress_update(
        record,
        "Hyperparameter optimization",
        {"detail": "trial 3 of 50", "trial_index": 3, "ignored": None},
    )

    assert updated is record
    assert record == {
        "backend_name": "lightgbm",
        "phase": "Hyperparameter optimization",
        "progress_message": "trial 3 of 50",
        "trial_index": 3,
    }
