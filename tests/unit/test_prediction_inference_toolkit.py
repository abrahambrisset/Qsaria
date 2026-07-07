from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from cs_copilot.tools.prediction.backend import PredictionModelRecord, PredictionTaskSpec
from cs_copilot.tools.prediction.prediction_inference_toolkit import PredictionInferenceToolkit


class _Catalog:
    def refresh_from_internal_store(self, *, persist=False):
        return None


class _Registry:
    def __init__(self, record):
        self.record = record
        self.catalog = _Catalog()

    def resolve_record(self, model_id, agent):
        assert model_id == self.record.model_id
        return self.record


class _Backend:
    backend_name = "mock"

    def predict_from_csv(self, input_csv, model_record, preds_path, *, return_uncertainty=False):
        df = pd.read_csv(input_csv)
        pd.DataFrame({"smiles": df["smiles"], "prediction": [1.0] * len(df)}).to_csv(
            preds_path,
            index=False,
        )
        return {"preds_path": preds_path}


def _record(tmp_path: Path) -> PredictionModelRecord:
    model_root = tmp_path / "model"
    model_root.mkdir()
    model_path = model_root / "best.pt"
    model_path.write_text("mock")
    metadata_path = model_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_id": "pxr_model",
                "backend_name": "mock",
                "task": {
                    "task_type": "regression",
                    "smiles_columns": ["smiles"],
                    "target_columns": ["pEC50"],
                },
                "artifacts": {"model_path": "best.pt"},
            }
        )
    )
    return PredictionModelRecord(
        model_id="pxr_model",
        backend_name="mock",
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        task=PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=["pEC50"],
        ),
    )


def test_failed_external_evaluation_blocks_implicit_blind_prediction(tmp_path):
    record = _record(tmp_path)
    toolkit = PredictionInferenceToolkit(
        backends={"mock": _Backend()},
        registry_toolkit=_Registry(record),
        register_tools=False,
    )
    agent = SimpleNamespace(session_state={})
    blinded_csv = tmp_path / "blinded.csv"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(blinded_csv, index=False)

    try:
        toolkit.evaluate_model_on_dataset(
            model_id="pxr_model",
            test_csv=str(blinded_csv),
            smiles_column="SMILES",
            target_columns=["pEC50"],
            agent=agent,
        )
    except ValueError as exc:
        assert "missing required target columns" in str(exc)
    else:
        raise AssertionError("missing target columns should fail")

    blocked = toolkit.predict_from_csv(
        model_id="pxr_model",
        input_csv=str(blinded_csv),
        smiles_column="SMILES",
        preds_path=str(tmp_path / "blocked.csv"),
        agent=agent,
    )
    assert blocked["status"] == "blocked_failed_external_evaluation"
    assert blocked["prediction_generated"] is False
    assert not (tmp_path / "blocked.csv").exists()
    assert agent.session_state["prediction_models"]["prediction_history"] == []

    fresh_csv = tmp_path / "fresh_blind_prediction.csv"
    pd.DataFrame({"SMILES": ["CCN"]}).to_csv(fresh_csv, index=False)
    result = toolkit.predict_from_csv(
        model_id="pxr_model",
        input_csv=str(fresh_csv),
        smiles_column="SMILES",
        preds_path=str(tmp_path / "allowed.csv"),
        agent=agent,
    )
    assert Path(result["preds_path"]).exists()
