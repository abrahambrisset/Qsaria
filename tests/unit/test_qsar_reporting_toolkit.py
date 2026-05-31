from types import SimpleNamespace

import pandas as pd

import cs_copilot.tools.reporting.qsar_reporting_toolkit as report_module
from cs_copilot.tools.reporting.qsar_reporting_toolkit import QSARReportingToolkit


def test_prediction_report_payload_includes_multitask_prediction_columns(tmp_path, monkeypatch):
    preds_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCC"],
            "pEC50_prediction": [6.1, 5.9],
            "solubility_prediction": [1.2, 1.4],
            "ad_status": ["in_domain", "edge_of_domain"],
        }
    ).to_csv(preds_path, index=False)

    record = SimpleNamespace(
        task=SimpleNamespace(
            target_columns=["pEC50", "solubility"],
            task_type="regression",
        ),
        display_name="multi-task model",
        status="experimental",
        backend_name="chemprop",
        metadata_path=None,
        known_metrics={},
        training_data_summary={},
    )
    fake_catalog = SimpleNamespace(
        models={"chemprop_multi": record},
        refresh_from_internal_store=lambda persist=False: None,
    )
    monkeypatch.setattr(
        report_module.PredictionModelCatalog,
        "load",
        staticmethod(lambda: fake_catalog),
    )
    agent = SimpleNamespace(
        session_state={
            "prediction_models": {
                "prediction_history": [
                    {
                        "model_id": "chemprop_multi",
                        "backend_name": "chemprop",
                        "preds_path": str(preds_path),
                        "applicability_domain_columns": [],
                        "applicability_domain": {},
                    }
                ]
            }
        }
    )

    payload = QSARReportingToolkit().build_prediction_report_payload(agent=agent)

    model_section = next(section for section in payload["sections"] if section["title"] == "Modele utilise")
    model_items = model_section["blocks"][0]["items"]
    assert ["Cibles", "pEC50, solubility"] in model_items

    prediction_section = next(
        section for section in payload["sections"] if section["title"] == "Resultats des predictions"
    )
    prediction_table = next(
        block for block in prediction_section["blocks"] if block["type"] == "table"
    )
    assert prediction_table["columns"] == [
        "SMILES",
        "Y predit (pEC50)",
        "Y predit (solubility)",
        "Statut AD",
        "Fiabilite",
    ]
