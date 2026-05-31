"""Tests for deterministic QSAR agent workflow contracts."""

from cs_copilot.agents.factories import (
    ModelInferenceFactory,
    QSARReportFactory,
    QSARTrainingFactory,
)
from cs_copilot.agents.prompts import MODEL_INFERENCE_INSTRUCTIONS, QSAR_REPORT_INSTRUCTIONS
from cs_copilot.agents.qsar_workflow import (
    QSARWorkflowKind,
    classify_qsar_workflow,
    copy_qsar_session_state,
    plan_qsar_workflow,
)


class SentinelTool:
    pass


class SentinelRegistryTool:
    pass


class SentinelPredictionTool:
    pass


class SentinelBenchmarkTool:
    pass


class SentinelEnsembleTool:
    pass


class SentinelReportingTool:
    pass


class FakeQSARContext:
    def __init__(self):
        self.training_toolkit = SentinelTool()
        self.reporting_toolkit = SentinelReportingTool()
        self.prediction_tools_calls = []

    def prediction_tools(self, *, include_inference: bool = False):
        self.prediction_tools_calls.append(include_inference)
        tools = [SentinelRegistryTool()]
        if include_inference:
            tools.append(SentinelPredictionTool())
        return tools

    def benchmark_toolkit(self):
        return SentinelBenchmarkTool()

    def ensemble_toolkit(self):
        return SentinelEnsembleTool()


def test_qsar_workflow_classifier_routes_core_requests():
    assert classify_qsar_workflow("curate this QSAR dataset").workflow == QSARWorkflowKind.CURATION_ONLY
    assert classify_qsar_workflow("train a LightGBM QSAR model").route == (
        "dataset_curation",
        "qsar_training",
        "model_registry",
        "qsar_report",
    )
    assert classify_qsar_workflow("predict lipophilicity for these SMILES").route == (
        "model_inference",
        "qsar_report",
    )
    assert classify_qsar_workflow("which QSAR backends are available?").route == (
        "model_registry",
        "qsar_report",
    )
    assert classify_qsar_workflow("cree un ensemble QSAR pour pEC50").workflow == QSARWorkflowKind.ENSEMBLE


def test_qsar_workflow_classifier_handles_export_only_latex_shortcut():
    plan = classify_qsar_workflow("@LaTeX")

    assert plan.workflow == QSARWorkflowKind.EXPORT_ONLY
    assert plan.route == ("qsar_report",)
    assert plan.export_only is True
    assert plan.rerun_prediction is False
    assert plan.report_language == "English"

    payload = plan_qsar_workflow("genere le payload latex pour la derniere prediction")
    assert payload["workflow"] == "export_only"
    assert payload["route"] == ["qsar_report"]
    assert payload["report_language"] == "French"


def test_qsar_session_state_defaults_are_independent_and_complete():
    first = copy_qsar_session_state()
    second = copy_qsar_session_state()

    first["prediction_models"]["registered"]["demo"] = {"model_id": "demo"}

    assert second["prediction_models"]["registered"] == {}
    for key in (
        "prediction_models",
        "prediction_outputs",
        "qsar_curation",
        "qsar_training",
        "qsar_registry",
        "qsar_inference",
        "qsar_report",
        "qsar_workflow",
    ):
        assert key in first


def test_qsar_factory_tool_boundaries_use_shared_context():
    context = FakeQSARContext()

    training_config = QSARTrainingFactory().get_agent_config(qsar_context=context)
    assert any(isinstance(tool, SentinelTool) for tool in training_config.tools)
    assert any(isinstance(tool, SentinelBenchmarkTool) for tool in training_config.tools)
    assert not any(isinstance(tool, SentinelRegistryTool) for tool in training_config.tools)

    inference_config = ModelInferenceFactory().get_agent_config(qsar_context=context)
    assert any(isinstance(tool, SentinelRegistryTool) for tool in inference_config.tools)
    assert any(isinstance(tool, SentinelPredictionTool) for tool in inference_config.tools)
    assert not any(isinstance(tool, SentinelReportingTool) for tool in inference_config.tools)

    report_config = QSARReportFactory().get_agent_config(qsar_context=context)
    assert any(isinstance(tool, SentinelReportingTool) for tool in report_config.tools)


def test_prompt_tool_ownership_is_consistent():
    assert not any("export_latest_prediction_report_bundle" in item for item in MODEL_INFERENCE_INSTRUCTIONS)
    assert any("QSARReportingToolkit" in item for item in QSAR_REPORT_INSTRUCTIONS)
