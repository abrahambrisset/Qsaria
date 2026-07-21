from cs_copilot.agents.factories import QSARTrainingFactory
from cs_copilot.tools.features.molecular_feature_toolkit import MolecularFeatureToolkit
from cs_copilot.tools.prediction.chemprop_toolkit import ChempropToolkit
from cs_copilot.tools.prediction.lightgbm_toolkit import LightGBMToolkit
from cs_copilot.tools.prediction.qsar_training_toolkit import QSARTrainingToolkit
from cs_copilot.tools.prediction.tabicl_toolkit import TabICLToolkit


def test_training_agent_exposes_only_the_qsar_training_facade_for_backend_training():
    tools = QSARTrainingFactory().get_agent_config().tools

    assert sum(isinstance(tool, QSARTrainingToolkit) for tool in tools) == 1
    assert not any(isinstance(tool, MolecularFeatureToolkit) for tool in tools)
    assert not any(
        isinstance(tool, (ChempropToolkit, LightGBMToolkit, TabICLToolkit))
        for tool in tools
    )
