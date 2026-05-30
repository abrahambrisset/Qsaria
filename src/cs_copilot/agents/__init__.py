#!/usr/bin/env python
# coding: utf-8
"""
Cs_copilot Agents Package

This package provides a comprehensive system for creating and managing
AI agents specialized in cheminformatics tasks.

Public API:
-----------

Agent Creation (Recommended):
    create_agent(agent_type, model, **kwargs) - Create agents by type
    list_available_agent_types() - List all available agent types

Team Coordination:
    get_cs_copilot_agent_team(model, **kwargs) - Multi-agent team with intelligent coordination
    get_qsar_agent_team(model, **kwargs) - Isolated QSAR-only multi-agent team

Utilities:
    get_last_agent_reply(agent) - Extract last message from agent

Exceptions:
    AgentCreationError - Raised when agent creation fails

Available Agent Types (5-Agent Architecture):
----------------------------------------------
Core Agents:
- "chembl_downloader" - Download and process bioactivity data from ChEMBL database
- "dataset_curation" - Prepare QSAR-ready datasets via isolated curation workflows
- "qsar_training" - Train and evaluate QSAR models on curated datasets
- "model_registry" - Apply model governance and persist QSAR models to the catalog
- "model_inference" - Select registered QSAR models and run predictions
- "qsar_report" - Draft the only final user-facing report for QSAR workflows
- "gtm_agent" - Unified GTM operations (build, load, density, activity, project) with smart caching
- "chemoinformatician" - Comprehensive chemoinformatics (chemotype, clustering, SAR, similarity, QSAR)
- "report_generator" - Universal presentation layer for all analysis types
- "autoencoder" - Molecular generation via LSTM autoencoders (standalone + GTM-guided)

Testing/Evaluation:
- "robustness_evaluation" - Analyze robustness test results and metrics

Agent Capabilities Breakdown:
-----------------------------
**Chemoinformatician** (Most Versatile):
  - Chemotype/Scaffold Analysis: Extract and analyze molecular frameworks
  - Clustering: Group molecules by structural similarity (k-means, hierarchical, DBSCAN)
  - SAR Analysis: Structure-Activity Relationships, activity cliffs, matched molecular pairs
  - Similarity/Diversity: Molecular similarity, diversity metrics, nearest neighbors
  - QSAR-adjacent analysis: exploratory structural analyses that can support QSAR work, without replacing the isolated QSAR system
"""

_FACTORY_EXPORTS = {"AgentConfig", "AgentCreationError", "BaseAgentFactory"}
_REGISTRY_EXPORTS = {"create_agent", "get_registry", "list_available_agent_types"}
_TEAM_EXPORTS = {"get_cs_copilot_agent_team", "get_qsar_agent_team"}
_UTIL_EXPORTS = {"get_last_agent_reply"}


def __getattr__(name: str):
    if name in _FACTORY_EXPORTS:
        from . import factories as module
    elif name in _REGISTRY_EXPORTS:
        from . import registry as module
    elif name in _TEAM_EXPORTS:
        from . import teams as module
    elif name in _UTIL_EXPORTS:
        from . import utils as module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    # Primary API
    "create_agent",
    "list_available_agent_types",
    "get_registry",
    # Team coordination
    "get_cs_copilot_agent_team",
    "get_qsar_agent_team",
    # Utilities
    "get_last_agent_reply",
    # Configuration and exceptions
    "AgentCreationError",
    "AgentConfig",
    "BaseAgentFactory",
]
