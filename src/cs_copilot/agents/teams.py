#!/usr/bin/env python
# coding: utf-8
"""
Team coordination functionality for multi-agent workflows.
"""

import logging
from typing import List, Tuple

from agno.db.sqlite import SqliteDb  # ✅ v2.1.x style DB import
from agno.models.base import Model  # Agno v2 base class
from agno.team import Team

from .config import CS_COPILOT_MEMORY_DB  # optional now; kept for compatibility
from .factories import AgentCreationError
from .prompts import AGENT_TEAM_INSTRUCTIONS
from .registry import create_agent


def get_cs_copilot_agent_team(
    model: Model,  # Agno Model instance, e.g. OpenAIChat(...) or Claude(...)
    *,
    markdown: bool = True,
    debug_mode: bool = False,
    show_members_responses: bool = True,
    enable_memory: bool = True,
    db_file: str = None,
    enable_mlflow_tracking: bool = True,
) -> Team:
    """
    Create a coordinated team of cs_copilot agents using Agno.

    Args:
        model: Agno Model instance used for team coordination and member agents
        markdown: Format output in markdown
        debug_mode: Enable debug logs
        show_members_responses: Print member responses during coordination
        enable_memory: Enable persistent memory (default: True). Set to False for
                      isolated testing to prevent state leakage between runs.
        db_file: Custom database file path. If not provided, uses CS_COPILOT_MEMORY_DB.
                Use unique paths for session isolation in testing.
        enable_mlflow_tracking: Enable MLflow tracking for agents (default: True).
                               Set to False to disable tracking.

    Returns:
        Team: Configured Cs_copilot team

    Raises:
        AgentCreationError: If one or more agents fail to initialize
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating Cs_copilot Agent Team")

    # ✅ Single DB handles session storage + user memories in v2.1.x
    # For testing, either disable memory or use unique DB files per session
    db = None
    if enable_memory:
        db = SqliteDb(
            db_file=db_file
            or CS_COPILOT_MEMORY_DB
            # NOTE: CS_COPILOT_MEMORY_TABLE is not required by SqliteDb.
            # Agno manages its own tables for sessions/memories. Kept import for compat.
        )

    # Common agent parameters supplied by the factory
    agent_params = {
        "markdown": markdown,
        "debug_mode": debug_mode,
        "enable_mlflow_tracking": enable_mlflow_tracking,
    }

    # ============================================================================
    # RUNTIME TEAM ARCHITECTURE
    # ============================================================================
    # Consolidation history:
    #   MERGED: GTM Optimization + Loading + Density + Activity → GTM Agent
    #   GENERALIZED: GTM Chemotype Analysis → Chemoinformatician (method-agnostic)
    #   MERGED: Autoencoder + Autoencoder GTM Sampling → Autoencoder (mode-based)
    #   ADDED: Report Generator (presentation layer)
    #   REPLACED: Property Predictor was superseded by the isolated QSAR team
    #   REMOVED: Robustness Evaluator (not included in main team, invoked separately)
    # ============================================================================

    # (type_key, human_name)
    agents_config: List[Tuple[str, str]] = [
        ("chembl_downloader", "ChEMBL Downloader"),
        (
            "gtm_agent",
            "GTM Agent",
        ),  # Unified GTM operations (build, load, density, activity, project)
        (
            "chemoinformatician",
            "Chemoinformatician",
        ),  # Comprehensive chemoinformatics (chemotype, clustering, SAR, similarity, QSAR)
        ("report_generator", "Report Generator"),  # Universal presentation layer
        ("autoencoder", "Autoencoder"),  # SMILES molecule generation (LSTM autoencoder)
        ("peptide_wae", "Peptide WAE"),  # Peptide sequence generation (Wasserstein autoencoder)
        ("synplanner", "SynPlanner"),
        # Note: Robustness Evaluator excluded from main team (invoked separately for testing)
    ]

    agents = []
    failures = []

    for agent_type, agent_name in agents_config:
        try:
            logger.info("Creating %s agent", agent_name)
            agent = create_agent(agent_type, model=model, **agent_params)
            agents.append(agent)
            logger.info("Successfully created %s agent", agent_name)
        except Exception as e:
            logger.exception("Failed to create %s agent", agent_name)
            failures.append(f"{agent_name}: {e!s}")

    if failures:
        msg = "Agent initialization failures:\n  - " + "\n  - ".join(failures)
        raise AgentCreationError(msg)

    team = Team(
        name="Cs_copilot Team",
        members=agents,
        model=model,
        # ✅ Attach DB directly to the team (persists sessions/history/memories)
        # If enable_memory=False, db=None prevents any persistence
        db=db,
        # Team-level capabilities (disabled when enable_memory=False)
        enable_agentic_memory=enable_memory,  # let the team manage memories
        enable_user_memories=False,  # Disable cross-session user memories for session isolation
        add_history_to_context=enable_memory,  # include recent history in prompts
        num_history_runs=5 if enable_memory else 0,  # 🔧 LIMIT context to last 5 runs
        share_member_interactions=True,  # share member messages across team
        store_history_messages=enable_memory,  # persist message history to DB
        store_tool_messages=enable_memory,  # persist tool results
        store_media=enable_memory,  # persist any media if used
        # Session state (always enabled for within-session data passing)
        add_session_state_to_context=True,
        enable_agentic_state=True,
        # Prompting
        description=(
            "You are an intelligent coordinator orchestrating a team of specialized cheminformatics agents. "
            "Your role is to understand user requests, select the appropriate agent(s) or workflows, "
            "and chain multiple agents when needed to complete complex analyses. "
            "You are also aware of a separate isolated QSAR sub-system dedicated to curation, training, model governance, inference, and reporting.\n\n"
            "• ChEMBL Downloader: Download bioactivity data from ChEMBL database\n"
            "• GTM Agent: All GTM operations (build/load/density/activity/project) with smart caching\n"
            "• Chemoinformatician: Downstream analysis (scaffold, SAR, similarity, clustering) - works with GTM output\n"
            "• Report Generator: Universal presentation layer for all analysis types\n"
            "• Autoencoder: Small molecule generation via LSTM autoencoders (SMILES, standalone + GTM-guided)\n"
            "• Peptide WAE: Peptide sequence generation + GTM on latent space + DBAASP antimicrobial activity landscapes\n"
            "• SynPlanner: Retrosynthetic planning for target molecules\n"
            "• QSAR Sub-system: Dataset Curation, QSAR Training, Model Registry, Model Inference, and QSAR Report\n\n"
            "**Molecule vs Peptide Routing**:\n"
            "  - 'peptide', 'amino acid', 'AMP', 'antimicrobial peptide' → Peptide WAE agent\n"
            "  - 'SMILES', 'molecule', 'compound', 'small molecule' → Autoencoder agent\n"
            "  - 'QSAR', 'Chemprop', 'predict lipophilicity', 'train a model', 'applicability domain' → isolated QSAR sub-system\n"
            "  - DBAASP/antimicrobial landscapes → Peptide WAE agent (has GTM tools)\n"
            "  - Unqualified 'generate' → Autoencoder (small molecules)\n\n"
            "When coordinating: (1) Assess if a predefined workflow covers the request, (2) Select and chain "
            "specialized agents for multi-step tasks (GTM → Chemoinformatician → Report Generator is common), "
            "(3) For analysis requests, automatically add Report Generator unless user explicitly requests raw data only, "
            "(4) For ambiguous opening requests, apply the INITIAL CLARIFICATION FLOW (peptides vs molecules, then exploratory vs generative), (5) Synthesize insights from agent outputs into coherent analyses. "
            "Do not attempt QSAR training or QSAR prediction inside this general team; those tasks belong to the isolated QSAR team."
        ),
        instructions=AGENT_TEAM_INSTRUCTIONS,
        # UX & observability
        markdown=markdown,
        debug_mode=debug_mode,
        stream_member_events=True,  # stream events from members (Team API)
        show_members_responses=show_members_responses,
    )

    logger.info("Successfully created Cs_copilot Agent Team")
    return team


def get_qsar_agent_team(
    model: Model,
    *,
    markdown: bool = True,
    debug_mode: bool = False,
    show_members_responses: bool = True,
    enable_memory: bool = True,
    db_file: str = None,
    enable_mlflow_tracking: bool = True,
) -> Team:
    """
    Create an isolated QSAR-only team.

    This team is intentionally compartmentalized from the broader Cs_copilot
    ecosystem so that dataset curation, training, model registration, and
    inference can evolve independently with minimal prompt cross-talk.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating isolated QSAR Agent Team")

    db = None
    if enable_memory:
        db = SqliteDb(db_file=db_file or CS_COPILOT_MEMORY_DB)

    agent_params = {
        "markdown": markdown,
        "debug_mode": debug_mode,
        "enable_mlflow_tracking": enable_mlflow_tracking,
    }

    agents_config: List[Tuple[str, str]] = [
        ("dataset_curation", "Dataset Curation"),
        ("qsar_training", "QSAR Training"),
        ("model_registry", "Model Registry"),
        ("model_inference", "Model Inference"),
        ("qsar_report", "QSAR Report"),
    ]

    agents = []
    failures = []

    for agent_type, agent_name in agents_config:
        try:
            logger.info("Creating %s agent", agent_name)
            agent = create_agent(agent_type, model=model, **agent_params)
            agents.append(agent)
            logger.info("Successfully created %s agent", agent_name)
        except Exception as e:
            logger.exception("Failed to create %s agent", agent_name)
            failures.append(f"{agent_name}: {e!s}")

    if failures:
        msg = "QSAR agent initialization failures:\n  - " + "\n  - ".join(failures)
        raise AgentCreationError(msg)

    team = Team(
        name="QSAR Team",
        members=agents,
        model=model,
        db=db,
        enable_agentic_memory=enable_memory,
        enable_user_memories=False,
        add_history_to_context=enable_memory,
        num_history_runs=5 if enable_memory else 0,
        share_member_interactions=True,
        store_history_messages=enable_memory,
        store_tool_messages=enable_memory,
        store_media=enable_memory,
        add_session_state_to_context=True,
        enable_agentic_state=True,
        description=(
            "You are the isolated coordinator of a dedicated QSAR sub-system. "
            "You may only orchestrate the following QSAR specialists: "
            "Dataset Curation, QSAR Training, Model Registry, Model Inference, and QSAR Report. "
            "Do not involve unrelated cheminformatics agents. "
            "Use Dataset Curation before any new training workflow. "
            "Use QSAR Training only on curated QSAR-ready datasets. "
            "Use Model Registry for catalog governance and persistence decisions. "
            "Use Model Inference for explicit predictions or model-selection-driven predictions. "
            "Use QSAR Report as the only final drafting agent for the user-facing answer. "
            "Keep handoffs structured and concise."
        ),
        instructions=[
            "You coordinate only the isolated QSAR agents in this team.",
            "Never route work to non-QSAR agents.",
            "For curation-only requests, orchestrate: dataset_curation -> qsar_report.",
            "For training requests, orchestrate: dataset_curation -> qsar_training -> model_registry -> qsar_report.",
            "For prediction requests on existing models, orchestrate: model_inference -> qsar_report.",
            "For QSAR backend/capability inventory requests, orchestrate: model_registry -> qsar_report. Do not answer directly from the coordinator or return the model_registry response directly.",
            "For QSAR catalog listing, catalog search, model recommendation, model summary, or model comparison requests, orchestrate: model_registry -> qsar_report. Do not answer directly from the coordinator or return the model_registry response directly.",
            "For existing ensemble summary requests, orchestrate: model_registry -> qsar_report. Do not return the model_registry handoff directly to the user.",
            "For explicit post-prediction LaTeX export requests, including the standalone shortcut token `latex` with optional `@` and any casing (`@Latex`, `@latex`, `@LATEX`, `@LaTeX`, `Latex`, `latex`, `LaTeX`), orchestrate `model_inference` only and treat the task as a documentation export for the latest completed prediction state.",
            "When the user asks only for LaTeX or payload export, do not rerun prediction and do not route to `qsar_report` unless the user also asked for a narrative report.",
            "For export-only LaTeX or payload requests handled by `model_inference`, return the `model_inference` answer verbatim without adding any extra narrative.",
            "Before routing, determine REPORT_LANGUAGE from the latest user message: English for English prompts, French for French prompts. Include `REPORT_LANGUAGE: English` or `REPORT_LANGUAGE: French` in every handoff, especially the final handoff to `qsar_report`.",
            "The final QSAR report must use REPORT_LANGUAGE even if operational handoffs, tool outputs, catalog metadata, or prior conversation are in another language.",
            "Only `qsar_report` may draft the final user-facing answer.",
            "Treat the other QSAR agents as operational specialists that provide structured handoffs only.",
            "If a member response contains useful information for the user, pass it to `qsar_report` for final drafting instead of exposing it yourself.",
            "When `qsar_report` has produced a final answer, return that answer verbatim without adding a preface, summary, duplication, or extra conclusion.",
            "Do not expose intermediate agent narration to the user unless the workflow is blocked before `qsar_report` can run.",
            "For blocked workflows, stop early and summarize only completed steps, blockers, files, and next steps.",
            "Prefer structured handoffs over long narrative summaries.",
        ],
        markdown=markdown,
        debug_mode=debug_mode,
        stream_member_events=True,
        show_members_responses=show_members_responses,
    )

    logger.info("Successfully created isolated QSAR Agent Team")
    return team
