#!/usr/bin/env python
# coding: utf-8
"""Tests for report-generation instruction requirements."""

from cs_copilot.agents.prompts import (
    MODEL_INFERENCE_INSTRUCTIONS,
    QSAR_REPORT_INSTRUCTIONS,
    REPORT_GENERATOR_INSTRUCTIONS,
)


def _report_instructions_text() -> str:
    return "\n".join(REPORT_GENERATOR_INSTRUCTIONS)


def _qsar_report_instructions_text() -> str:
    return "\n".join(QSAR_REPORT_INSTRUCTIONS)


def _model_inference_instructions_text() -> str:
    return "\n".join(MODEL_INFERENCE_INSTRUCTIONS)


def test_report_instructions_require_named_captioned_inline_figures():
    """Report Generator instructions should enforce named, captioned inline figures."""
    instructions = _report_instructions_text()

    assert (
        "Every available non-Plotly static PNG or registered inline_static figure" in instructions
    )
    assert "Every figure object MUST include name and caption unless" in instructions
    assert "Number figures sequentially across the whole report" in instructions
    assert (
        "GTM density landscape figures MUST appear directly after density analysis" in instructions
    )
    assert (
        "GTM activity landscape figures MUST appear directly after activity analysis"
        in instructions
    )
    assert (
        "do not defer density or activity landscapes to a final Visualizations section"
        in instructions
    )
    assert "pass mark_nodes for every GTM node discussed in the density text" in instructions
    assert "pass mark_nodes for every GTM node discussed in the activity text" in instructions
    assert "Every GTM node discussed in the report text MUST be explicitly labeled" in instructions
    assert "Never save a rich report with only title/summary" in instructions
    assert "saving a summary-only file" in instructions
    assert "Use the registered figure metadata for colorscale" in instructions
    assert "do not invent density color meanings" in instructions
    assert (
        "label dense/potent nodes by color unless the metadata explicitly says so" in instructions
    )
    assert "Include only the static PNG" in instructions
    assert "do not include Plotly PNGs, Plotly HTML, or Plotly artifact_path" in instructions
    assert "Prefer registered session figure metadata" in instructions
    assert "Do not put Plotly paths or GTM interactive .html artifact_path values" in instructions
    assert "structure_smiles or smiles" in instructions
    assert (
        "save_rich_report will generate a section-local compound image automatically"
        in instructions
    )
    assert "a scaffold/SAR paragraph contains an untagged valid SMILES" in instructions
    assert "Scaffold_1, Scaffold_2" in instructions
    assert "Molecule_1, Molecule_2" in instructions
    assert "Resolve molecule and scaffold IDs separately" in instructions
    assert (
        "compound_id/compound_ids, molecule_id/molecule_ids, "
        "molecule_chembl_id/molecule_chembl_ids" in instructions
    )
    assert "scaffold_id/scaffold_ids before generic structure/source IDs" in instructions
    assert "dataset-provided display names" in instructions
    assert "ChEMBL ID" in instructions
    assert "Only when no type-specific source ID exists" in instructions
    assert "MUST reference the matching figure" in instructions
    assert "CMPD-123, top potency source analog (Figure 4)" in instructions
    assert "Scaffold_1, Piperidine urea phenyl scaffold (Figure 3)" in instructions
    assert "Piperidine urea phenyl scaffold" in instructions
    assert "Top potency piperidine urea analog" in instructions
    assert "structure_type ('scaffold' or 'molecule')" in instructions
    assert "after_paragraph_index" in instructions
    assert "Scaffold ID / Scaffold / SMILES / Name / Node / Description" in instructions
    assert "Molecule ID / Molecule / SMILES / Name / Node / Description" in instructions
    assert "Scaffold inventory table rows with scaffold SMILES" in instructions
    assert "Do not render every valid SMILES" in instructions


def test_report_instructions_define_required_report_structures():
    """Workflow-specific reports should have stable required structures."""
    instructions = _report_instructions_text()

    assert "GTM analysis report required structure" in instructions
    assert "User Request and Data Source" in instructions
    assert "Retrieved and Standardized Data" in instructions
    assert "Descriptors" in instructions
    assert "GTM Construction or Loading" in instructions
    assert "Map Analysis" in instructions

    assert "Analog generation report required structure" in instructions
    assert "Reference Maps" in instructions
    assert "Generated Compound Analysis" in instructions

    assert "SynPlanner Routes and Attempts" in instructions
    assert "Route Analysis" in instructions


def test_qsar_report_instructions_require_fact_first_v2_handoffs():
    instructions = _qsar_report_instructions_text()

    assert "Resume executif" in instructions
    assert "Dataset, curation et qualite des donnees" in instructions
    assert "Optimisation des hyperparametres" in instructions
    assert "Analyse des outliers" in instructions
    assert "Resultats de test externe" in instructions
    assert "Resultats de test interne" in instructions
    assert "NEVER use the legacy `Partie 1`, `Partie 2`, `Partie 3`, or `Partie 4`" in instructions
    assert "allowed only for curation-only reports" in instructions
    assert "reporting_handoff" in instructions
    assert "report_facts" in instructions
    assert "report_tables" in instructions
    assert "selection_validation_metrics" in instructions
    assert "final_test_comparison" in instructions
    assert "Concise tables for curation, working environment, protocol, durations, tuning, and persisted models" in instructions
    assert "it is mandatory in `Resultats de test externe`" in instructions
    assert "automatic development-only Activity Cliff annotation" in instructions
    assert "Immediately after every copied table" in instructions
    assert "beginning with `Commentaire :`" in instructions
    assert "Interpret only its visible values" in instructions
    assert "non calculable" in instructions
    assert "out_of_domain" in instructions


def test_model_inference_instructions_do_not_export_prediction_summary_after_external_eval():
    instructions = _model_inference_instructions_text()

    assert "After a successful `evaluate_model_on_dataset`" in instructions
    assert "do not call `export_prediction_summary`" in instructions
