# Qsaria report contract

Use `report_facts.schema_version=2.0` as the compact source of truth and use only table material supplied in `report_tables`. Structured artifacts and structured tool results outrank handoffs, which outrank narrative text.

## Table editorial policy

`report_tables` supplies verified material; it does not require every available table to appear. Select zero to three tables according to explanatory value and avoid repeating facts already clearer in prose. A selected table may omit redundant rows or columns or reorder them, but every displayed label, population, metric, and value must come unchanged from the structured table. Do not recompute cells, combine non-comparable splits or variants, or infer a new numerical table from narrative fields.

## Final status

Report is dispatched exactly once and therefore returns only `completed`, `partial`, or `terminal_failure`. It never pauses with `needs_user_input` and is never retried with `retryable_error`; the coordinator resolves prerequisites before dispatch. A `completed` or `partial` handoff claims exactly the saved report artifact. If verification fails after the save, `terminal_failure` claims the same artifact and treats it as unvalidated evidence.

## Language

Write the whole report in `REPORT_LANGUAGE` when supplied; otherwise use the dominant language of the latest user request. Preserve technical identifiers and keys exactly.

## Training report

For French training reports, preserve this ordered hierarchy:

1. `Resume executif`
2. `Dataset, curation et qualite des donnees`
3. `Environnement de travail`
4. `Protocole d'entrainement`
5. `Optimisation des hyperparametres`
6. `Analyse des outliers`
7. `Resultats de test externe` or `Resultats de test interne`
8. `Gouvernance et artefacts generes`

Use equivalent plain English headings for English reports. Do not use French `Partie N` prefixes.

Within `Dataset, curation et qualite des donnees`, keep `Description du dataset`, `Curation`, `Activity Cliffs`, and `Applicability Domain` in that order when facts exist. Keep descriptive Activity Cliffs separate from development-only outlier annotations and from any feedback retraining loop.

State whether evaluation scope is `external`, `internal`, or `none`. For external evaluation, state that isolated test data were not used for selection. For internal evaluation, state that evidence is internal/OOF and final refits lack independent evaluation. For `none`, state that full-train produced no internal evaluation and requires labelled external evaluation.

## Standalone labelled evaluation

Use only: executive summary, evaluated dataset, external test results, applicability domain, and governance/artifacts. Do not describe historical training or curation as newly executed.

Each substantive section needs concise evidence-grounded prose, including explicit mention of skipped or disabled modules. Do not add tables not supplied by the reporting context.

## Other canonical report shapes

Use exactly one shape that matches the completed workflow.

For curation-only work, use `Partie 1 : Curation` with this stable order: `Source dataset`, `Colonnes conservees`, `Comptage des lignes`, `Filtrage structural`, `Gestion des doublons`, `Politique de curation`, `Avertissements`, `Blocages`, `Statut final de la curation`, and `Fichiers generes`. Keep zero/none cases visible.

For ordinary prediction, use: `Modele utilise`, `Resultats des predictions`, `Evaluation de fiabilite par statut AD`, `Resume statistique`, `Interpretation des valeurs Y`, `Fichier de resultats complet`, and `Recommandations d'utilisation`.

For ensemble inference, use: `Modele ensemble utilise`, `Composants appeles`, `Resultats des predictions`, `Desaccord inter-composants`, `Resume statistique`, `Fichiers generes`, and `Limites et recommandations`. The official consensus is the returned median; component standard deviation is disagreement, not calibrated uncertainty.

For ensemble creation or evaluation, use: `Objectif`, `Inventaire des modeles compatibles`, `Criteres de selection`, `Composants retenus`, `Composants non retenus`, `Modele ensemble cree`, `Evaluations disponibles`, and `Fichiers generes`. Do not claim ensemble improvement without same-row ensemble evaluation evidence.

Begin with one factual title and a short introduction. Preserve applicability-domain labels exactly as `in_domain`, `edge_of_domain`, and `out_of_domain`. Show prediction results once, and use only literal returned file references for downloads. A session artifact is not a persisted catalog model; persistence requires explicit canonical evidence from Registry or the originating tool.

Keep facts, interpretation, and recommendations separate. Use sober scientific wording and never upgrade `experimental`, `workflow_demo`, `validated`, or `robust_validated` beyond the evidence.
