# Curation policy

## Required checks

- Confirm the molecular column contains parseable structures and retain invalid-row evidence.
- Confirm target columns exist and match the requested regression or classification task.
- Regression targets must be numeric and non-constant after curation.
- Preserve the original source path and an audit trail from input rows to curated rows.
- Standardize structures through the existing toolkit. Prefer `curation_backend=chembl_structure_v1`; use `legacy_rdkit_v1` only when explicitly requested or reported as a fallback. Do not implement alternate chemistry rules in the agent.
- Remove detected inorganic structures, organometallic structures, and true multi-organic-fragment mixtures before standardization. Keep salt/counterion cases distinct from mixtures.
- Treat ChEMBL checker flags as diagnostics unless the structured artifact explicitly records row removal.
- Use the toolkit's `strip_then_deduplicate` QSAR identity policy. For regression, retain the default duplicate conflict threshold `0.5` unless the user explicitly changes it: aggregate coherent groups by the toolkit policy and remove strongly conflicting groups.
- Preserve `duplicate_conflicting_rows_removed` and `duplicate_groups_aggregated` as distinct facts; never combine them into one removal count.
- Report salt/fragment handling, charge and tautomer decisions when returned by the toolkit.
- Preserve activity units and transformations. Never mix incompatible units or infer a log transform.
- Flag statistical target outliers without removing them unless the user explicitly requested removal.
- Retain target, assay, organism, and provenance constraints when present.
- Preserve available experimental-context and measurement-quality columns as downstream signals; their absence alone does not justify dropping rows.
- Separate excluded rows from the curated output and cite the returned exclusion artifact.

## Readiness decision

Mark a dataset ready only when the toolkit confirms a usable curated path, curated SMILES column, target columns, task type, row counts, and no blocking validation error. Missing credible SMILES, no valid target, constant regression targets, unresolved target-unit conflicts, or an empty curated dataset are hard blockers. Warnings remain warnings; do not hide them to claim readiness.

Facts should include at least source and curated row counts, valid molecule count, duplicate handling, missing targets, units or transform, curated columns, exclusions, and readiness.
