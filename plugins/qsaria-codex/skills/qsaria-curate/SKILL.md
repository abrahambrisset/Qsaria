---
name: qsaria-curate
description: Inspect, validate, curate, summarize, and document molecular datasets for Qsaria QSAR workflows. Use only inside the qsaria_curation project agent when the coordinator delegates dataset preparation or curation review.
---

# Curate Qsaria Data

Prepare a traceable QSAR-ready dataset without training models or making catalog decisions.

Read [curation-policy.md](references/curation-policy.md), [scientific-invariants.md](../../contracts/scientific-invariants.md), and [handoff.schema.json](../../contracts/handoff.schema.json) before using tools.

## Workflow

1. Open the provided experiment and inspect its current state.
2. Inspect the exact input dataset path. Never replace it with a guessed path or a similarly named file.
3. Identify molecular and target columns. Preserve explicit user column choices; request input if an ambiguity changes scientific meaning.
4. Run the existing Qsaria curation facade with explicit task type and units when known.
5. Summarize the curated dataset and write the curation report when the mission requires it.
6. Verify that every returned artifact exists in the experiment and that the curated SMILES column, target columns, row counts, exclusions, unit handling, duplicates, and warnings are represented in facts.

Do not train, infer, persist a model, or draft the final user report. Do not bypass a scientific blocker by dropping rows, changing units, changing targets, or relaxing validation silently.

## Return the handoff

Build the common handoff envelope. Set `agent` to `qsaria_curation`, reference only returned artifact identifiers, and state whether the data are ready for downstream QSAR. Call `qsaria_record_handoff` exactly once, immediately before returning its normalized envelope to the coordinator.
