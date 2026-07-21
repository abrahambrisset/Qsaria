---
name: qsaria-curate
description: Inspect, validate, curate, summarize, and document one molecular dataset for a bounded Qsaria curation mission delegated to the Claude Code curation specialist.
user-invocable: false
---

# Curate Qsaria Data

Prepare one traceable QSAR-ready dataset. Do not train, infer, select models,
persist models, or write the final report.

## Workflow

1. Open the supplied experiment and verify its state.
2. Inspect the exact input path. Never guess, reconstruct, or substitute a
   similarly named file.
3. Identify molecular and target columns while preserving explicit user
   choices. Return `needs_user_input` when ambiguity changes scientific meaning.
4. Call the curation facade with explicit task type and units when known.
   Structure standardization always uses `chembl_structure_v1`; there is no
   public backend selector or fallback standardizer. Preserve row-level
   `standardization_failed` evidence and continue when usable compounds remain.
5. Summarize the curated dataset and write the curation report when requested.
6. Verify every artifact id and preserve row counts, exclusions, normalized
   SMILES, targets, units, duplicate handling, warnings, and blockers.

Never remove rows, change units or targets, or relax validation silently to
bypass a blocker.

## Return one public handoff

Build a fresh envelope with `schema_version="1.0"`, the exact experiment id,
`agent="qsaria_curation"`, `execution_mode="project_agent"`, status, summary,
facts, artifact ids, model ids, warnings, blockers, and recommended next
action. A toolkit `handoff` is only source evidence and must never be submitted
directly.

Call `qsaria_record_handoff` exactly once immediately before returning. Return
the exact normalized envelope from that call to the coordinator.
