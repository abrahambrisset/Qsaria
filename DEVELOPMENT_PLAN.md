# Qsaria — Development Plan for Coding Agents

Status: analysis of commit `61a0cc2`, branch `claude/qsar-repo-analysis-76ac23`.
Audience: coding agents and maintainers picking up discrete work items.

Every claim below carries a `file:line` reference. Every absence was verified by
grep before being written down. Task items are written to be executed
independently; each states its files, its acceptance criterion, and its
dependencies.

---

## 0. Where the repository stands

78 825 LOC under `src/`, 37 327 LOC under `tests/`, 441 commits, three separate
orchestration runtimes.

### What is genuinely strong

These are load-bearing and should not be refactored away:

| Capability | Evidence |
| --- | --- |
| Typed scientific contracts with `extra="forbid"` | `tools/prediction/qsar_contracts.py:101`, `TRAINING_CONTRACT_VERSION = "2.0"` |
| ChEMBL Structure Pipeline curation with full audit counters | `tools/curation/backends/chembl_structure_v1.py:47`, `tools/curation/backend.py:61` |
| Duplicate resolution with a conflict threshold, not silent averaging | `tools/curation/dataset_curation_toolkit.py:360` |
| Deterministic handoff that overwrites LLM prose with tool-derived JSON | `agents/qsar_agent_handoffs.py:407` |
| Artifact existence verified on disk before it enters a report | `agents/qsar_agent_handoffs.py:207` |
| Report saved, re-read, and SHA-256 verified | `mcp/qsaria/experiments.py:1625` |
| Content-addressed artifact IDs, unmanaged paths refused | `mcp/qsaria/contracts.py:363`, `experiments.py:2379` |
| Detached operations survive client disconnect, atomic state writes | `mcp/qsaria/operations.py:496`, `:104` |
| Server-side role state machine, single-retry budget | `mcp/qsaria/experiments.py:2116`, `:2178` |
| MCP server runs no LLM at all | `mcp/server.py:87` |
| Feature cache keyed by dataset fingerprint + representation | `tools/prediction/qsar_training_toolkit.py:132` |
| Applicability domain bound to a feature `schema_hash` | `tools/prediction/applicability_domain.py:130` |
| Outlier-driven refit with an explicit, defensible eligibility rule | `tools/prediction/outlier_analysis.py:141` |
| SALI activity-cliff analysis with normalised scoring | `tools/activity_cliffs/service.py:119` |
| Typed report-facts handoff, `REPORT_FACTS_SCHEMA_VERSION = "2.0"` | `tools/prediction/qsar_reporting.py:31` |

The engineering discipline here is above average for a research codebase. The
gaps below are almost entirely **scientific scope** and **evaluation**, not code
quality.

### The one-sentence diagnosis

Qsaria is a well-engineered *pipeline executor* with a thin *scientific
methodology surface* and *no measurement of whether its scientific decisions are
correct*. It can run a QSAR workflow reproducibly; it cannot yet demonstrate
that the workflow it chose was the right one, nor that its models beat a
random-forest baseline on a public benchmark.

---

## 1. QSAR methodology: what is missing

### 1.A Correctness risks — fix before the public pre-release

These are not missing features. They are paths where the current code produces a
result that is misleading.

---

#### Q-A1 — The default protocol cannot pass the repository's own quality gates

**Observed.** `standard_qsar` is the only named protocol and resolves to exactly
one random 80/10/10 split (`tools/prediction/qsar_training_policy.py:85`,
`:438`). The governance assessor compares every non-`random` split family
against the `random` family (`:587`) and returns early when the `random` family
is the only one present (`:573`).

**Consequence.** Under the default protocol `delta_vs_random` is `{}`,
`hardest_split` is `None`, `passes_hardest_split_gate` and
`passes_robustness_gate` are both `False`, and `recommended_status` degrades to
`workflow_demo` (`:784`). Every model trained the default way is permanently
un-promotable, and the thresholds at `:20-26` never execute. The benchmark
leaderboard inherits this: `scaffold_r2`, `cluster_kmeans_r2`, `hardest_split`
and `delta_r2` are all `None`, so ranking silently degenerates to single-split
random R² then wall-clock time (`tools/prediction/benchmark_toolkit.py:577`).

**Also.** A single random split for QSAR is the textbook optimism trap: random
splits place close analogues on both sides and overstate prospective
performance, often by a wide margin on congeneric series.

**Task.** Redefine `standard_qsar` as a multi-family protocol: one random split
plus one scaffold split plus one cluster split, sharing a seed policy. Keep the
single-split behaviour reachable as a new explicit protocol name
(`single_random_holdout`) for speed-critical use.

- Files: `tools/prediction/qsar_training_policy.py` (`_standard_split_templates`,
  `resolve_validation_protocol`), `tools/prediction/qsar_validation_strategy.py`
  (must emit >1 family per run), `tools/prediction/lightgbm_toolkit.py:2344`
  (family derivation from run label).
- Acceptance: a default `train_qsar_model` call produces `split_results`
  containing ≥2 distinct `strategy_family` values; `assess_protocol_results`
  returns a non-empty `delta_vs_random`; a model can reach `validated`. Add a
  regression test asserting this.
- Effort: medium. Dependency: none. **Do this first** — Q-A4, Q-B5 and the whole
  benchmark plan in §4 depend on it.

---

#### Q-A2 — Curation and featurisation use two different standardizations

**Observed.** Curation standardizes with the ChEMBL Structure Pipeline
(`standardize_mol` → `get_parent_mol` → checker), then builds the dedup key by
stripping stereochemistry while **keeping** the stereo-bearing parent in the
`smiles` column (`tools/curation/dataset_curation_toolkit.py:725`, `:760`;
`tools/curation/identity.py:10`).

Featurisation then re-standardizes with a *different* RDKit pipeline —
`Cleanup` → `LargestFragmentChooser` → `Uncharger` →
`TautomerEnumerator.Canonicalize` → `RemoveStereochemistry`
(`tools/chemistry/standardize.py:89-97`), with `remove_stereochemistry: bool =
True` as the default (`:122`) and every call site in the training path passing
only two positional arguments (`tools/features/molecular_feature_toolkit.py:292`,
`:377`; `tools/prediction/qsar_training_toolkit.py:952`).

**Consequence, two parts.**

1. *Tautomer collapse after deduplication.* The dedup key never saw tautomer
   canonicalisation. Two rows kept as distinct by curation can become
   byte-identical SMILES at featurisation time. Nothing deduplicates afterwards —
   `drop_duplicates` across `tools/features/` and `tools/prediction/` returns one
   unrelated hit (`outlier_analysis.py:224`). The split is built on row indices,
   so identical feature vectors can land on both sides.
   **This is train/test leakage introduced by the pipeline itself.**
2. *Structural stereo-blindness.* Features are computed on stereo-stripped
   structures. Qsaria cannot represent, learn, or predict enantioselective
   activity, and nothing in the documentation says so.

**Task.** Make standardisation single-sourced and explicit.

- Featurisation must consume the curated `smiles` column as-is, not
  re-standardize. If a guard is wanted, make it assert idempotence and fail
  loudly rather than silently transform.
- Add a post-standardisation duplicate check that reports or raises when
  standardisation collapses rows that curation kept distinct.
- Make `remove_stereochemistry` an explicit, recorded curation policy field
  (`tools/curation/policies.py:11` already has `stereochemistry_policy`), and
  surface the chosen value in the curation report and the model metadata.
- Files: `tools/chemistry/standardize.py`,
  `tools/features/molecular_feature_toolkit.py:292,377`,
  `tools/prediction/qsar_training_toolkit.py:952`,
  `tools/curation/dataset_curation_toolkit.py`.
- Acceptance: a dataset containing a tautomer pair that survives curation either
  (a) fails featurisation with a clear duplicate error, or (b) is merged before
  splitting — never silently split across train and test. Regression test
  required, plus an explicit enantiomer pair asserting documented behaviour.
- Effort: medium. Priority: **highest** — this is a silent correctness defect.

---

#### Q-A3 — No chemical leakage check between training and evaluation sets

**Observed.** What exists is index-level and column-level only: split index
disjointness (`tools/prediction/chemprop_adapter.py:168`,
`tools/prediction/qsar_splitters.py:391`) and a guard preventing target/SMILES
columns from becoming features (`tools/prediction/lightgbm_backend.py:190`).
`external_evaluation.py:429` never compares external SMILES against the training
set. The literal `"training_like_or_potentially_leaky"` exists
(`tools/prediction/ensemble_toolkit.py:37`) but is a caller-supplied label
validated only for membership (`:828`) — nothing detects the condition.

**Consequence.** External evaluation can silently re-score training compounds and
report it as prospective performance. Scaffold splits can still place near
neighbours across the boundary, because `scaffold_balanced` is a balanced
bin-pack, not a disjointness guarantee (`qsar_splitters.py:113`).

**Task.** Add a `leakage_report` computed for every train/test pair and every
external evaluation:

- exact duplicate count on canonical SMILES;
- near-duplicate count at Tanimoto ≥ 0.95 and ≥ 0.99 on count-Morgan (reuse
  `tools/activity_cliffs/service.py:137`, which already builds these);
- maximum and median nearest-neighbour similarity from each test compound to the
  training set.
- Files: new `tools/prediction/leakage.py`, wired into
  `training_orchestration.py` and `external_evaluation.py`; surfaced in
  `qsar_reporting.py` report facts.
- Acceptance: `evaluate_model_on_dataset` returns a `leakage` block; the report
  renders it; a test with a deliberately leaked row flags it.
- Effort: small-medium. High value per line of code.

---

#### Q-A4 — Classification benchmarking compares nothing

**Observed.** `_resolve_backends` admits LightGBM and TabICL only when
`task_type == "regression" and len(target_columns) == 1`
(`tools/prediction/benchmark_toolkit.py:174-185`), although
`backend_capabilities.py:65,80` declares both support `classification` and
`multiclass_classification`.

**Consequence.** A classification benchmark runs Chemprop alone and produces a
one-row leaderboard presented as a comparison.

**Task.** Remove the task-type restriction, or replace it with a capability
lookup against `backend_capabilities`. If the restriction encodes a real
limitation, the limitation belongs in `backend_capabilities`, not in the
benchmark.

- Files: `tools/prediction/benchmark_toolkit.py:163-194`.
- Acceptance: a binary-classification benchmark yields ≥3 candidates; test added.
- Effort: small. Blocks all classification work in §4.

---

#### Q-A5 — The leaderboard compares tuned against untuned models

**Observed.** `TuningConfig(enabled=backend_name != "tabicl")`
(`tools/prediction/benchmark_toolkit.py:312`). LightGBM gets Optuna/TPE with 50
trials by default; Chemprop gets HPO; TabICL gets none, because
`qsar_contracts.py:541` forbids it.

**Consequence.** Any backend ranking is confounded by tuning budget. TabICL is
structurally disadvantaged and the report does not say so.

**Task.** Either equalise the budget (a parity mode running LightGBM/Chemprop
untuned) or record `tuning_enabled` and `n_trials` per candidate in
`leaderboard.csv` and state the asymmetry in the report's limitations section.

- Files: `tools/prediction/benchmark_toolkit.py:302-320`,
  `_candidate_summary_row:435`, `_render_benchmark_report:653`.
- Acceptance: leaderboard carries a tuning-budget column; report names the
  confound.
- Effort: small.

---

### 1.B Missing methodology — scope expansion

Ordered by scientific return per unit of work.

---

#### Q-B1 — No baselines. Nothing can be shown to be an improvement

**Observed.** Exactly four backends exist: Chemprop, LightGBM, TabICL, and a
post-hoc Ensemble (`tools/prediction/backend_factory.py:10`). The entire sklearn
import surface in `src/` is PCA, StandardScaler, NearestNeighbors,
IsolationForest, KMeans, RepeatedKFold, `train_test_split`, and four plotting
`Display` classes. No Random Forest, SVM/SVR, XGBoost, CatBoost, k-NN, PLS,
Gaussian Process, ridge/lasso, or a dummy/majority baseline.

**Consequence.** Random Forest on ECFP4 is the field's default reference point;
without it, no claim of the form "Qsaria's models are good" is checkable. A
dummy baseline is also the cheapest detector of a broken target column.

**Task.** Add a `sklearn` backend exposing at minimum `dummy` (mean/majority),
`random_forest`, `svm`, `pls` (regression) and `knn`. Reuse the existing
`PredictionBackend` interface and `TABULAR_REPRESENTATIONS`; these are roughly
40 lines each, not new science.

- Files: new `tools/prediction/sklearn_backend.py`,
  `tools/prediction/backend_factory.py:10`,
  `tools/prediction/backend_capabilities.py`, catalog/registry wiring.
- Acceptance: `benchmark_qsar_models` with
  `backends=["random_forest","lightgbm"]` produces a two-row leaderboard; a dummy
  baseline appears in every benchmark report as a reference row.
- Effort: medium. **Blocks §4 entirely** — a benchmark without baselines is not a
  benchmark.

---

#### Q-B2 — The descriptor space is one family wide

**Observed.** Five named representations
(`tools/prediction/tabular_representations.py:28-70`), all built from exactly two
primitives: RDKit 2D descriptors (`Descriptors._descList`,
`tools/features/molecular_feature_toolkit.py:75`) and Morgan fingerprints. Morgan
radius is hardcoded to 2 and bit count to 2048 in the training path
(`tools/prediction/qsar_training_toolkit.py:1082,1090,1130,1138`), and the public
helper actively rejects any other radius:
`"The current Morgan helper supports radius=2 only in this V1 implementation."`
(`tools/features/molecular_feature_toolkit.py:280`).

Present but **not trainable**: MACCS keys
(`tools/chemistry/base_chemistry.py:239`), RDKit topological FP (`:237`) —
similarity only. Present but **unreachable from the QSAR stack**: the Chemprop
learned embedding, used only for the applicability domain
(`tools/prediction/chemprop_backend.py:606`), and the SMILES autoencoder latent
space (`tools/chemistry/autoencoder_toolkit.py:74`), whose only callers are
`clean_dataset.py:439` and the GTM module.

Verified absent: Mordred, Avalon, atom-pair, topological torsion, pharmacophore
(Gobbi 2D), ErG, USRCAT, ChemBERTa, MolFormer, Uni-Mol, CDDD, Mol2Vec, ISIDA
fragment descriptors, any 3D or conformer-dependent descriptor, any
proteochemometric or target-side feature.

**Consequence.** Representation benchmarking — advertised as a feature — compares
three variants of the same two descriptor families. ISIDA fragment descriptors
are notable by absence given the lab's own provenance.

**Task, staged.**

- **B2.1 (small):** un-hardcode Morgan radius and bit count; expose
  `radius ∈ {1,2,3}`, `n_bits ∈ {1024,2048,4096}`, and `use_features` (FCFP) on
  `GeneratedRepresentation`. Remove the `radius=2` guard.
- **B2.2 (small):** promote MACCS, Avalon, atom-pair and topological-torsion to
  trainable `TABULAR_REPRESENTATIONS` entries. All four are one RDKit generator
  call each; the plumbing exists.
- **B2.3 (medium):** add Mordred as an optional dependency and a `mordred_2d`
  representation. Guard it — Mordred is slow and emits NaNs; needs a NaN policy
  and a column-variance filter (Q-B8).
- **B2.4 (medium):** make the Chemprop learned embedding and the autoencoder
  latent available as `PrecomputedRepresentation` producers, so a LightGBM model
  can train on a learned representation. Mostly wiring; extraction already works.
- **B2.5 (larger, optional):** a pretrained-transformer representation
  (ChemBERTa-2 or MolFormer via `transformers`, already a dependency).
- Files: `tools/prediction/tabular_representations.py`,
  `tools/features/molecular_feature_toolkit.py`,
  `tools/prediction/qsar_training_toolkit.py:1082-1150`,
  `tools/prediction/qsar_contracts.py:19`.
- Acceptance per stage: the new representation appears in
  `describe_backend_capabilities`, trains end to end, and is selectable in a
  benchmark campaign.

---

#### Q-B3 — No predictive uncertainty anywhere

**Observed.** `supports_uncertainty="none"` is declared for all three trainable
backends (`tools/prediction/backend_capabilities.py:55,68,83`), with explicit
refusals: `"LightGBM V1 does not support predictive uncertainty export."`
(`lightgbm_backend.py:520`), `"TabICL does not support predictive uncertainty
export."` (`tabicl_backend.py:336`). Chemprop's `--uncertainty-method` and
`--calibration-method` are forwarded only if the model record already carries
them (`chemprop_backend.py:581`), and **no training path ever sets them** — they
are manual metadata fields on `register_model`
(`model_registry_toolkit.py:652`).

The only shipped "uncertainty" is ensemble component spread
(`ensemble_backend.py:529`), and the code is careful to label it as *not*
calibrated uncertainty (`:559`) — correctly.

Verified absent: conformal prediction, MAPIE, Venn-ABERS, MC-dropout, deep
ensembles, evidential regression, quantile regression, NGBoost, Gaussian
processes.

**Consequence.** A QSAR platform whose stated purpose includes flagging
unreliable predictions ships applicability domain as a binary in/out flag and no
per-prediction error bar.

**Task.** Implement **inductive conformal prediction** as a backend-agnostic
wrapper. It is the right first choice: distribution-free, finite-sample valid,
identical across LightGBM/TabICL/Chemprop/sklearn, needs only a calibration
split that the validation strategies already produce, and yields prediction
intervals (regression) or prediction sets (classification) with a guaranteed
coverage level.

- New module `tools/prediction/conformal.py`: fit nonconformity scores on the
  calibration fold, expose `predict_interval(alpha)`; Mondrian/class-conditional
  variant for classification.
- Wire `uncertainty_method="inductive_conformal"` into `PredictionTaskSpec`,
  `backend_capabilities`, inference output columns and the report.
- Report empirical coverage on the test split — a conformal implementation that
  does not report its realised coverage is untestable.
- Acceptance: on a held-out set, empirical coverage at α = 0.1 is within ±0.03 of
  0.90 for a synthetic dataset; interval width recorded per prediction; AD status
  and conformal interval reported side by side.
- Effort: medium. **Highest scientific return of any item in §1.B.**

Second stage (optional): expose Chemprop's native `--uncertainty-method` and
`--calibration-method` from the training contract — the passthrough exists and is
simply never populated.

---

#### Q-B4 — The applicability domain has three methods, none of them classical

**Observed.** Exactly three, pinned in the contract
(`tools/prediction/qsar_contracts.py:379`): `bounding_box`, `isolation_forest`,
`similarity_matrix`. Combination is a logical OR — out-of-domain if any method
says so (`applicability_domain.py:1784`). `isolation_forest` runs with stock
sklearn defaults (`:689`). The similarity method is a top-k support score
thresholded at a percentile of the training support distribution (`:1057`),
a reasonable k-NN variant but not a distance-to-model.

Verified absent: leverage / Williams plot, Mahalanobis distance, DModX, standard
k-NN distance-to-model with a σ-based cutoff.

**Consequence.** Leverage and the Williams plot are what a QSAR reviewer and the
OECD guidance expect to see; their absence will be noticed. The OR-combination is
also unvalidated: three methods OR-ed together mark more compounds out-of-domain
than any single one, with no analysis of whether that improves the
in-domain/out-of-domain error separation.

**Task.**

- Add `leverage` (hat-matrix diagonal, h\* = 3p/n warning threshold) and the
  Williams plot (standardised residual vs leverage) — the plot is more
  recognisable than the metric.
- Add `mahalanobis` with covariance regularisation for p > n.
- Make the combination rule configurable (`any`, `all`, `majority`) and report,
  per rule, the error separation between in-domain and out-of-domain subsets.
  `metrics_by_ad_status` (`:2377`) already computes the halves.
- Files: `tools/prediction/applicability_domain.py`,
  `tools/prediction/qsar_contracts.py:379`.
- Acceptance: AD report contains a Williams plot; a combination-rule comparison
  table shows ΔRMSE between in-domain and out-of-domain per rule.
- Effort: medium.

---

#### Q-B5 — No statistical validation of any kind

**Observed.** Aggregation produces mean and **population** standard deviation
only (`tools/prediction/qsar_training_policy.py:496`). Model ranking is a
lexicographic sort on point estimates (`benchmark_toolkit.py:577`).

Verified absent across `src/`: y-scrambling / Y-randomisation, permutation test,
permutation importance, bootstrap, confidence intervals, paired t-test, Wilcoxon,
DeLong, Friedman/Nemenyi, nested cross-validation, Golbraikh–Tropsha criteria,
Q²F1/F2/F3.

**Consequence.** `best_overall_candidate` is an unqualified argmax over noisy
point estimates. With a single seed and a single split, two candidates differing
by 0.01 R² are reported as ranked. There is also no chance-correlation control:
y-scrambling is the standard, cheap demonstration that a QSAR model is not
fitting noise, and is a hard requirement in several regulatory contexts.

**Task.**

- **B5.1** y-scrambling: retrain n times (default 10) on permuted targets, report
  the null distribution of the primary metric and the observed model's position
  in it. Small, self-contained, high credibility return.
- **B5.2** bootstrap confidence intervals on test-set metrics (percentile
  bootstrap over test predictions, 1000 resamples — cheap, no retraining).
- **B5.3** paired comparison between candidates: paired bootstrap or Wilcoxon
  signed-rank over per-compound errors on the shared test set. Attach a
  significance verdict to the leaderboard so "best" means "significantly best" or
  is explicitly marked as a tie.
- **B5.4** Q²F1/F2/F3 and the Golbraikh–Tropsha acceptance criteria as reported
  statistics — a handful of formulas, and what external reviewers of QSAR work
  look for.
- Files: new `tools/prediction/statistics.py`;
  `tools/prediction/qsar_training_policy.py` (aggregation),
  `tools/prediction/benchmark_toolkit.py:577` (ranking),
  `tools/prediction/qsar_reporting.py` (facts).
- Acceptance: leaderboard carries CI bounds; ties are marked as ties;
  y-scrambling output appears in the report facts.
- Effort: medium. Depends on Q-A1 for repeated splits to exist by default.

---

#### Q-B6 — Metric set has gaps that matter for classification and for ranking

**Observed.** Regression: `mse, mae, rae, rmse, r2, spearman, kendall`
(`tools/prediction/training_orchestration.py:358`). Classification: `accuracy,
balanced_accuracy, precision/recall/f1 (macro + per class), roc_auc` (binary
only, hand-rolled Mann–Whitney at `:406`).

Verified absent: MCC, PR-AUC / average precision, Cohen's κ, Brier score, ECE,
explicit specificity, concordance correlation coefficient, enrichment factor,
BEDROC. Chemprop's CLI metric literals include `binary-mcc` and `prc`
(`qsar_contracts.py:30`) — those strings are passed through to Chemprop and never
computed or reported by Qsaria.

**Consequence.** MCC is the standard single-number metric for imbalanced binary
QSAR; PR-AUC is the standard curve metric when actives are rare. Both missing.
ROC-AUC alone on a 5 %-active dataset is close to uninformative.

**Task.** Extend `compute_classification_metrics` with `mcc`,
`average_precision`, `cohen_kappa`, `brier_score`, `specificity`; extend
`compute_regression_metrics` with `ccc` and `q2_f1/f2/f3`. Add all new names to
the aggregation tuple (`qsar_training_policy.py:496`), the `TuningMetric` literal
(`qsar_contracts.py:45`) and the report tables.

- Effort: small. Pure addition, no behaviour change.

---

#### Q-B7 — No class-imbalance handling, no threshold optimisation, no calibration

**Observed.** Nothing: no `class_weight`, `scale_pos_weight`, `is_unbalance`,
SMOTE, imblearn, over/under-sampling or stratified splitting anywhere.
`classification_threshold` is a fixed input defaulting to 0.5
(`qsar_contracts.py:764`). Calibration exists only as a **plot** —
`sklearn.calibration.CalibrationDisplay` at `tools/prediction/qsar_plots.py:515`
— with no `CalibratedClassifierCV`, Platt scaling, isotonic regression, Brier
score or ECE.

**Consequence.** Real activity datasets are imbalanced. A 0.5 threshold on an
uncalibrated LightGBM model trained on 5 % actives predicts almost nothing
positive, and the balanced-accuracy gate at `qsar_training_policy.py:24` fails
for reasons that have nothing to do with the chemistry.

**Task.**

- Add `class_weight="balanced"` to the LightGBM and sklearn backends — parameter
  passthrough.
- Add stratified splitting for classification: the random splitter does not
  stratify (`qsar_splitters.py:213`), so small classes can vanish from a fold.
- Add threshold selection on the validation fold (maximise MCC or Youden's J) and
  record the chosen threshold in the model record.
- Add probability calibration (Platt / isotonic) as opt-in post-processing,
  reporting Brier score before and after.
- Files: `tools/prediction/lightgbm_backend.py`, `qsar_splitters.py:213`,
  `training_orchestration.py`, `qsar_contracts.py`.
- Effort: medium. Pairs naturally with Q-B6.

---

#### Q-B8 — No feature selection or degenerate-column handling

**Observed.** No variance filter, correlation filter, `SelectKBest`, RFE, Boruta
or importance-based pruning. `rdkit_all` emits the full `Descriptors._descList` —
roughly 210 columns including descriptors that are constant, collinear or NaN for
whole chemotypes.

**Consequence.** Tree models tolerate this; any linear baseline added under Q-B1
will not, and Mordred (Q-B2.3) makes it acute. It also degrades the bounding-box
applicability domain, which is sensitive to constant and near-constant columns.

**Task.** Add a pre-model `feature_filter` step: drop zero-variance columns, drop
columns with NaN fraction above a threshold, optionally drop one of each pair
above a correlation threshold. Record dropped columns in the training manifest so
AD and inference stay schema-consistent (`applicability_domain.py:130` already
hashes the schema).

- Effort: small-medium. Prerequisite for Q-B1's linear models and Q-B2.3.

---

#### Q-B9 — No model interpretability at all

**Observed.** Zero SHAP usage — all 100 `shap` grep hits are the word `shape`.
LightGBM's `feature_importances_` is never read. No attention or atom
attribution, no RDKit similarity maps, no counterfactuals. Matched molecular
pairs exist only as prompt text (`agents/prompts.py:337`) and a `None`
placeholder (`agents/factories.py:492`).

What does exist is dataset-level SAR, not model explanation: SALI activity cliffs
(`tools/activity_cliffs/service.py:119`), per-row AD violation attribution
(`applicability_domain.py:1811`), and GTM chemical-space maps
(`tools/chemography/`, 5 293 lines, sharing no code with the prediction stack).

**Consequence.** A medicinal chemist asking "why did the model predict this?"
gets nothing. This is also the fifth OECD validation principle (mechanistic
interpretation) with no implementation.

**Task.**

- **B9.1** LightGBM/sklearn feature importance (gain and permutation), exported
  and plotted. Trivial and immediately useful.
- **B9.2** SHAP for tree backends — `shap.TreeExplainer` is fast and exact for
  LightGBM; add as an optional dependency, produce a summary plot and a
  per-compound waterfall for flagged compounds.
- **B9.3** Fingerprint-bit → substructure mapping, so a Morgan-bit importance
  renders as a highlighted substructure rather than "bit 1423". RDKit supplies
  the bit info; this is what makes B9.2 chemically readable.
- **B9.4 (larger)** matched molecular pairs over the curated dataset, replacing
  the prompt placeholder with a real implementation.
- Effort: B9.1 small, B9.2–B9.3 medium, B9.4 larger.

---

#### Q-B10 — Censored data is detected and then ignored

**Observed.** A relation column (`standard_relation`, `relation`, `operator`,
`activity_relation`) is detected and carried as metadata
(`tools/chemistry/activity_schema.py:236`, `:105`;
`tools/chemistry/clean_dataset.py:61,461`). **No code anywhere branches on the
relation value.** No censored-aware loss, no filtering, no interval handling.

**Consequence.** ChEMBL is full of `>10000 nM` records. Treating `>10000` as
`10000` biases every potency model toward the assay ceiling, silently.

**Task.** At minimum make the policy explicit and visible: count censored records
in the curation report and offer a documented policy choice — `drop`,
`keep_as_point` (current behaviour, which must be named), or `censor_aware`. The
third is larger work (Chemprop supports bounded losses natively —
`bounded-mse`/`bounded-mae`/`bounded-rmse` already appear in the metric literal
at `qsar_contracts.py:30`), so stage it: visibility first, bounded loss second.

- Files: `tools/chemistry/activity_schema.py`,
  `tools/curation/dataset_curation_toolkit.py`, `tools/curation/policies.py`.
- Effort: small for visibility, medium for bounded loss.

---

#### Q-B11 — Validation protocol gaps: no time split, CV is random-only

**Observed.** Three split types total: `random`, `scaffold_balanced`, `kmeans`
(`tools/prediction/qsar_splitters.py:197-273`). Cross-validation explicitly
refuses anything else: `"cross_validation currently supports
split_family='random' only."` (`qsar_validation_strategy.py:325`). No temporal
split, no stratified split, no `GroupKFold`, no Butina, no sphere exclusion, no
Lo-Hi.

Note also that `scaffold_balanced` uses non-generic Bemis–Murcko scaffolds
(`qsar_splitters.py:30`) assigned by a greedy balanced bin-pack (`:113`) — this
balances set sizes but does **not** guarantee scaffold disjointness, which is
what a scaffold split is usually taken to mean.

**Consequence.** Time-split validation is the closest proxy to prospective
performance for a project dataset and is what pharma evaluation asks for. Its
absence, plus random-only CV, puts the harder evaluation regimes out of reach.

**Task.**

- Add `time_split` taking a date/assay-year column: train on past, test on future.
- Allow scaffold and cluster families in cross-validation (lift the restriction
  at `qsar_validation_strategy.py:325`).
- Add stratified random splitting for classification (Q-B7).
- Document precisely what `scaffold_balanced` guarantees and add a strict
  `scaffold_disjoint` variant.
- Files: `tools/prediction/qsar_splitters.py`,
  `tools/prediction/qsar_validation_strategy.py:325`,
  `tools/prediction/qsar_contracts.py:26`.
- Effort: medium. Feeds §4 directly.

---

#### Q-B12 — Multitask only in Chemprop; several analyses are single-target-only

**Observed.** `multi_target_task_types` is set for Chemprop alone
(`backend_capabilities.py:52`). Hard refusals elsewhere: `"LightGBM V1 requires
exactly one target column."` (`lightgbm_backend.py:611`), `"TabICL requires
exactly one target column."` (`tabicl_backend.py:419`). Outlier analysis is
single-target (`outlier_analysis.py:93`), activity-cliff feedback is
single-target regression only (`qsar_contracts.py:591`), ensemble multiclass is
blocked at runtime (`ensemble_backend.py:375`).

**Task.** Lower priority than B1–B5. The honest short-term action is to make the
limitation visible in `describe_backend_capabilities` output and in the report,
so a user asking for a multitask ADMET panel is told which backends can serve it
rather than discovering it through a refusal mid-workflow.

---

#### Q-B13 — No OECD / regulatory framing

**Observed.** Zero occurrences of OECD, QMRF, QPRF or REACH anywhere in `src/`,
`docs/`, `plugins/` or `skills/` (all `reach` grep hits are the English word).

**Consequence.** The OECD's five principles for QSAR validation — defined
endpoint, unambiguous algorithm, defined applicability domain, appropriate
measures of goodness-of-fit/robustness/predictivity, mechanistic interpretation —
map almost one-to-one onto what a Qsaria report should contain. Principle 3 is
implemented, principle 4 partly (Q-B5, Q-B6), principles 1, 2 and 5 are not
framed at all.

**Task.** Add a QMRF-shaped export to `tools/io/report_export.py`: a structured
document with the five principles as top-level sections, populated from existing
report facts. Most of the content already exists; this is a mapping and a
template, not new computation. It is also a strong differentiator — see §3.

- Effort: medium. High external-credibility return.

---

#### Q-B14 — No 3D, no conformers, no protein-side features

**Observed.** No conformer generation (ETKDG, `EmbedMolecule`,
`EmbedMultipleConfs`), no 3D descriptors (WHIM, GETAWAY, RDF, MORSE, autocorr),
no force-field optimisation. `PredictionTaskSpec` carries `smiles_columns` and
`reaction_columns` only (`tools/prediction/backend.py:42`); `reaction_columns` is
forwarded verbatim to the Chemprop CLI.

**Task.** Deliberate non-goal for now — record it as such in the docs rather than
leaving it ambiguous. 3D QSAR and proteochemometrics are each a project, not a
task. Revisit after §4 establishes a baseline.

---

## 2. Agentic AI: what is missing

The agentic engineering here is unusually disciplined — deterministic handoffs, a
server-side state machine, artifact verification, path confinement, an LLM-free
MCP server. The gaps concentrate in three places: **the agent cannot plan, cannot
learn, and is never measured.**

---

#### A-1 — No planner, no re-planning, no reflection

**Observed.** Verified zero hits for planner classes, task graphs, task
decomposition, `replan`, reflection, self-critique or a critic agent. What stands
in for planning is: static skill/workflow catalogs matched by integer keyword
scoring (`skills/registry.py:78`, `workflows/registry.py`), a one-shot
`action_plan` from `mcp_bootstrap` that is never re-invoked on failure
(`mcp/facades/bootstrap.py:99`), prompt-encoded pipelines
(`agents/teams.py:274`), and server-side sequencing enforcement
(`mcp/qsaria/experiments.py:2116`).

**Consequence.** The system executes a fixed pipeline competently and cannot
adapt it. When a scientific result suggests a different next step — the scaffold
split collapsed, the dataset is too small for the requested protocol, the target
is nearly constant — nothing re-plans. The retry budget of exactly one
(`experiments.py:2178`) combined with "everything unrecognised is terminal"
(`mcp/qsaria/adapter.py:453`) means the first unanticipated failure ends the
experiment.

**Task, staged and deliberately conservative.** Do not add an open-ended planner
loop — it would fight the state machine, which is a genuine asset.

- **A-1.1** Make the protocol choice a *decision with recorded justification*.
  Add a `recommend_protocol` tool that inspects the curated dataset (n, target
  distribution, scaffold diversity, class balance, censored fraction) and returns
  a recommended validation protocol plus reasons. The coordinator presents it;
  the user or the agent confirms. This converts an implicit prompt decision into
  an auditable artifact.
- **A-1.2** Add a bounded repair step for a small enumerated set of recoverable
  conditions — "dataset too small for requested n_folds", "class missing from a
  fold", "feature matrix degenerate" — each with a single predefined remedy and a
  recorded `repair_applied` event. Enumerated, not open-ended.
- **A-1.3** Widen the retryable classification: currently only `TimeoutError`,
  `ConnectionError` and a specific errno set are retryable
  (`adapter.py:453-478`). OOM during training and transient GPU allocation
  failures are both terminal today and both are retryable in practice with a
  smaller compute profile.
- Files: `mcp/qsaria/adapter.py:453`, `mcp/qsaria/experiments.py:2116`, new tool
  spec in `mcp/tool_specs/qsaria.py`.
- Effort: A-1.1 medium, A-1.2 medium, A-1.3 small.

---

#### A-2 — The agent is never evaluated for correctness

**Observed.** `tests/robustness/` (15 files, ~6 000 lines) measures
*prompt-perturbation consistency*: N semantically equivalent prompts through a
real LLM, scored by agreement between runs — weighted
`{data 0.4, semantic 0.3, process 0.2, visual 0.1}`, pass threshold 0.75
(`tests/robustness/metrics.py:31`, `robustness_config.yaml`). Comparators use
Jaccard, KS tests, SBERT similarity, pHash/SSIM
(`tests/robustness/comparators.py`).

Verified zero hits for trajectory evaluation, rubric, scorecard, golden set,
ground truth or reference answer. `ToolSequenceComparator`
(`tests/robustness/tool_tracker.py:22`) compares runs to **each other**, never to
a reference trajectory. The one chemistry-truth assertion in the entire suite is
`mean SMILES validity > 0.90` (`test_autoencoder_robustness.py:552`).

CI runs `tests/unit/` only (`.github/workflows/ci.yml:59`); `tests/robustness/`
and `tests/integration/` are referenced by no workflow. Only `chembl_download` is
`enabled: true` in the shipped robustness config.

**Consequence, stated plainly: a system that is consistently wrong scores 1.0.**
Consistency is a necessary property and the harness that measures it is good
work — but it is not correctness, and nothing measures correctness.

**Task — build the missing harness. The single highest-value item in §2.**

Create `evals/` with scenario-based, assertion-bearing cases:

- **Fixed inputs**: a small set of committed datasets with known properties — one
  clean regression set, one imbalanced classification set, one with a deliberate
  duplicate across a split, one with conflicting units, one too small for the
  requested protocol, one with a near-constant target.
- **Expected decisions, not expected prose**: assert that the agent *refused* the
  unit-conflicted dataset, *flagged* the leaked row, *chose* a protocol
  appropriate to n, *reported* class imbalance, *did not promote* a model that
  failed a gate.
- **Expected numerics with tolerance**: for the clean dataset, R² within a band
  of a reference Random Forest (available once Q-B1 lands).
- **Trajectory assertions**: curation ran before training; registry ran before
  report; the report cites only artifact IDs that exist.
- Reuse what exists: `ToolSequenceComparator` for trajectory extraction, the
  `tool_events` log (`experiments.py:2234`) as the ground-truth record of what the
  agent actually did, and the typed handoffs as assertion targets.
- Acceptance: `evals/` runs headless against a fixed model, produces a pass/fail
  scorecard per scenario, and is runnable in CI on a cheap model.
- Effort: large, but decomposable — one scenario at a time, starting with the
  three correctness scenarios that map to Q-A2, Q-A3 and Q-B7.

---

#### A-3 — 180 KB of static prompts with no templating

**Observed.** `agents/prompts.py` is 180 594 bytes, 1 737 lines, 15 module-level
`list[str]` constants, and **100 % static** — zero f-strings, zero `.format()`,
zero `%s`. `QSAR_REPORT_INSTRUCTIONS` alone is 32 260 bytes ≈ 8 000 tokens, paid
on every `qsar_report_agent` run before any evidence is loaded.

**Consequence.** Every run pays for every instruction, including branches
irrelevant to the current task. The instructions encode scientific policy —
"mandatory smoke test" (`prompts.py:673`), evidence precedence, reporting rules —
in a form that cannot be tested, versioned independently, or varied by context.
Several of these policies have no code enforcement at all.

**Task.**

- **A-3.1** Move enforceable policies out of prose and into code. The "mandatory
  smoke test" is the clearest example: prompt text with no implementation. Either
  implement it as a post-training tool call or delete the instruction.
- **A-3.2** Split instructions into a stable core plus task-conditional blocks
  assembled per request. A classification run does not need the regression
  reporting rules.
- **A-3.3** Version prompt blocks and record which version produced a given
  report in the experiment manifest — `tracking/core.py:288` already implements
  MLflow prompt-version registration, currently inert (A-5).
- Effort: A-3.1 small per policy, A-3.2 medium, A-3.3 small.

---

#### A-4 — No memory across experiments; catalog retrieval is keyword scoring

**Observed.** Cross-session memory is deliberately disabled —
`enable_agentic_memory=False`, `enable_user_memories=False`,
`add_memories_to_context=False` (`agents/teams.py:142`), with a comment
explaining it caused incorrect recall of prior chemical-space analyses. That
decision was defensible. The consequence is that the only thing surviving an
experiment is the model catalog, searched by hand-tuned integer keyword scoring
(`tools/prediction/catalog.py:467`, `score += 50/25/18/10/8/6/4`).

`lancedb`, `tantivy` and `sentence-transformers` are declared dependencies with
zero usage in `src/`. Verified absent: vector store, embeddings, knowledge base,
RAG.

**Consequence.** Qsaria cannot answer "have we modelled this target before, and
what worked?" — the question that makes an agentic QSAR platform more valuable
than a script. It also cannot learn that, say, scaffold splits always collapse on
a particular assay family.

**Task.** Not general agentic memory — a **structured experiment index**.

- Index each completed experiment by: target/endpoint, dataset fingerprint, n
  compounds, task type, backends tried, best metric per split family, chosen
  protocol, governance status, failure reasons.
- Make it queryable: "what did we try for this endpoint", "which representation
  usually wins for n < 500", "which datasets failed curation and why".
- Store it as a queryable table rather than a vector index — the questions are
  structured, and a semantic search over experiment prose would reintroduce
  exactly the incorrect-recall problem that motivated disabling memory.
- Either use the declared `lancedb` for this or drop `lancedb`, `tantivy` and
  `sentence-transformers` from `pyproject.toml`. Carrying unused heavy
  dependencies is its own cost.
- Files: new `tools/prediction/experiment_index.py`, wired into
  `mcp/qsaria/experiments.py` completion path; new MCP read tool.
- Effort: medium. Pairs with §4 — the benchmark produces exactly this data.

---

#### A-5 — MLflow is wired, undeclared, and partly dead

**Observed.** `tracking/core.py` implements `MLflowTracker` with session, run and
tool-call tracking plus prompt-version registration. But `mlflow` is **not in
`pyproject.toml`**, so `_initialize_mlflow` raises `ImportError` and disables
tracking in every default install (`core.py:43`). `track_tool_call` and
`track_session` have zero call sites in `src/`. All 308 lines of
`tracking/decorators.py` are applied to nothing.
`storage/layout.py:44 OutputLayout.manifest_rel_path` is dead code.

**Task.** Decide, then act. Either declare `mlflow` as an optional extra, wire
`track_tool_call` into `tool_adapter._invoke` and make tracking real; or delete
`tracking/decorators.py` and the unused methods. The current state — a tracking
subsystem that looks implemented and is inert — is worse than either.

- Effort: small either way. Do not leave it as is.

---

#### A-6 — Shared mutable session state with a destructive clear

**Observed.** `teams.py:215` passes **one shared dict object** to every QSAR agent
and to the `Team` (`:263`); `factories.py:113` merges defaults in place. The
report agent's pre-hook `compact_qsar_report_session_context` whitelists about six
keys and then calls `session_state.clear()`
(`agents/qsar_agent_handoffs.py:458`).

**Consequence.** If Agno does not copy per agent, that clear is a team-wide
destructive mutation executed mid-workflow. Even if it is safe today, it is safe
by accident of a dependency's internals.

**Task.** Verify the aliasing empirically with a test that mutates state in one
agent and asserts isolation in another. If shared, replace `clear()` with
construction of a new scoped dict for the report agent. Pin the behaviour with
the test either way.

- Effort: small. Risk if wrong: high and hard to debug.

---

#### A-7 — No cost controls, no approval gate before expensive work

**Observed.** Verified zero hits for token budget, cost limit, spend limit or max
cost. The only rate limit in `src/` is the ChEMBL HTTP client's 10 req/s
(`tools/databases/chembl.py:252`). The only confirmation flag anywhere is
`confirm_explicit_export_request` on a training export
(`qsar_training_toolkit.py:928`). Codex agents run with
`default_tools_approval_mode = "auto"`, so MCP tools — including training and
catalog writes — auto-approve. Detached operations cannot be cancelled at all
(`experiments.py:292`, `"cancellation": False`).

**Consequence.** A benchmark campaign is 7 candidates × up to 50 Optuna trials,
sequential, with no estimate shown beforehand, no budget, and no way to stop it
once started.

**Task.**

- **A-7.1** Pre-flight cost estimate: before a benchmark or tuned training, return
  an estimate (candidates × trials × observed per-fit time from the feature cache
  or a quick probe) and require confirmation above a configurable threshold.
- **A-7.2** Implement cancellation for detached operations — the state directory,
  the PID and the heartbeat already exist (`operations.py:239`); a
  `qsaria_cancel_operation` tool writing a cancel flag the worker polls is a
  contained change.
- **A-7.3** Per-experiment token/cost accounting recorded in the experiment state,
  even if no limit is enforced initially.
- Effort: A-7.1 medium, A-7.2 medium, A-7.3 small.

---

#### A-8 — Candidate failure aborts an entire benchmark campaign

**Observed.** Candidates run in a plain sequential loop
(`benchmark_toolkit.py:892`); an exception is logged and re-raised, ending the
campaign. No per-candidate error row, no resume. Features are cached
(`qsar_training_toolkit.py:132`) but trained models are not, so a rerun retrains
everything.

**Task.** Catch per candidate, record a failure row in the leaderboard, continue.
Add campaign resume keyed on the existing campaign seed policy. Parallelise across
candidates where the compute profile allows.

- Effort: small for failure isolation, medium for resume and parallelism.
  Required by §4.

---

#### A-9 — Report claims are bound to evidence at packet level, not claim level

**Observed.** `report_build_context` (`experiments.py:1007`) returns
`report_facts`/`report_tables` packets tagged with originating agent, tool name
and `call_id`, with an explicit `evidence_precedence` ordering. The saved report
is SHA-256 verified. This is better than most systems.

**Gap.** Nothing checks that a *sentence* in the report corresponds to a *number*
in `report_facts`. That binding is enforced only by prompt text
(`prompts.py:819,875,884,896`).

**Task.** Add a post-generation numeric audit: extract numerals from the report
narrative, match against values in `report_facts` within tolerance, flag unmatched
numbers as unsourced. This needs no LLM and catches the failure mode that matters
— a fabricated or drifted metric in prose.

- Files: `tools/prediction/qsar_reporting.py`, `mcp/qsaria/experiments.py:1625`
  (extend the existing validation step).
- Acceptance: a report containing an invented R² fails validation.
- Effort: medium. Natural companion to A-2.

---

#### A-10 — Three orchestration runtimes with diverging behaviour

**Observed.** The Agno/Chainlit runtime (two LLM-coordinated teams), the generic
MCP profile (102 tools, flat), and the Qsaria MCP profile (53 tools plus the
experiment state machine). They do not share a planner. The durable experiment
lifecycle, the role state machine, the retry policy, the artifact provenance and
the detached operations exist **only** in the Qsaria MCP path. The Agno path
writes benchmark output to `benchmark/<time_ns hex>` with no registry
(`benchmark_toolkit.py:136`) and records no manifests.

**Consequence.** The native web application — the one the README presents first —
is the least auditable surface.

**Task.** Decide which runtime is primary and say so in the docs. Then lift the
experiment lifecycle into a shared layer both runtimes call, rather than
reimplementing provenance in the Agno path. If the Agno app is a demo surface,
label it as such.

- Effort: large. Architectural decision first, code second. Do not start before
  §1.A and A-2.

---

## 3. Comparison with similar systems

Survey date 2026-11-25. Every row was checked against a primary source (arXiv
abstract page, publisher record, GitHub repository, project site). Rows that
could not be confirmed are marked. **The Qsaria row is filled from the audit in
§0–§2 of this document, not from the project's own abstract** — where the two
disagree, the audit wins.

### 3.1 Two findings to act on first

**MolAgent and ChemLint are both real, and both are already in Qsaria's own
bibliography.** The CS3-2026 poster
(`infochim.u-strasbg.fr/IMG/pdf/p2_abraham_brisset_abstract_poster_cs3-2026.pdf`)
cites them as refs [3] and [4]. They are the two closest pieces of prior art and
are treated as the primary comparators below.

**Two bibliography items to verify by hand before the poster goes to print:**

1. Poster ref [4] cites ChemLint as `10.26434/chemrxiv.15000386/v1`. The DOI
   resolves (302) but ChemRxiv blocks automated retrieval, so the record could
   not be read. The number is in the **pre-2021 Figshare-style ChemRxiv format**;
   2026 ChemRxiv preprints use `10.26434/chemrxiv-2026-xxxxx`. The ChemLint
   *software* is unambiguously real — it is the *preprint DOI* that is
   unconfirmed, and the same string appears in the project's README.
2. Poster ref [2] can be upgraded: Lu et al., *Towards end-to-end automation of
   AI research*, is now peer-reviewed — ***Nature* 651, 914–919 (2026)**,
   DOI `10.1038/s41586-026-10265-5`.

Also relevant when citing activity cliffs: MoleculeACE
(van Tilborg, Alenicheva, Grisoni, *JCIM* 2022, 62(23), 5938–5951) **has a 2023
correction**, DOI `10.1021/acs.jcim.3c01576`, for a train/test splitting bug
caused by mislabelled activity cliffs. Use the corrected repository numbers, not
the 2022 figures.

### 3.2 The real comparators — agents that actually train models

Only four verified systems build predictive models rather than calling
pre-existing ones.

**MolAgent** — the closest architectural analogue.
Gómez-Tamayo et al. (Univ. Antwerp / Open Analytics / Johnson & Johnson),
***JCIM* 2025, 65(20), 10808–10818**, DOI `10.1021/acs.jcim.5c01938`.
Code: `github.com/openanalytics/MolAgent`, GPL-3.0.
Trains via the `AutoMol` package with **nested CV + Bayesian HPO + ensemble
stacking**; features are a pretrained Bottleneck encoder, ECFP and RDKit
descriptors. Scaffold splitting explicitly to avoid leakage. **Two MCP servers**
(model + data), smolagents/LiteLLM examples, a Gradio chatbot, and a Claude Code
plugin. Model registry as `model_registry.json`. Output is a model card plus a
self-contained HTML/PDF, not a narrative report. No applicability domain, no
activity cliffs. Evaluated on TDC ADMET (Caco2, Lipophilicity, hERG, AMES) over
5 seeds.
*Overlap with Qsaria is high: MCP + registry + scaffold splits + nested CV are
all already there, and nested CV is something Qsaria does not have.*

**ChemLint** — the closest capability analogue.
Derek van Tilborg (+ Grisoni), `github.com/derekvantilborg/ChemLint`, MIT,
FastMCP. ~150 tools across 13 categories including **40 machine-learning tools**
(33 scikit-learn algorithms, CV, grid/random search), **activity cliffs**, and
**8 data-leakage checks for split quality**. No applicability domain, no
uncertainty, no registry, no deep-learning backends.
*Architecturally it is a pure MCP tool server with no agent of its own — the
client is the agent. Qsaria orchestrates; ChemLint exposes. But note that it
already ships the leakage checks that Q-A3 proposes to build.*

**ADMET-EvO** — the closest on methodological rigour.
Zhou et al., arXiv:2609.10121, `github.com/OTeam-AI4S/ADMET-EvO`, MIT.
Single agent where the LLM is only the *planning policy* and a deterministic
evaluator does all fitting and metrics — the same separation of concerns Qsaria
achieves with its typed handoffs. Curation desalts, neutralises, standardises,
and excludes near-analogues at Tanimoto ≥ 0.8. **The chosen recipe is SHA-256
hash-locked before the test set is loaded**, then retrained over ≥5 seeds.
Applicability domain is partial — a "familiarity score" (maximum fingerprint
similarity to training data) with a risk–coverage curve. 22 TDC ADMET tasks.
*This is the validation-protocol bar to match. The hash-lock is cheap to adopt
and should go into §4.*

**Closed-loop Auto Research for Molecular Property Prediction.**
Ning et al. (CMU / DP Technology), arXiv:2606.22731, Apache-2.0. Multi-agent
feature/model/evidence specialists plus a meta agent over a shared trial lineage.
GBDTs only. **Three-layer leakage filter**: InChIKey dedup, reject any external
file with >5 % test overlap, drop molecules at ECFP4 Tanimoto ≥ 0.9 to a test
molecule. Evaluated on **36 endpoints** — TDC 22, MoleculeNet 10, Polaris 4.
*The result to internalise: model-search gain collapsed from **+0.041 on
validation to +0.003 on test**, and agent-curated external data went **+0.022
validation to −0.019 test**. Agent-reported validation gains routinely fail to
transfer. §4 must separate discovery from held-out certification.*

### 3.3 The uncomfortable baseline

**Deep Origin**, *Evaluating Autonomous LLM Agents Across Molecular Prediction
and Optimization Benchmarks*, bioRxiv 2026-10-08, DOI
`10.64898/2026.10.01.755314`. Not a new system: it runs **stock Codex and Claude
Code** with no domain scaffolding, no human scientific guidance, 15 h wall clock
on one H100. The agents built TabM ensembles over RDKit 2D + Mordred + PaDEL +
conformer 3D descriptors + frozen GIN embeddings, selected by scaffold-split
validation over ≥5 runs. They **beat the TDC leaderboard reference on 18 of 21
tasks**, placed equivalent to 4th and 6th of 103 entries on OpenADMET
ExpansionRx, and beat the best published PXR entry.

**Implication for Qsaria, stated plainly: raw TDC numbers cannot be the
differentiator.** A general-purpose coding agent already clears that bar. The
defensible claims are applicability domain, activity cliffs, registry,
reproducibility, the narrative report, and the product surface — which is exactly
what §1 and §2 of this plan are about.

A related and more encouraging precedent: Southiratn, Schwaller & Genheden,
*Accelerating QSAR Modeling with Human-AI Collaboration*, **NeurIPS 2026 Workshop
on Agentic Systems for Molecular Sciences** — a human-in-the-loop agent campaign
placed **13th of 100** in the OpenADMET PXR blind challenge. Human-steered agent
QSAR is an established publication pattern.

### 3.4 Identity and capability

Legend: ✅ yes · ⚠️ partial · ❌ no · ? not verified

| System | Year / venue | Open source | Trains models | Architecture | Evaluation | Conf. |
| --- | --- | --- | --- | --- | --- | --- |
| **Qsaria** | 2026, CS3 poster | MIT | ✅ Chemprop + LightGBM + TabICL | Multi-agent; Agno + Chainlit + MCP (3 runtimes) | **none published** | — |
| **MolAgent** | 2025, *JCIM* 65(20) | ✅ GPL-3.0 | ✅ AutoMol, nested CV + Bayes HPO | 2 MCP servers + smolagents; Claude Code plugin | TDC ADMET ×4, 5 seeds | High |
| **ChemLint** | 2026, ChemRxiv ⚠️ | ✅ MIT | ✅ 33 sklearn algorithms | MCP server only (FastMCP), no agent | none verified | Med-High |
| **ADMET-EvO** | 2026, arXiv:2609.10121 | ✅ MIT | ✅ GBDT/RF/SVM/GNN/Uni-Mol2 | LLM as planning policy + deterministic evaluator | TDC ADMET 22 | High |
| **Closed-loop Auto Research** | 2026, arXiv:2606.22731 | ✅ Apache-2.0 | ✅ XGB/CatBoost/LGBM | Multi-agent + meta agent over trial lineage | 36 endpoints (TDC+MolNet+Polaris) | High |
| **Deep Origin study** | 2026, bioRxiv | ? | ✅ (agent-built TabM) | Stock Codex / Claude Code, no scaffolding | TDC 21, ExpansionRx, PXR, PMO | High |
| **DrugAgent** (Liu) | 2024, arXiv:2411.15692 | ⚠️ anonymised only | ✅ sklearn RF | Planner + Instructor, MLAgentBench actions | 3 tasks, random splits | High |
| **ChemCrow** | 2024, *Nat. Mach. Intell.* 6(5) | ✅ (⚠️ licence disputed) | ❌ 18 tools | LangChain, single agent | expert + LLM-judge | High |
| **Coscientist** | 2023, *Nature* 624 | ⚠️ partial | ❌ | GPT-4 + search/code/robotics | 6 tasks, real wet-lab | High |
| **ChemToolAgent** | 2025, NAACL Findings | ✅ (ChemMCP) | ❌ | ReAct, MCP toolkit | MMLU/SciBench/GPQA-Chem | High |
| **El Agente Q** | 2025, *Matter* 8 | ? | ❌ (quantum chemistry) | Hierarchical multi-agent, **LaTeX reporting** | 6 exercises, >87 % | High |
| **QSARtuna** (AstraZeneca) | 2024, *JCIM* 64(14) | ✅ Apache-2.0 | ✅ Optuna 3-stage | CLI pipeline, **not an agent** | — | High |
| **Chemprop v2** | 2024-26, *JCIM* | ✅ MIT | ✅ D-MPNN | library / CLI | — | High |
| **DeepChem** | 2024 (2.8.0) | ✅ MIT | ✅ | library | — | High |
| **ZairaChem** | 2023, *Nat. Commun.* 14 | ✅ GPL-3.0 | ✅ AutoML stack | pipeline | — | High |
| **DeepAutoQSAR** | vendor | ❌ commercial | ✅ | Maestro product | vendor claim 20/22 TDC | Med |
| **VEGA** | ongoing | free | ❌ pre-trained | desktop app | ECHA REACH screening | Med-High |
| **OECD QSAR Toolbox** | v4.7, 2024 | free | ❌ | desktop, read-across | regulatory use | High |

### 3.5 QSAR-rigour matrix — the table that actually differentiates

| System | Curation | Splitting | Validation protocol | **Applicability domain** | **Uncertainty** | **Activity cliffs** | Interpretability | Registry | Serving | **Auto report** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Qsaria (audited)** | ✅ ChEMBL pipeline, conflict-aware dedup | ⚠️ random + scaffold + cluster, but **default is random only**; no temporal | ⚠️ holdout/repeated/CV, **CV random-only, no statistics, no baselines** | ✅ 3 methods (box, isolation forest, k-NN similarity); **no leverage/Williams** | ❌ **none** (ensemble spread, explicitly not calibrated) | ✅ SALI, first-class | ❌ none | ✅ catalog + registry | ✅ MCP + Chainlit | ✅ **narrative, evidence-bound, SHA-256 verified** |
| **MolAgent** | ⚠️ TDC retrieval | ✅ scaffold | ✅ **nested CV** + Bayes HPO | ❌ | ⚠️ classification calibration diagram | ❌ | ❌ | ✅ JSON | ⚠️ MCP predict | ⚠️ model card + HTML |
| **ChemLint** | ✅ cleaning | ✅ + **8 leakage checks** | ✅ CV + grid/random | ❌ | ❌ | ✅ | ⚠️ | ❌ | ⚠️ MCP tools | ⚠️ data-quality report |
| **ADMET-EvO** | ✅ desalt/neutralise, Tanimoto ≥0.8 exclusion | ✅ TDC official + overlap checks | ✅ **SHA-256 hash-lock before test load, ≥5 seeds** | ⚠️ familiarity score + risk–coverage | ⚠️ conformal assessed, unreported | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Closed-loop Auto Res.** | ✅ 3-layer leakage filter | ✅ official splits | ✅ held-out certification separate from search | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Deep Origin** | ⚠️ traces audited | ✅ TDC scaffold | ✅ ≥5 seeds, mean ± SD | ❌ | ⚠️ bootstrap CIs | ❌ | ❌ | ❌ | ❌ | ❌ |
| **DrugAgent** | ❌ | ❌ **random splits** | ⚠️ validation-driven selection | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **QSARtuna** | ✅ dedup + transform | ✅ **best in class**, incl. **temporal**, Butina, stratified group | ✅ Optuna 3-stage + replicated CV | ❌ (MODI ≠ AD) | ✅ **MAPIE conformal + Venn-ABERS** | ❌ | ✅ SHAP | ⚠️ MLflow optional | ⚠️ CLI | ❌ |
| **Chemprop v2** | ❌ | ✅ 5 types | ✅ | ❌ | ✅ **10 UQ + 10 calibration methods incl. conformal** | ❌ | ❌ | ❌ | ⚠️ CLI | ❌ |
| **DeepChem** | ⚠️ | ✅ 10 splitters | ✅ | ❌ | ⚠️ MC-dropout | ❌ | ❌ | ❌ | ❌ | ❌ |
| **ZairaChem** | ✅ | ✅ | ✅ | ❌ | ⚠️ ensemble | ❌ | ❌ | ❌ | ❌ | ⚠️ templated |
| **VEGA** | — | — | — | ✅ **Applicability Domain Index — strongest in the field** | ⚠️ | ❌ | ⚠️ read-across | — | desktop | ⚠️ |
| **OECD QSAR Toolbox** | ✅ | — | read-across | ⚠️ category framing | ❌ | ❌ | ✅ mechanistic | — | desktop | ✅ **regulatory template** |

### 3.6 What Qsaria can defensibly claim — and what it cannot

**Defensible today:**

1. **Activity cliffs as a first-class pipeline output.** Only ChemLint also has
   them, and ChemLint is a toolbox, not an agent. Effectively unique among
   agentic systems.
2. **Applicability domain inside a modern agentic training pipeline.** DeepChem,
   Chemprop, QSARtuna, Polaris, TDC and DeepMol all lack AD; ADMET-EvO has a
   similarity proxy. The real prior art is VEGA's Applicability Domain Index and
   OCHEM, neither of which trains modern models. Claim *"AD as a first-class
   output of an agentic training pipeline"* — not *"first AD in QSAR"*.
3. **Narrative auto-generated scientific report.** Nobody else writes one.
   ZairaChem and the OECD Toolbox emit templates; El Agente Q does LaTeX for
   quantum chemistry; The AI Scientist writes papers but has no chemistry notion.
   Qsaria's evidence-bound, SHA-256-verified report with typed facts packets is a
   clean claim — and A-9 would strengthen it further.
4. **Chemprop as an agentic backend.** No verified agentic QSAR system uses it.
   MolAgent uses a Bottleneck encoder, ADMET-EvO uses Uni-Mol2, Closed-loop uses
   GBDTs, Deep Origin's agents used TabM + GIN.
5. **Model registry + inference serving.** Only MolAgent (JSON) and QSARtuna
   (optional MLflow) have any registry; essentially nobody in the open-source
   QSAR stack serves inference.

**Not defensible, and should not be claimed:**

6. **Uncertainty.** Qsaria has none (Q-B3). Chemprop v2 ships 10 UQ and 10
   calibration methods including conformal; QSARtuna ships MAPIE + Venn-ABERS;
   ADMETlab 3.0 ships evidential deep learning. This is the single largest
   capability deficit against the direct comparators, and Chemprop — already a
   Qsaria dependency — has the machinery sitting unused.
7. **TabICL as a novelty.** TabICL and TabPFN are now stock AutoGluon models,
   TabPFN is already a QSARtuna dependency, and Deep Origin's agents reached for
   TabM/TabPFN/TabICLv2 independently. Frame TabICL as a supported backend, not a
   differentiator.
8. **Raw benchmark numbers.** See §3.3.
9. **Validation rigour.** Today the default protocol is a single random split
   (Q-A1) with no statistics (Q-B5) and no baselines (Q-B1), against MolAgent's
   nested CV, ADMET-EvO's hash-locked ≥5-seed protocol, and Closed-loop's
   three-layer leakage filter. This is currently a weakness, not a strength.

### 3.7 What this changes in the plan

| Finding | Action in this document |
| --- | --- |
| Every comparator uses scaffold splits by default; Qsaria does not | Raises **Q-A1** to the top of Phase 0 |
| ChemLint ships 8 leakage checks; Closed-loop ships a 3-layer filter | **Q-A3** is table stakes, not a nice-to-have |
| MolAgent uses nested CV; ADMET-EvO hash-locks and runs ≥5 seeds | Add nested CV and a **hash-lock-before-test-load** step to **Q-B5** and §4 |
| Chemprop v2 has 10 UQ methods, already a dependency | **Q-B3**'s cheap first move is to expose Chemprop's native UQ, in parallel with the backend-agnostic conformal wrapper |
| QSARtuna has temporal splits and SHAP; it is the richest non-agentic comparator | **Q-B11** (time split) and **Q-B9** (SHAP) have a concrete reference implementation to match |
| VEGA's ADI is the strongest AD in the field | **Q-B4** should cite it as prior art and position against it, not ignore it |
| Closed-loop: +0.041 validation → +0.003 test | §4 must separate discovery from certification and include a **matched classical AutoML control** |
| Deep Origin: stock coding agents beat TDC 18/21 | §4.8's success statement must rest on AD/cliffs/report/reproducibility, not leaderboard position |
| No benchmark exists for "agent builds a QSAR model end to end" | A real, citable gap — DrugAgent's authors say so explicitly. **A-2**'s `evals/` harness is publishable work, with BixBench as the structural template and ScienceAgentBench (whose comp-chem tasks include ClinTox and IGC50 model building) as the nearest runnable harness today |

---

## 4. Benchmarking plan

### 4.0 Why this is the gating work

Three of the four audits converged on the same finding: Qsaria currently cannot
produce a number that anyone outside the project can check. There is no public
benchmark loader, no committed reference dataset beyond a 12-row toy CSV
(`examples/prediction/demo_solubility.csv`, 181 bytes, referenced by nothing), no
baseline model, no statistical comparison, no cross-run aggregation, and CI trains
no real model. The single training smoke test that would train a neural model is
triple-gated and never runs (`tests/unit/test_hyperparameter_tuning.py:725`).

Until §4 lands, every performance claim in the README is unverifiable.

### 4.1 Prerequisites

These §1–§2 items block the benchmark and should land first:

| Blocker | Why |
| --- | --- |
| Q-A1 (multi-family default protocol) | Otherwise every leaderboard column is `None` and all models are `workflow_demo` |
| Q-A2 (standardisation unification) | Otherwise benchmark numbers are contaminated by pipeline-introduced leakage |
| Q-A4 (classification backends) | Otherwise half the benchmark datasets cannot run |
| Q-B1 (baselines) | A benchmark without Random Forest and a dummy is not a benchmark |
| A-8 (failure isolation, resume) | A multi-candidate × multi-dataset campaign will hit a failure |

Q-A3 (leakage report), Q-B5 (statistics) and Q-B6 (metrics) should land during,
not before — they change what is reported, not whether it runs.

### 4.2 Dataset layer — the missing foundation

**Nothing exists today.** Verified zero hits for MoleculeNet, DeepChem, TDC /
PyTDC, Polaris, OpenADMET, MoleculeACE, Lo-Hi, DOCKSTRING and every named
MoleculeNet dataset. `admet-ai` is in the `prediction` extra with zero imports —
install weight implying a capability that does not exist; drop it or use it.

**Task B-1: build `tools/benchmarks/datasets.py`.** A loader registry returning
`(dataframe, canonical_split_or_None, provenance)` with checksums and local
caching under a benchmark cache root. Each loader records source URL, version,
retrieval date and a SHA-256 of the raw file, written into the campaign manifest.

Recommended sources, in priority order:

| Priority | Source | What it gives | Notes |
| --- | --- | --- | --- |
| 1 | **Therapeutics Data Commons (TDC)** | Many small-molecule ADMET/HTS tasks, regression and classification, with prescribed scaffold splits and published leaderboards | One dependency (`PyTDC`), one API, widest span of endpoints. `identify_qsar_columns` already anticipates TDC's `Drug`/`Drug_ID` naming (`dataset_curation_toolkit.py:553,566`) — finish what that implies. **Start here.** |
| 2 | **MoleculeNet** | ESOL, FreeSolv, Lipophilicity, BACE, BBBP, Tox21, SIDER, ClinTox, HIV, MUV, QM9 | The field's common reference; essential for comparability with published numbers. Use the canonical scaffold splits, not regenerated ones. |
| 3 | **OpenADMET** | Open, modern, actively curated ADMET data with explicit emphasis on data quality and prospective evaluation | Directly aligned with Qsaria's curation strengths. Verify current dataset inventory and licence at integration time. |
| 4 | **Polaris** | Benchmarks with enforced splits and a hosted leaderboard, designed against the methodological flaws of older suites | Strong external-credibility return: a Polaris submission is a third-party-checkable result. |
| 5 | **MoleculeACE** | ChEMBL bioactivity sets built specifically around activity cliffs | Directly exercises Qsaria's existing SALI module (`activity_cliffs/service.py`) — a benchmark where the platform's distinctive feature should pay off. |
| 6 | **Lo-Hi / low-similarity splits** | Split protocols targeting the optimism of random splits | The measurement that justifies Q-A1. |
| 7 | **In-house ChEMBL targets** | Via the existing `ChemblToolkit` | Exercises the full curation path end to end, which public pre-split datasets bypass entirely. Keep a frozen, checksummed snapshot — ChEMBL changes between releases. |

Licensing: record each dataset's licence in the loader and in the manifest. Do
not commit raw third-party data — cache it, checksum it, and commit only the
checksum plus the loader.

**Task B-2: commit a small, real reference dataset.** Three to five hundred
compounds, one regression and one classification, permissively licensed, committed
to the repo. This is what CI trains on and what the `evals/` harness (A-2) asserts
against. Note `build_qsar_split_payload` hard-rejects n < 10
(`qsar_splitters.py:206`), so the current 12-row demo file is below any meaningful
floor — delete it or replace it.

### 4.3 Split protocol

Every dataset runs under a fixed matrix. The point is to measure the *gap* between
regimes, not to report a single number.

| Split | Purpose | Status |
| --- | --- | --- |
| Random | Optimistic reference; comparability with published numbers | Exists (`qsar_splitters.py:213`) |
| Canonical published split | Direct comparison with the literature | **Missing** — loaders must pass through the provided split, not regenerate it (B-1) |
| Scaffold (Bemis–Murcko) | Standard generalisation test | Exists but balanced, not disjoint — clarify and add a strict variant (Q-B11) |
| Cluster (k-means on features) | Chemotype-level extrapolation | Exists (`:258`) |
| Time split | Closest proxy to prospective use | **Missing** (Q-B11) |
| Low-similarity (Lo-Hi style) | Worst-case extrapolation | **Missing** |

Report, for every model, the **delta between random and the hardest split** — the
machinery exists at `qsar_training_policy.py:597` and is unreachable only because
of Q-A1. Fixing Q-A1 activates it.

Repetition: ≥3 seeds per (dataset, split, model) cell. Single-seed results carry
`std = 0.0` by construction (`qsar_training_policy.py:496`) and are not
interpretable.

### 4.4 Model matrix

| Tier | Models | Purpose |
| --- | --- | --- |
| Baseline | Dummy (mean/majority), Random Forest on ECFP4, Random Forest on RDKit-2D | The reference everyone recognises. **Required** (Q-B1) |
| Classical | SVM, PLS (regression), k-NN | Cheap, strong on small datasets, standard in QSAR papers |
| Current | LightGBM × {rdkit_all, morgan_only, morgan_count_only}, Chemprop, TabICL | What Qsaria ships |
| Extended | Any representation added under Q-B2 | Measures whether new descriptors earn their cost |
| **Control** | A matched classical AutoML run (AutoGluon or an Optuna sweep over the baseline tier) under the same budget | Closed-loop Auto Research (arXiv:2606.22731) showed an agent's +0.042 Polaris gain against a classical control's +0.006. Without this control, an agentic-pipeline claim is unfalsifiable |

Tuning budget must be equal across tiers, or recorded per candidate (Q-A5).

### 4.4b Protocol discipline: separate discovery from certification

Adopt ADMET-EvO's protocol (arXiv:2609.10121), which is cheap and is the current
bar for credibility:

1. All model/representation/hyperparameter search happens on training and
   validation data only.
2. The chosen recipe — backend, representation, hyperparameters, split
   definition, seed policy — is serialised and **SHA-256 hash-locked before the
   test set is loaded**. Qsaria already hashes artifacts
   (`mcp/qsaria/contracts.py:363`) and persists a replayable seed policy
   (`qsar_training_policy.py:103`), so this is a small addition.
3. Only then is the test set read, and the locked recipe retrained over ≥5 seeds.
4. The report states the lock hash and the number of seeds.

This directly counters the failure mode that Closed-loop Auto Research measured:
model-search gains of **+0.041 on validation collapsing to +0.003 on test**, and
agent-curated external data going **+0.022 validation to −0.019 test**. Any
number Qsaria reports from a search-then-evaluate loop without this separation
should be assumed optimistic.

Also adopt **nested cross-validation** for the model-selection tier (MolAgent
already does this): the outer loop estimates generalisation, the inner loop
selects hyperparameters. The current single-level Optuna-on-the-validation-fold
design reports a selection-biased estimate.

### 4.5 Metrics and statistics

Regression: RMSE, MAE, R², Q²F1–F3, Spearman, CCC, plus conformal coverage and
mean interval width once Q-B3 lands.
Classification: MCC, PR-AUC, ROC-AUC, balanced accuracy, Brier score, plus
calibration error.

Every metric reported with a bootstrap confidence interval (Q-B5.2). Pairwise
model comparison by paired bootstrap or Wilcoxon over per-compound errors
(Q-B5.3); "best" without a significance verdict is not reported as best.

Also report per cell: fit time, prediction time, peak memory, applicability-domain
coverage, and the in-domain/out-of-domain error gap
(`applicability_domain.py:2377` already computes the halves).

### 4.6 Infrastructure

**Task B-3: a results schema and store.** Today each campaign writes to
`benchmark/<time_ns hex>` with no index (`benchmark_toolkit.py:136`), and
`catalog.search()` cannot query by dataset (`catalog.py:467`). Nothing can be
aggregated across datasets or dates.

Define one long-format row — `(run_id, dataset, dataset_version, split, seed,
model, representation, tuning_budget, metric, value, ci_low, ci_high, fit_time,
env_hash)` — written to Parquet and appended across campaigns. Everything else
(leaderboards, plots, regression detection) becomes a query over that table.

**Task B-4: a CLI driver.** There is no CLI today (`cli.py:338` is a chat REPL)
and no benchmark script; the only way to run a campaign is to ask an LLM. A
benchmark must be runnable headless and deterministically:

```bash
uv run qsaria-bench --suite tdc-admet --models baseline,current --splits random,scaffold --seeds 3 --out results/
```

**Task B-5: multi-dataset campaign support.** `benchmark_qsar_models` takes a
scalar `train_csv` (`benchmark_toolkit.py:835`). Add a campaign layer above it
that iterates datasets, isolates failures (A-8), supports resume, and emits rows
into the B-3 store.

**Task B-6: complete the reproducibility manifest.** Current capture is seeds
(`qsar_training_policy.py:103`) plus hardware (`:312`) — good but insufficient.
Add git SHA, resolved dependency versions (RDKit, LightGBM, Chemprop, TabICL,
torch, scikit-learn), dataset checksum and the `uv.lock` hash. Two campaigns run
on different days are not currently provably comparable.

Also: `seed_policy_reporting_text` is hard-coded French
(`qsar_training_policy.py:202`) and asserted verbatim in a test. For a benchmark
artifact consumed externally, make it localisable or English.

**Task B-7: make CI scientific.** Today CI runs lint plus mocked unit tests; every
backend is faked and no real model is ever trained. Add a nightly job that trains
the real stack on the committed reference dataset (B-2) and asserts metrics within
a tolerance band. Without this, a numerics regression in LightGBM, Chemprop,
TabICL or RDKit ships green.

### 4.7 Phasing

| Phase | Content | Exit criterion |
| --- | --- | --- |
| **0** | Q-A1, Q-A2, Q-A4, Q-B1, A-8 | Default protocol yields ≥2 split families; a Random Forest baseline trains; a classification benchmark produces ≥3 candidates |
| **1** | B-1 (TDC + MoleculeNet loaders), B-2 (reference dataset), B-3 (results store), B-4 (CLI), §4.4b hash-lock | One command reproduces a full campaign on one dataset and appends rows to the store; the recipe hash is recorded before the test set is read |
| **2** | Q-B5 (incl. nested CV), Q-B6, Q-A3, Q-A5, B-5, B-6, AutoML control | Multi-dataset campaign with CIs, significance verdicts, leakage report, complete manifest, and a matched classical control |
| **3** | Q-B3 (conformal), Q-B4 (leverage/Williams), Q-B11 (time split, Lo-Hi) | Uncertainty and hard-split results reportable |
| **4** | B-7 (nightly scientific CI), OpenADMET/Polaris/MoleculeACE loaders, A-2 (`evals/`) | External-checkable results; agent correctness measured |

### 4.8 What success looks like

A single table, regenerable by one command, reporting for every
(dataset × split × model) cell: the metric with a confidence interval, the delta
against the random-split reference, the delta against the Random Forest baseline,
the delta against the matched AutoML control, applicability-domain coverage, and
conformal coverage. Plus a paragraph that can honestly say: on these N public
datasets under scaffold splits, with the recipe hash-locked before the test set
was read, Qsaria's automated pipeline reaches within X % of a tuned expert
baseline and within Y % of a matched classical AutoML control, with calibrated
uncertainty, an applicability-domain-stratified error profile, and a complete
provenance trail.

Per §3.3, the last three clauses are what make the paragraph worth writing —
leaderboard position alone is already achievable by a general-purpose coding
agent with no domain scaffolding.

That paragraph is currently unwritable. Everything in this plan is in service of
making it writable.

---

## Appendix — verified absences

Every item below returned zero matches in `src/` at the analysed commit. Recorded
so future work does not re-derive them.

**Descriptors / representations:** mordred, chemberta, molformer, unimol, cddd,
mol2vec, avalon, atom-pair, topological torsion, pharmacophore (Gobbi), ErG,
USRCAT, ISIDA.

**3D:** conformer, EmbedMolecule, EmbedMultipleConfs, rdDistGeom, Descriptors3D,
MMFF/UFF optimisation, WHIM, GETAWAY, RDF, MORSE, autocorrelation.

**Splitting:** time_split, temporal_split, butina, sphere_exclusion, lohi,
GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, StratifiedKFold, TimeSeriesSplit,
stratify.

**Statistics:** y_scrambling, permutation_test, permutation_importance, bootstrap,
confidence_interval, ttest, wilcoxon, mannwhitney, friedman, delong, nested CV,
golbraikh, tropsha, q2f1.

**Metrics:** matthews_corrcoef, cohen/kappa, brier_score, average_precision,
bedroc, concordance_correlation.

**Uncertainty:** conformal, nonconformity, mapie, venn, evidential, mc_dropout,
deep_ensemble, quantile_regression, ngboost, platt, isotonic,
CalibratedClassifierCV.

**Applicability domain:** leverage, williams, mahalanobis, dmodx.

**Models:** RandomForest, SVR, SVC, XGB, catboost, PLSRegression, GaussianProcess,
Ridge, Lasso, ElasticNet, MLPRegressor, DecisionTree, DummyRegressor.

**Imbalance:** class_weight, scale_pos_weight, is_unbalance, smote, imblearn,
oversampling, undersampling.

**Interpretability:** shap (all 100 textual hits are `shape`), shapley,
attribution, saliency, integrated_gradients, SimilarityMaps, counterfactual,
feature_importances_.

**Feature selection:** feature_selection, SelectKBest, VarianceThreshold, RFE,
boruta.

**Benchmark data:** moleculenet, deepchem (the 20 `deepchemography` hits are a
different, author-owned package), tdc, pytdc, polaris, openadmet, moleculeace,
dockstring, ocelot, qm9, esol, freesolv, bace, bbbp, tox21, sider, clintox, muv.

**Agentic:** planner class, task_graph, decompose_task, replan, reflexion,
self_critique, critic_agent, trajectory, rubric, scorecard, golden_set,
ground_truth, token_budget, cost_limit, spend_limit, require_approval,
await_approval, lancedb/vectordb/embedder/RAG usage.

**Regulatory:** OECD, QMRF, QPRF, REACH.

**Dead or undeclared:** `admet-ai` (declared, zero imports), `mlflow` (used, never
declared), `lancedb` / `tantivy` / `sentence-transformers` (declared, zero usage),
`OutputLayout.manifest_rel_path` (no callers), `tracking/decorators.py` (applied
to nothing).
