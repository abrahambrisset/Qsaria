# Qsaria V1 for traceable molecular property modelling

## Abstract

Qsaria V1 provides a workflow for curating molecular data, training and evaluating QSAR models, and recording evidence for scientific reports. We describe its architecture and evaluate its LightGBM backend against Random Forest and naïve references on two public partitions from the Therapeutics Data Commons ADMET benchmark. For AqSolDB, the mean absolute error across three seeds was 1.279 log mol/L for LightGBM and 1.378 for Random Forest on 1,993 eligible test rows. On BBB_Martins, the mean area under the ROC curve was 0.865 for LightGBM and 0.856 for Random Forest on 406 test rows. The paired interval for the BBB_Martins difference at the prespecified reference seed included zero. Random Forest had lower RMSE than LightGBM on both activity-cliff subsets in a two-endpoint MoleculeACE follow-up. Chemprop reduced AqSolDB MAE to 0.923 in a full-training comparison. TabICL reduced AqSolDB MAE from 1.547 to 1.480 against a LightGBM fit using the same 800 training rows and fingerprints. The evaluation records standardized molecular identity overlap, data exclusions, and the provenance of the data, configuration, and predictions.

## Software and scope

Qsaria V1 originated in ChemSpace Copilot. It provides the shared toolkits for dataset curation, model training, inference, model registration, and reporting. The native QSAR team runs through Agno and Chainlit. The Qsaria MCP server exposes these operations under typed requests and records an experiment's artifacts and handoffs. A client can retrieve the evidence used to prepare the final report. Model selection, validation configuration, representation, and optimization are explicit request fields. The evaluated V1 revision requires an explicit request for hyperparameter optimization. [ChemSpace Copilot](https://github.com/Laboratoire-de-Chemoinformatique/chemspacecopilot) · [Qsaria](https://github.com/abrahambrisset/Qsaria)

The predictive study estimates performance for a fixed LightGBM configuration and comparable references. The workflow study checks whether the tool interfaces preserve the requested method and expose traceable evidence.

## Methods

We evaluated aqueous solubility on AqSolDB and binary blood–brain barrier permeability on BBB_Martins using the published development and test memberships in TDC ADMET archive `4426004`. Mean absolute error in log mol/L and AUROC were the respective primary metrics. We recorded the archive hash and dataset row numbers. The distributions and counts in this study refer to this exact archive. [TDC ADMET benchmark](https://tdcommons.ai/benchmark/admet_group/overview/) · [AqSolDB](https://doi.org/10.1038/s41597-019-0151-1) · [BBB_Martins](https://doi.org/10.1021/ci300124c)

Development validation grouped equivalent structures using Qsaria's standardized identity. For classification we used the first fold of five-fold stratified group cross-validation. For regression we used a group holdout targeting 20% of the groups. The final study preserved the archived test partition and fitted each model on the usable development rows. We report the full scored test cohort and a prespecified sensitivity analysis excluding test rows whose standardized identity occurred in the fit partition. Failed structure conversions were logged in the run. The script read final test structures for featurization and opened the final test labels for scoring after the model predictions for that dataset had been generated.

Both tree models used identical 2,048-bit Morgan fingerprints with radius two, 80 trees, one thread and fixed model settings. The LightGBM fit and prediction used the Qsaria V1 backend. Random Forest used scikit-learn's classifier or regressor. The naïve reference predicted the fitted mean for regression and used the fitted class prior for AUROC. The final analysis used model seeds 42, 1701, and 2026. The seed-to-seed standard deviation describes variability across these three fits. For seed 42 we computed a paired percentile interval from 1,000 bootstrap draws of standardized identities, retaining all rows for each sampled identity and applying the same draws to both models. The interval is conditional on the fitted models and this test set. The Qsaria strict bounding box was fitted only on development features and evaluated as an applicability diagnostic.

## Predictive results

| Dataset and cohort | Rows | Metric | Qsaria LightGBM | Random Forest | Naïve |
| --- | ---: | --- | ---: | ---: | ---: |
| AqSolDB, all scored | 1,993 | MAE | 1.279 ± 0.003 | 1.378 ± 0.005 | 1.853 |
| AqSolDB, unseen identities | 1,966 | MAE | 1.280 ± 0.003 | 1.387 ± 0.005 | 1.848 |
| BBB_Martins, all scored | 406 | AUROC | 0.865 ± 0.004 | 0.856 ± 0.015 | 0.500 |
| BBB_Martins, unseen identities | 403 | AUROC | 0.860 ± 0.004 | 0.850 ± 0.016 | 0.500 |

Table 1. Mean and sample standard deviation across three model seeds. The naïve predictions are invariant across seeds. Four of 1,997 AqSolDB test rows could not be featurized. There were 27 scored AqSolDB rows and three BBB_Martins rows whose standardized identities occurred in the respective fit partitions.

At seed 42, the paired difference between LightGBM and Random Forest on all scored AqSolDB rows was −0.099 MAE (95% interval −0.123 to −0.073). The difference on unseen identities was −0.107 (−0.132 to −0.081). On BBB_Martins, the paired difference at seed 42 was −0.012 AUROC (−0.041 to 0.016) on all scored rows. The direction of the difference at this seed and the uncertainty across molecules are relevant when interpreting the three-seed means.

![Paired differences in MAE and AUROC for the full scored cohorts and unseen standardized identities](results/qsaria-v1/model_differences.png)

Figure 1. The point is the paired difference for seed 42 and the line is its 95% identity-bootstrap interval. Each panel uses its own metric and scale. Negative differences favour LightGBM for MAE. Positive differences favour LightGBM for AUROC.

The bounding box classified all 1,993 scored AqSolDB rows as in-domain. On BBB_Martins, 397 rows were in-domain and nine were out-of-domain. The subgroup AUROCs at seed 42 were 0.864 and 0.850, respectively. The counts limit the evidence about performance outside the fitted box.

### Matched classical AutoML comparison

A follow-up used the same archived partitions, grouped development validation and molecular representation. Both arms evaluated four candidates on one core with 80 trees per fit. The Qsaria arm searched LightGBM learning rate and leaf count. The classical arm selected between Random Forest and ExtraTrees, each at two feature-subsampling settings. ExtraTrees with 10% feature subsampling won on both development endpoints. The checksummed selections were recorded before the follow-up opened either test partition. Both winners were refitted at seeds 42, 1701 and 2026. This matches tree, candidate and core budgets. Fit and prediction times differed and are included in the run reports. [Protocol](qsaria-v1-automl.toml) · [Development run](results/qsaria-v1/automl-development.json)

| Dataset and cohort | Rows | Metric | Qsaria LightGBM | Selected classical AutoML | Paired difference (95% interval) |
| --- | ---: | --- | ---: | ---: | ---: |
| AqSolDB, all scored | 1,993 | MAE | 1.215 | 1.306 | −0.090 [−0.120, −0.063] |
| AqSolDB, unseen identities | 1,966 | MAE | 1.219 | 1.313 | −0.095 [−0.126, −0.067] |
| BBB_Martins, all scored | 406 | AUROC | 0.8668 | 0.8670 | −0.00025 [−0.03537, 0.03575] |
| BBB_Martins, unseen identities | 403 | AUROC | 0.8621 | 0.8618 | 0.00022 [−0.03542, 0.03609] |

Table 2. Test results at seed 42. The difference is Qsaria LightGBM minus the selected classical model, with the same paired identity bootstrap as in the fixed-model comparison. The AqSolDB intervals favoured Qsaria in all three seeds, and all BBB_Martins intervals included zero. The molecular overlap audit found 27 exact matches and one nonidentical test row with count-Morgan Tanimoto similarity of at least 0.95 on AqSolDB, and three exact matches and zero such neighbours on BBB_Martins. [Official test run](results/qsaria-v1/automl-evaluation.json)

### Activity cliffs on MoleculeACE

We evaluated two MoleculeACE regression datasets, ABL1 binding affinity (CHEMBL1862 Ki) and FXR activation (CHEMBL2047 EC50). We retained the official train and test assignments and activity-cliff membership in the corrected MoleculeACE 3.0.0 release. The two tree models used the fixed binary Morgan settings, 80 trees, one CPU thread and the same three model seeds as the TDC comparison. The naïve reference predicted the mean of the training targets. We measured RMSE on all scored test molecules and RMSE on the test molecules that MoleculeACE marks as activity-cliff members. Both use the source log-scale targets. [MoleculeACE study](https://doi.org/10.1021/acs.jcim.2c01073) · [Correction](https://doi.org/10.1021/acs.jcim.3c01576) · [Corrected data release](https://github.com/molML/MoleculeACE/releases/tag/v3.0.0)

| Dataset and cohort | Test rows | Qsaria LightGBM | Random Forest | Naïve |
| --- | ---: | ---: | ---: | ---: |
| ABL1, all | 161 | 0.851 | 0.768 | 1.479 |
| ABL1, activity-cliff subset | 67 | 0.811 | 0.678 | 1.439 |
| FXR, all | 128 | 0.694 | 0.621 | 0.983 |
| FXR, activity-cliff subset | 50 | 0.752 | 0.674 | 0.910 |

Table 3. Mean RMSE across three model seeds. All official test rows could be scored. Random Forest had the lowest mean error for both endpoints on the full tests and on their cliff subsets. No standardized identity crossed the training and test partitions. Four ABL1 test rows and one FXR test row had a nonidentical training neighbour with count-Morgan Tanimoto similarity of at least 0.95. [Protocol and replay](qsaria-v1-moleculeace.md) · [Results and identity audit](results/qsaria-v1/moleculeace.json)

### Comparison across model families

We extended the benchmark to Chemprop v2 and TabICL v2, keeping the four official test partitions and the original scoring cohorts. Chemprop used standardized molecular graphs and trained a directed message-passing neural network on the full usable training partition. We fixed training at 18 epochs, depth three, a hidden dimension of 64 and no hyperparameter search. To bound TabICL's CPU workload, we ran a second comparison in which TabICL and LightGBM both used binary Morgan fingerprints of radius two and 128 bits with a fixed sample of at most 800 official training rows. Sampling was stratified for BBB_Martins. TabICL used one ensemble member with the official pretrained weights, and the matched LightGBM fit used 80 trees. Both comparisons used the same three seeds as the original benchmark. [Chemprop v2](https://doi.org/10.1021/acs.jcim.5c02332) · [TabICL v2](https://arxiv.org/abs/2602.11139) · [Protocol and replay](qsaria-v1-backends.md)

| Dataset and cohort | Test rows | Metric | Chemprop | Qsaria LightGBM | Random Forest |
| --- | ---: | --- | ---: | ---: | ---: |
| AqSolDB, all | 1,993 | MAE | 0.923 | 1.279 | 1.378 |
| BBB_Martins, all | 406 | AUROC | 0.681 | 0.865 | 0.856 |
| ABL1, all | 161 | RMSE | 1.236 | 0.851 | 0.768 |
| ABL1, activity-cliff subset | 67 | RMSE | 1.083 | 0.811 | 0.678 |
| FXR, all | 128 | RMSE | 0.905 | 0.694 | 0.621 |
| FXR, activity-cliff subset | 50 | RMSE | 0.850 | 0.752 | 0.674 |

Table 4. Mean test scores across three seeds with the full eligible training partitions. Chemprop had the lowest MAE on AqSolDB in this comparison. The two tree models achieved higher AUROC on BBB_Martins and lower RMSE on both MoleculeACE datasets. [Chemprop results](results/qsaria-v1/chemprop.json) · [Original LightGBM and Random Forest results](README.md)

| Dataset and cohort | Training rows | Test rows | Metric | TabICL | Matched LightGBM |
| --- | ---: | ---: | --- | ---: | ---: |
| AqSolDB, all | 800 | 1,993 | MAE | 1.480 | 1.547 |
| BBB_Martins, all | 800 | 406 | AUROC | 0.8273 | 0.8262 |
| ABL1, all | 633 | 161 | RMSE | 0.782 | 0.831 |
| ABL1, activity-cliff subset | 633 | 67 | RMSE | 0.794 | 0.796 |
| FXR, all | 503 | 128 | RMSE | 0.634 | 0.712 |
| FXR, activity-cliff subset | 503 | 50 | RMSE | 0.722 | 0.735 |

Table 5. Mean test scores across three seeds with the shared training context and 128-bit fingerprints. TabICL reduced mean error on the three regression tests. The BBB_Martins AUROCs differed by 0.0011. The range of TabICL scores across the three fixed-context runs was zero at the precision reported. [TabICL and matched LightGBM results](results/qsaria-v1/tabicl.json)

## Workflow verification

An installation check in an isolated Codex configuration verified plugin registration. An MCP stdio client initialized the Qsaria profile and listed 53 tools. A local functional check created an experiment and trained the V1 LightGBM backend with 40 synthetic structures, 12 trees and the optimization field omitted. The run produced prediction artifacts. This fixture uses a computed cLogP target and supplies functional evidence about the tool path.

We tested two equivalent requests for curation and lightweight LightGBM training in fresh GPT-6.1 Terra chats at high reasoning effort. Both accessed the actual Qsaria MCP server through a repository-local stdio bridge. Both preserved the requested CPU, `local_light`, binary Morgan, 12-estimator budget and disabled optimization and training outlier analysis. Each kept 40 rows after curation and produced four test predictions. The reported MAEs, 0.4340 and 1.0935, matched recomputation from the respective CSVs. Different effective seeds preclude interpreting the MAE difference as an effect of the wording. The coordinator initially attempted training before recording the curation handoff in both chats. The server enforced the handoff and both completed an authorized fit. One chat created a second experiment during recovery. The trials expose first-attempt sequencing and session-recovery issues. [Terra evidence](results/qsaria-v1/agentic.json)

A separate read-only test confirmed direct MCP use in Codex Desktop. With the local plugin installed and the repository root open as a project, a fresh GPT-6.1 Terra chat at high reasoning effort called `qsaria_bootstrap` and `qsaria_list_experiments` through the native integration. Bootstrap returned `ok` with the `qsaria` profile and server-side LLM use disabled. The experiment query completed and returned an empty list. [Native Codex test](results/qsaria-v1/agentic.json)

We also exercised the team instantiated by Chainlit against a scripted model transport. The real team delegated curation, training and reporting for the same fixture. It retained 40 rows and returned a test MAE of 0.884744, verified against four predictions. This test covers the Agno team and its tools with deterministic model responses. [Chainlit evidence](results/qsaria-v1/chainlit.json)

The CI reference checks run the Chainlit scenario and two scripted coordinator workflows against the real MCP server over stdio. The nominal path goes from curation to a session-scoped model and a structured report. The recovery path rejects training before its required handoff and rejects an unknown artifact ID, then completes the same experiment after a valid handoff. Server manifests establish the order of completed operations. The two runs link the recorded test split, registered model and prediction artifact, and the test recomputes MAE and RMSE from the four saved predictions. [MCP reference scenarios](../tests/integration/test_qsaria_v1_mcp_workflow.py) · [CI configuration](../.github/workflows/ci.yml)

## Limitations

The predictive evaluation covers two TDC endpoints, two MoleculeACE endpoints and three model families available in Qsaria. Its conclusions apply to these representations, comparator configurations, and partitions. The AutoML follow-up reuses test partitions already used in the fixed-model comparison. It evaluates finite searches over two tree-based model families and the Qsaria LightGBM settings. Restricting evaluation to unseen standardized identities controls exact overlap under this definition. It allows closely related structures to occur on both sides. The applicability-domain result concerns a strict bounding box on binary fingerprints. It supplies a coverage diagnostic and the observed subgroup scores. Using three seeds on fixed test memberships leaves the effect of alternative dataset partitions unmeasured. The MoleculeACE follow-up measures regression error by the supplied cliff annotations. It leaves the pair-discovery performance of Qsaria's SALI tool unmeasured. The graph and tabular extensions use fixed settings. TabICL's context and feature caps limit inference about its performance with more training rows, wider fingerprints or larger ensembles.

The workflow trials use one small synthetic fixture. The reference checks exercise software contracts and known trajectories with deterministic inputs. Prompt robustness and open-ended Chainlit interaction with an external model provider remain unmeasured in these checks.

## Code, data and reproduction

The fixed-model TDC results refer to Qsaria V1 revision `1179b63d42eea5be044bdc041a0f12c35983019c`. The AutoML follow-up can be replayed at `89d02af3760d495a280b924f254be0c3e9721239` and records checksums for its inputs and source files. The MoleculeACE manifest records the corrected source revision, checksums, runtime, and model settings. The benchmark configurations, commands, aggregate results and figure are supplied with the manuscript. Replaying the benchmarks produces row-level predictions tied to their source tables. [Reproduction guide](README.md) · [Fixed-model manifest](results/qsaria-v1/manifest.json) · [AutoML manifest and results](results/qsaria-v1/automl-evaluation.json) · [MoleculeACE manifest and results](results/qsaria-v1/moleculeace.json)
