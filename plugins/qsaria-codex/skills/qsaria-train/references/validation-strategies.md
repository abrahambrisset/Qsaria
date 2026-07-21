# Validation strategies

## Default and explicit overrides

- For an unqualified single-model request, use `validation={"kind":"standard_qsar"}`.
- `standard_qsar` is the fixed random 80/10/10 train/validation/test split implemented by the toolkit.
- Use another typed `validation` object only for an explicit split family, split ratio, holdout, repeated holdout, cross-validation, fold count, repetition count, isolated external test, or full-dataset final training request.
- Never translate cross-validation into repeated holdout.
- A full-train request uses `{"kind":"full_train"}`. It is trained but not evaluated and has no internal test metrics.

## Canonical vocabulary

- `kind`: `standard_qsar`, `holdout`, `repeated_holdout`, `cross_validation`, `full_train`
- `split_family`: `random`, `scaffold`, `cluster` for holdouts; cross-validation currently accepts only `random`
- `split_sizes`: `[train, test]` or `[train, validation, test]`
- `n_folds`, `n_repeats`, `outer_test_size`, `final_refit`, `selection_metric`

Use `outer_test_size` for an isolated cross-validation test set. Never invent aliases. Never create split columns outside the training facade.

## Evidence

Keep every split family distinct. Preserve per-split and aggregated metrics, primary deployment split, hardest split, robustness gaps, random stability, and the canonical gates: `Dataset Gate`, `Hardest Split Gate`, `Robustness Gap Gate`, and `Random Stability Gate`.

Report mean and standard deviation when returned for repeated protocols. Never compute missing metrics manually.
