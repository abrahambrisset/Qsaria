# Terminal training failures

A training tool error before a successful result is terminal for the current mission unless the user explicitly authorizes another attempt.

Do not recover by:

- changing backend, representation, validation protocol, split, folds, seed, tuning, compute profile, or training facade;
- omitting or inferring failed-call arguments;
- probing guessed model, summary, prediction, or split paths;
- treating a JSON summary as a CSV;
- starting a benchmark after single-model failure;
- switching between Chemprop, LightGBM, or TabICL.

Return completed artifacts, the exact error, preserved user constraints, and a recommended user decision. Use `retryable_error` only for an unambiguous transient technical problem whose retry would keep the same scientific call. Only a handoff explicitly carrying `status=retryable_error` is eligible for the coordinator's single automatic retry; otherwise use `terminal_failure`.
