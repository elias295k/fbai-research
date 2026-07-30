# LR52 baseline design

## Research question

Can the verified 52-feature pre-match table reproduce the historical
multinomial Logistic Regression baseline under a strict expanding
chronological protocol?

## Preregistered design

- Input: exactly the ordered 52 names returned by `feature_columns()`.
- Target: `target_1x2`.
- Class/probability order: H, D, A.
- Preprocessing: training-fold median imputation followed by training-fold
  standard scaling.
- Model: `lbfgs` multinomial Logistic Regression, effective L2 penalty,
  `C=1.0`, `max_iter=2000`, intercept enabled, no class weighting,
  `random_state=42`, `tol=1e-4`.
- Development folds: test 2022, 2023, and 2024.
- Historically frozen final fold: test 2025.
- Aggregation: sample-count-weighted.
- Primary metric: multiclass log loss.
- Secondary metrics: multiclass Brier score and top-label ECE.

The success criterion is an all-fold weighted log-loss difference no greater
than `1e-8` from the strongest committed LR52 reference, with every available
per-fold log loss also within `1e-8`. Hyperparameters must not be changed to
force agreement.

## Leakage protections

Every fit selects the fixed tuple and invokes the semantic feature guard
immediately before pipeline fitting. Each fold builds a new imputer, scaler,
and classifier from training rows only. Test features and labels cannot change
training preprocessing or fitting. Same-date information batches are already
enforced by Phase 2B, and fold boundaries require strict date separation.

## Scope boundary

Uniform and training-prior probabilities provide non-market references. Market
comparison is deferred to a later phase; no odds enter this experiment.
Neither a model binary nor match-level predictions are committed.
