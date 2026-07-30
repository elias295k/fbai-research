# Historical pseudo-xG experiment

## Status and boundary

Pseudo-xG is an isolated research candidate. It does not modify
`FEATURE_COLUMNS`, `LR52Config`, `fit_lr52`, the default evaluator, Match2Vec,
or the market benchmark. The stable path remains exactly 52 pre-match
features into LR52.

The authority is, in descending order:

1. `FBAI_NEW/src/fbai_new/xg.py` and
   `FBAI_NEW/scripts/run_xg_eval.py`;
2. `FBAI_NEW/tests/test_xg.py`;
3. `FBAI_NEW/results/xg_eval.json`;
4. `FBAI_NEW/reports/XG_RESULTS.md`.

This phase ports only the primary LR experiment. The authority's secondary
Match2Vec diagnostic and all market simulation are outside Phase 5B1.
Understat and every other external xG source are also outside scope.

## Estimator

Each outer fold fits a fresh scikit-learn `PoissonRegressor` on completed
team-match sides from its training partition:

```text
target:      goals
predictors:  shots_on_target, shots_off_target

shots_off_target = total_shots - shots_on_target

E[goals | predictors] =
    exp(intercept
        + beta_target * shots_on_target
        + beta_off * shots_off_target)
```

The effective source configuration is `alpha=1e-6`, `max_iter=1000`,
`fit_intercept=true`, `solver=lbfgs`, and `tol=1e-4`. Home and away sides
become separate training observations. Missing or negative predictors are
excluded from fitting; invalid completed-match predictors produce missing
pseudo-xG. Like the authority, the estimator is pooled across all divisions
inside its eligible training window; only the rolling team history is
division-isolated.

## Experimental feature contract

History is keyed by `(Division, team)`, restricted to the target
`SeasonStartYear`, and cut with a strict date boundary. Partial early-season
windows are retained. For each target side, windows 5 and 10 produce:

- pseudo-xG for average;
- pseudo-xG against average;
- goals minus pseudo-xG-for average.

The explicit ordered tuple is:

```text
PxgForAvg5_H_pxg_pre
PxgAgainstAvg5_H_pxg_pre
GminusPxgForAvg5_H_pxg_pre
PxgForAvg10_H_pxg_pre
PxgAgainstAvg10_H_pxg_pre
GminusPxgForAvg10_H_pxg_pre
PxgForAvg5_A_pxg_pre
PxgAgainstAvg5_A_pxg_pre
GminusPxgForAvg5_A_pxg_pre
PxgForAvg10_A_pxg_pre
PxgAgainstAvg10_A_pxg_pre
GminusPxgForAvg10_A_pxg_pre
```

These 12 names are approved only inside the pseudo-xG research namespace.
They are never added to the stable 52-name tuple.

## Leakage-safe interpretation

The authority fitted one outer-training Poisson model and used it to
transform both training and test history. That is outer-fold-local, but a
later training match could alter an earlier transformed training row through
the learned coefficients. Phase 5B1 removes that ambiguity:

- candidate-training features use season-boundary walk-forward refits;
- each refit sees only matches strictly before the first target date in that
  season;
- the fold test estimator sees the complete outer training partition and no
  test row;
- rolling histories use only completed matches strictly before the target
  date;
- all matches on a date are excluded as one batch;
- prior completed test dates may inform later test-date rolling history, but
  never estimator parameters.

This safety adaptation preserves the estimator formula, hyperparameters,
feature definitions, downstream candidate, folds, metrics, and gate. It is
recorded explicitly because its aggregate results can differ from the
authority.

## Candidate and gate

The primary candidate is Logistic Regression on the exact 52 LR52 features
plus the 12 pseudo-xG features:

```text
64 features
    -> SimpleImputer(strategy="median")
    -> StandardScaler()
    -> LogisticRegression(
           solver="lbfgs",
           effective penalty="l2",
           C=1.0,
           max_iter=2000,
           fit_intercept=true,
           class_weight=None,
           random_state=42,
           tol=1e-4)
```

LR52 is refitted independently on the same outer training rows and evaluated
on identical test keys. Probabilities are explicitly reordered to `H, D, A`.
The development aggregate covers test seasons 2022–2024 and is weighted by
test sample count. The 2025 fold is historical-final.

```text
candidate_improvement_log_loss =
    lr52_log_loss - candidate_log_loss

pass only if:
    development improvement >= 0.005
    and at least 2 of 3 development folds improve
```

Positive improvement favors pseudo-xG. A failed gate has the source
disposition `XG_SIGNAL_REJECTED_FOR_NOW`.

See the immutable aggregate record in
`research/pseudo_xg/result.json` and the narrative result in
`research/pseudo_xg/RESULTS.md`.
