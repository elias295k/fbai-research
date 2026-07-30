# Evaluation protocol

## Fold roles

- Development folds: test seasons 2022, 2023, and 2024.
- Historical final fold: test season 2025.

The 2025 fold is **the historically frozen final evaluation fold used by the
original research**.

Every expanding fold trains on `SeasonStartYear <= test_year - 1` and tests on
`SeasonStartYear == test_year`. The implementation then checks actual
`MatchDate` values and raises unless every training date is strictly earlier
than every test date.

## Stable order and same-date matches

Rows are deterministically ordered by:

```text
MatchDate, Division, HomeTeam, AwayTeam
```

This order is for reproducibility, not an information-timing claim. Matches on
the same calendar date form one indivisible batch. No result from one match may
be added to history before all feature/model inputs for that date have been
created.

The Phase 2B Elo, context, form, and rolling builders all implement this rule
directly: they calculate a complete date from pre-date state and update history
only after the date batch is complete.

Chronological inner train/validation splitting chooses a date boundary and
moves the complete boundary date into validation. Therefore:

```text
max(inner_train.MatchDate) < min(inner_validation.MatchDate)
```

## Metrics

The class/probability-column order is always `H, D, A`.

- Primary: multiclass log loss.
- Secondary: multiclass Brier score and expected calibration error.
- Multiple folds: sample-count-weighted means.

Invalid probability matrices fail rather than being silently renormalized.

## LR52 no-tuning baseline

Model validation consumes only the immutable 52-name tuple exported by the
feature schema. Labels, metadata, same-match statistics, odds, and arbitrary
`_pre` aliases are rejected by the semantic fit-time guard.

LR52 has one fixed source-verified configuration. This phase performs no
hyperparameter search, calibration, feature selection, or final-fold-driven
configuration change.

Every fold constructs and fits a fresh median imputer, standard scaler, and
Logistic Regression pipeline from training rows only. Test rows are
transform-only. Phase 2B early-season missing values remain untouched until
the fold-local imputer.

Scikit-learn's internal class order is never exposed positionally.
Probabilities are remapped and validated in the public order H, D, A.

Fold records remain separated by role:

- development aggregate: 2022–2024 test folds;
- historical-final result: 2025 test fold;
- all-fold sample-count-weighted diagnostic reproduction.

The all-fold view is explicitly diagnostic because the historical final fold
has previously been examined.

## Closing-market comparison

The closing market is an external, untrained, near-kickoff benchmark. Its
timing is later than the historical information represented by LR52, so the
comparison is not described as equal-timing or days-ahead forecasting.

Closing rows attach only through an exact one-to-one match on:

```text
MatchDate, Division, HomeTeam, AwayTeam
```

Coverage reports candidate, supplied, valid, duplicate, incomplete, invalid,
unmatched-both-directions, and final aligned rows, plus percentages overall,
by division, by `SeasonStartYear`, and by fold. There is no fuzzy name match,
row-position join, duplicate selection, forward fill, or odds repair.

Within each fold, LR52 still fits on the complete valid training history using
exactly 52 non-market fields. Both LR52 and market are then evaluated against
the same targets on the market-covered test-key intersection. Uniform and
training-prior references use that same intersection. The original full-test
LR52 metric remains a distinct context value.

Aligned folds are sample-count weighted into development, historical-final,
and all-historical diagnostic views. The sign convention is:

```text
market_advantage_log_loss =
    lr52_aligned_log_loss - market_log_loss
```

Positive means the market has lower Log Loss; negative means LR52 has lower
Log Loss; zero is a tie.

## Match2Vec candidate protocol

Match2Vec uses the same four expanding folds and public metric functions. A
fresh vocabulary, inner-fit imputer/scaler, representation network, and
candidate head are created in each fold. Team and league tokens come only from
the outer training window. A test-only team maps to its training league's
explicit unknown team token; a wholly unseen division fails.

Every training and test sequence contains at most ten matches for that team in
the same division, ordered chronologically and strictly earlier than the
target date. Matches on the target date are excluded together through a date
boundary, never through row order. Network parameters remain fixed throughout
test prediction; earlier test results may appear in a later test date's
historical descriptors because they were available by then.

The predefined development gate is:

```text
candidate_improvement_log_loss =
    lr52_log_loss - match2vec_log_loss

required:
    candidate_improvement_log_loss >= 0.005
    and at least 2 of 3 development folds improve
```

Positive values favor Match2Vec. The 2025 historical-final result is reported
but cannot tune the candidate or rescue a failed development gate. See
[Match2Vec](MATCH2VEC.md).

## Pseudo-xG candidate protocol

Pseudo-xG uses the same four outer folds, public metrics, identical-key LR52
comparison, sample-count weighting, and development/final separation. The
Poisson estimator target is goals and its predictors are shots on target and
shots off target. Candidate-training features use walk-forward estimator
refits at season boundaries; test features use only the current outer
training estimator. Every rolling window is restricted to the target season
and dates strictly before the complete target-date batch.

The candidate is median-imputed, standardized Logistic Regression on 52 plus
12 explicitly approved research features. Its gate is:

```text
candidate_improvement_log_loss =
    lr52_log_loss - pseudo_xg_candidate_log_loss

required:
    candidate_improvement_log_loss >= 0.005
    and at least 2 of 3 development folds improve
```

The historical-final result cannot select configuration or rescue a failed
development gate. See [Pseudo-xG](PSEUDO_XG.md).

## Understat xG candidate protocol

The external-xG experiment uses the same four outer folds and public metrics,
but evaluates only divisions present in the validated external artifact.
LR52 and the LR68 candidate use identical training and test keys. External
rows attach to canonical history through a one-to-one exact normalized
natural-key join with explicit source aliases.

For each target side, windows 5 and 10 summarize xG for, xG against, xG
difference, and goals minus xG from strictly earlier same-season matches.
Same-date rows are excluded as a batch. Median imputation, scaling, and the
candidate Logistic Regression fit only on each outer training partition.

```text
candidate_improvement_log_loss =
    lr52_log_loss - understat_xg_candidate_log_loss

required:
    candidate_improvement_log_loss >= 0.005
    and at least 2 of 3 development folds improve
```

The 2025 historical-final result cannot tune or rescue the candidate. See
[Understat xG](UNDERSTAT_XG.md).
