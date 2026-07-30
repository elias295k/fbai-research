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
