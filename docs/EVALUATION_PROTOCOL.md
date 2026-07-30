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
