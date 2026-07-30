# LR52 modeling

## Why LR52

LR52 is the public package's single probabilistic baseline. It is deliberately
small, deterministic, and auditable: the model consumes the exact 52
leakage-safe Phase 2B features and adds no feature search, tuning, calibration,
odds, or market inputs. More complex models are deferred until this baseline
and its evaluation boundary are stable.

## Fixed input and target

The model input is exactly `feature_columns()` in its declared order. Columns
are never discovered by dtype or suffix. Immediately before every fit, the
tuple passes the semantic leakage guard, which rejects metadata, labels,
same-match statistics, odds, and unapproved aliases.

The target is `target_1x2` with all three values H, D, and A required in every
training fold. Scikit-learn internally orders these classes as A, D, H. Public
predictions are therefore explicitly remapped into:

```text
probability_home, probability_draw, probability_away
H,                D,                A
```

Outputs are checked for finite values, the unit interval, and row sums of one.

## Train-only preprocessing

Each chronological fold constructs a fresh scikit-learn pipeline:

```text
52 numeric features
    -> SimpleImputer(strategy="median")
    -> StandardScaler()
    -> LogisticRegression
```

The imputer medians and scaler mean/scale are fitted only on that fold's
training rows. Test rows are transform-only. Legitimate early-history NaNs are
accepted; infinities fail before fit. There is no full-table imputation,
backfill, target-dependent filling, or reusable preprocessing fitted across
folds.

## Exact Logistic Regression configuration

```text
solver             lbfgs
effective penalty  L2
C                  1.0
max_iter           2000
fit_intercept      true
class_weight       none
random_state       42
tol                1e-4
```

Modern scikit-learn deprecates passing `penalty="l2"` explicitly while
retaining the same L2 behavior when the parameter is omitted. The public
configuration records the effective L2 penalty and omits the deprecated
constructor argument. No `multi_class` argument is used; `lbfgs` performs
multinomial optimization for this three-class problem. A convergence warning
or exhausted iteration budget raises an explicit failure.

## Chronological protocol

The runner uses `fbai.core.splits.expanding_folds`; it does not implement a
second split system.

```text
test 2022: train SeasonStartYear <= 2021  (development)
test 2023: train SeasonStartYear <= 2022  (development)
test 2024: train SeasonStartYear <= 2023  (development)
test 2025: train SeasonStartYear <= 2024  (historically frozen final)
```

Every training date must be strictly earlier than every test date. The 2025
fold is the final fold from the historical research protocol, not currently
unseen future data. Configuration is fixed; this phase performs no tuning.

Each fold reports multiclass log loss, multiclass Brier score, top-label
expected calibration error, and sample count. Aggregates use test-row counts
as weights. Development, historical-final, and all-fold diagnostic views stay
separate.

## Non-market references

The uniform reference emits exactly one-third for H, D, and A. The
training-prior reference fits H/D/A frequencies separately from each fold's
training labels. Both use the same probability validator and metrics as LR52.
They are statistical references, not market benchmarks.

## Historical reproduction

The local reproduction reads the authorized historical canonical archive
without modifying or copying it, rebuilds all 52 features with the public
Phase 2B implementation, and runs the public Phase 3 evaluator. Only fold and
aggregate metadata is committed under `research/lr52_baseline/`.

Raw historical rows, team-level output, match predictions, and model binaries
are not included. The historical archive is not available to CI, which uses
deterministic synthetic data. No market-comparison claim is made in Phase 3.
