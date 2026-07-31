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

## Phase 4 market boundary

Phase 4 does not alter LR52's configuration, preprocessing, targets, or
52-column input tuple. The model fit path has no market opt-in: raw odds,
canonical closing odds, market probabilities, and arbitrary extra columns
remain rejected by the closed selector and semantic guard.

Closing probabilities are constructed and evaluated through a separate
evaluation-only path. LR52 is fitted first on complete training history;
same-match comparison restricts only test metrics to exact
market-covered keys. See [Closing market benchmark](MARKET_BENCHMARK.md) for
the source triplet, timing, transformation, validation, and coverage contract.

## Phase 5A experimental boundary

LR52 remains unchanged and is still the default internal model. Advanced
candidates live under `fbai.research`, use separate input contracts, and must
pass an explicitly recorded gate before their disposition can change.

The first candidate is the source-selected Match2Vec sequence hybrid. It
attention-pools strictly-prior match descriptors into 32-value home and away
states, adds their difference and a four-value league embedding, and supplies
that 100-value learned representation together with the exact 52 LR52 numeric
inputs to a linear three-class head. The representation and head are fitted
jointly inside each fold; this is not a second Logistic Regression.

The candidate failed its development improvement threshold and remains
experimental. Full semantics and results are in [Match2Vec](MATCH2VEC.md).

## Phase 5B1 pseudo-xG boundary

The second research candidate estimates completed team-side goals with a
fold-local Poisson model over shots on target and shots off target. Its 12
within-season, strict-prior rolling aggregates are combined with the exact 52
stable features only inside a separate median-imputed, standardized
Logistic Regression pipeline.

Training-row transforms use season-boundary walk-forward Poisson refits so
current or later training statistics cannot alter an earlier row. Test-row
transforms use one estimator fitted only on the outer training window. The
candidate neither changes nor wraps `fit_lr52`; LR52 is independently
refitted for comparison on identical test keys. Full semantics and results
are in [Pseudo-xG](PSEUDO_XG.md).

## Phase 5B2 external xG boundary

The third candidate consumes a separate, validated completed-match xG table.
It builds 16 strict-prior rolling fields with windows 5 and 10, division/team
isolation, a season reset, and complete target-date exclusion. Those fields
join the exact stable 52 only inside a separate LR68 candidate pipeline.

Its model configuration is the source-defined median imputer, standard
scaler, and L2 `lbfgs` Logistic Regression with `C=1`, `max_iter=2000`, and
`random_state=42`. LR52 is fitted independently on identical
covered-division rows. The experiment failed its development gate and remains
optional research. See [Understat xG](UNDERSTAT_XG.md).

## Phase 5C1 deep-capacity boundary

The capacity audit keeps the input fixed and varies only a bounded neural
classifier. Its two source architectures are `52→64→3` with 0.10 dropout and
`52→128→64→3` with 0.20 dropout. Hidden layers use ReLU and no normalization.
Both use deterministic CPU Adam with the frozen source settings.

Median imputation and standard scaling fit on chronological inner-fit rows.
Complete dates remain indivisible during early stopping. Development folds
select the architecture; 2025 is evaluated only once for that selection.
Neither network wraps or changes LR52, and neither is part of the default
model path. See [Deep capacity](DEEP_CAPACITY.md).

## Phase 5C2 temporal graph-model boundary

The graph audit retains the source's dependency-light team-player bipartite
SVD design. Accumulated player-team edge weights use past minutes plus frozen
starter and bench contributions. The last ten same-season historical matches
produce fixed 8- or 16-dimensional side representations and derived
comparison fields.

Graph-only and LR52-plus-graph use separate median-imputed, standardized,
L2 `lbfgs` Logistic Regression pipelines with `C=1`, `max_iter=2000`, and
seed 42. SVD and classifier preprocessing are rebuilt per fold. Complete
dates update history as batches, and only development evidence selects the
representation. No graph field can enter stable `fit_lr52`; the experiment is
not part of inference or the default model path. See
[Graph model](GRAPH_MODEL.md).

## Phase 5C3 player-availability boundary

The availability candidate joins the unchanged 52 stable features with 63
source-defined fields derived only from completed prior matches and strictly
earlier valuations. It uses a separate median-imputed, standardized L2
`lbfgs` Logistic Regression pipeline with `C=1`, `max_iter=2000`, and seed 42.

No target official lineup, bench, participation, minutes, substitutions, or
unverified injury label may enter the candidate. Histories reset by season and
update only after a complete date batch. Coverage filtering is applied before
both candidate and LR52 fitting, so their test keys are identical. The failed
candidate remains research-only. See [Player availability](PLAYER_AVAILABILITY.md).
