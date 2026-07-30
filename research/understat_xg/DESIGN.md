# Understat xG historical reproduction design

## Research question

Can completed-match external expected-goals history add stable, leakage-safe
predictive information beyond LR52?

## Frozen primary comparison

- evaluation subset: every canonical row in a division represented by the
  external artifact;
- folds: expanding 2022, 2023, 2024 development and 2025 historical-final;
- baseline: a freshly fitted LR52 on the subset;
- candidate: the exact LR52 52 plus the source-defined 16 xG features;
- preprocessing: training-only median imputation and standard scaling;
- classifier: source-default L2 `lbfgs` Logistic Regression, `C=1`,
  `max_iter=2000`, `random_state=42`;
- metrics: H/D/A Log Loss, multiclass Brier, and top-label ECE;
- aggregation: test-sample-count weighted.

Both models use identical fold test keys. The Match2Vec secondary diagnostic
from the private research cycle is outside this port. No market evaluation is
part of this experiment.

## External contract

The external table stays separate from canonical data. It is normalized and
joined one-to-one by date, division, normalized home name, and normalized away
name. Only executable-source aliases are permitted. Duplicate normalized keys,
invalid values, unknown divisions, or failed aggregate quality gates stop
evaluation.

The internal left join retains unmatched canonical rows as missing xG because
the authority counts those matches in its trailing-ten slot policy before it
filters to present xG values.

## Leakage controls

- completed-match xG is accessible only to later dates;
- target and same-date xG is excluded with a strict date boundary;
- state is isolated by division and team;
- history resets by `SeasonStartYear`;
- input ordering is canonicalized before feature construction;
- candidate preprocessing is newly fitted inside every outer fold;
- the historical-final fold does not alter the development gate.

Synthetic tests alter future, current, and same-date xG and shuffle both
inputs to prove keyed invariance.

## Gate

Positive improvement means the candidate is better:

```text
LR52 development Log Loss - candidate development Log Loss
```

Passing requires improvement of at least `0.005` and positive improvement in
at least two of the three development folds. Otherwise the disposition is
`XG_SIGNAL_REJECTED_FOR_NOW`.
