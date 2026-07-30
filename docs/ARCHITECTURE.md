# Architecture

The Phase 0–1 package has three focused areas.

```text
src/fbai/
├── core/
│   ├── leakage.py   semantic column roles and table contracts
│   ├── splits.py    chronological folds, inner splits, date batches
│   └── metrics.py   validated H/D/A probabilistic metrics
└── testing/
    └── synthetic.py deterministic invented fixtures
```

## Leakage boundary

`core.leakage` distinguishes model features from labels, metadata,
same-match values, odds, and unknown columns. A valid predictive feature:

1. is explicitly included in the caller's approved feature set;
2. ends in `_pre`;
3. does not reduce to a forbidden semantic base such as `HomeShots`,
   `FTHG`, `FTR`, `MatchDate`, or `AvgH`.

Feature-table checks separately enforce natural-key uniqueness, non-null keys,
target domains, and a closed column schema.

## Chronology boundary

`core.splits` normalizes dates and sorts with mergesort on:

```text
MatchDate, Division, HomeTeam, AwayTeam
```

Expanding folds enforce `max(train.MatchDate) < min(test.MatchDate)`.
Chronological inner validation never separates rows sharing a date.
`chronological_date_batches` exposes the batching contract to future
stateful feature or model code: all results from a date become available only
after the complete date batch has been processed.

## Metric boundary

`core.metrics` fixes probability columns to `(H, D, A)`. It rejects malformed,
non-finite, negative, out-of-range, or non-normalized probability matrices
before computing log loss, Brier score, calibration error, or weighted fold
summaries.
