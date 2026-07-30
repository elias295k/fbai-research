# Architecture

The package now has a canonical historical-match layer alongside the Phase 1
evaluation contracts.

```text
src/fbai/
├── data/
│   ├── sources.py   supported public source coordinates
│   ├── acquire.py   credential-free atomic CSV acquisition
│   ├── loader.py    source aliases, dates, and season normalization
│   ├── schema.py    canonical schema and integrity contract
│   ├── canonical.py validated per-division Parquet output
│   ├── audit.py     structured dataset audit verdict
│   └── store.py     in-memory DuckDB access
├── core/
│   ├── leakage.py   semantic column roles and table contracts
│   ├── splits.py    chronological folds, inner splits, date batches
│   └── metrics.py   validated H/D/A probabilistic metrics
└── testing/
    └── synthetic.py deterministic invented fixtures
```

## Canonical data path

```text
synthetic/source CSV
    -> loader
    -> canonical validator
    -> per-division Parquet partitions
    -> DuckDB query layer
```

The canonical schema contains match identity, full-time result and goals, and
the same-match counts needed by the future rolling-feature builder. Bookmaker
odds, database identifiers, operational metadata, and provider-specific extra
columns are excluded.

Parquet destinations are explicit caller inputs. Files are written through
temporary paths, read back for equality, and only then atomically installed.
DuckDB uses an in-memory connection and unions discovered partitions by name.

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
