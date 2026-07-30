# FBAI — Football Intelligence Research

`football-outcome-lab` is a compact, leakage-aware foundation for football
research. Phase 4 adds a separate closing-market benchmark to the reproducible
LR52 pipeline:

- source-shaped CSV normalization into one canonical schema;
- strict match and natural-key integrity validation;
- atomic, per-division Parquet partitions with read-back verification;
- an in-memory DuckDB query layer over those partitions;
- exactly 52 approved pre-match features (3 Elo, 13 context, 36 rolling);
- date-batched state updates and truncation-invariance tests;
- validated, atomic per-division feature Parquet partitions;
- semantic model-input and feature-table validation;
- deterministic chronological folds and same-date match batching;
- H/D/A-order-safe probabilistic metrics;
- train-only median imputation and standard scaling in every fold;
- multinomial Logistic Regression on the fixed 52-feature tuple;
- uniform and training-prior reference probabilities;
- strict, provider-neutral closing-odds normalization;
- source-verified reciprocal implied probabilities and overround removal;
- exact-key market coverage and same-match LR52 comparison;
- separated development, historical-final, and diagnostic reports;
- deterministic, wholly synthetic raw, canonical, and separate market records;
- tests and CI for those guarantees.

No third-party football or odds data is committed. CI trains and evaluates
only on invented synthetic data. LR52 and the closing market remain distinct
systems; the historical records are offline reproductions, not live forecasts,
market-beating claims, or a betting service.

## Install

Python 3.11 or 3.12 is recommended.

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
mypy src/fbai/core src/fbai/data src/fbai/features src/fbai/models src/fbai/evaluation
```

## Small example

```python
from fbai.evaluation import evaluate_lr52_vs_closing_market
from fbai.features import build_feature_table
from fbai.testing import make_synthetic_canonical_matches, make_synthetic_closing_market

canonical = make_synthetic_canonical_matches(seed=42)
features = build_feature_table(canonical)
market = make_synthetic_closing_market(features, seed=43)
report = evaluate_lr52_vs_closing_market(features, market)
print(report.development.market.log_loss)
```

The development folds are 2022–2024. The 2025 fold is **the historically
frozen final evaluation fold used by the original research**.

## Design boundary

An `_pre` suffix is necessary but never sufficient. A model feature must be
explicitly approved, must carry the suffix, and must not semantically alias a
same-match statistic, label, metadata field, or odds field.

Matches sharing a date are processed as a batch. Their stable display order is
`MatchDate, Division, HomeTeam, AwayTeam`, but no match on a date may consume
another match's result from that same date.

The implemented data path is:

```text
source-shaped CSV
    -> canonical match table
    -> 52-feature pre-match table
    -> train-only preprocessing
    -> LR52

separate closing-odds table
    -> validated, de-vigged H/D/A market probabilities

LR52 + market
    -> identical-key chronological comparison report
```

Closing prices are a strong late-information, near-kickoff benchmark. They are
not assumed to share the earlier timing of every LR52 feature. Historical odds
are not distributed, while CI covers the full path with synthetic market
records.

See:

- [Architecture](docs/ARCHITECTURE.md)
- [Evaluation protocol](docs/EVALUATION_PROTOCOL.md)
- [Data sources](docs/DATA_SOURCES.md)
- [Feature engineering](docs/FEATURE_ENGINEERING.md)
- [Modeling](docs/MODELING.md)
- [Closing market benchmark](docs/MARKET_BENCHMARK.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Project evolution](docs/PROJECT_EVOLUTION.md)

## Licence and data

The code is available under the MIT License.

**The code licence does not grant redistribution rights over third-party
data.** This repository contains synthetic fixtures only.
