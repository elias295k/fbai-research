# FBAI — Football Intelligence Research

`football-outcome-lab` is a compact, leakage-aware foundation for football
research. Phase 2B adds deterministic pre-match feature engineering:

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
- deterministic, wholly synthetic raw and canonical matches;
- tests and CI for those guarantees.

No third-party football data is committed. Phase 2B builds the 52-feature
table but does not train or evaluate a model.

## Install

Python 3.11 or 3.12 is recommended.

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
mypy src/fbai/core src/fbai/data src/fbai/features
```

## Small example

```python
from fbai.core.leakage import validate_feature_table
from fbai.core.splits import ALL_TEST_YEARS, expanding_folds
from fbai.testing.synthetic import SYNTHETIC_FEATURE_COLUMNS, make_synthetic_fixtures

matches = make_synthetic_fixtures(seed=42)
validate_feature_table(
    matches,
    approved_pre_features=SYNTHETIC_FEATURE_COLUMNS,
)
folds = expanding_folds(matches, test_years=ALL_TEST_YEARS)
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
    -> future evaluation layer
```

See:

- [Architecture](docs/ARCHITECTURE.md)
- [Evaluation protocol](docs/EVALUATION_PROTOCOL.md)
- [Data sources](docs/DATA_SOURCES.md)
- [Feature engineering](docs/FEATURE_ENGINEERING.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Project evolution](docs/PROJECT_EVOLUTION.md)

## Licence and data

The code is available under the MIT License.

**The code licence does not grant redistribution rights over third-party
data.** This repository contains synthetic fixtures only.
