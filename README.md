# FBAI — Football Outcome Intelligence

`football-outcome-lab` is a compact, leakage-aware foundation for football
outcome research. The current Phase 0–1 release focuses on correctness
contracts rather than model breadth:

- semantic model-input and feature-table validation;
- deterministic chronological folds and same-date match batching;
- H/D/A-order-safe probabilistic metrics;
- deterministic, wholly synthetic football fixtures;
- tests and CI for those guarantees.

No real football dataset, betting system, live API, database, dashboard, or
research model is included.

## Install

Python 3.11 or 3.12 is recommended.

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
mypy src/fbai/core
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

See:

- [Architecture](docs/ARCHITECTURE.md)
- [Evaluation protocol](docs/EVALUATION_PROTOCOL.md)
- [Data sources](docs/DATA_SOURCES.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Project evolution](docs/PROJECT_EVOLUTION.md)

## Licence and data

The code is available under the MIT License.

**The code licence does not grant redistribution rights over third-party
data.** This repository contains synthetic fixtures only.
