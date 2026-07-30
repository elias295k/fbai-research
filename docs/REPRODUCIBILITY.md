# Reproducibility

## Supported environment

- Python 3.11 or 3.12
- runtime: NumPy, pandas, scikit-learn, PyArrow, DuckDB, requests
- development: pytest, Ruff, mypy

Install and validate:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
mypy src/fbai/core src/fbai/data src/fbai/features src/fbai/models src/fbai/evaluation
```

## Determinism controls

- Synthetic generation uses a caller-visible fixed seed.
- Match ordering uses the stable key
  `MatchDate, Division, HomeTeam, AwayTeam`.
- Same-date matches stay in one information batch.
- Probability columns use the fixed order `H, D, A`.
- Fold aggregation is weighted only by recorded sample counts.
- Source aliases normalize into one 20-column canonical schema.
- Canonical Parquet partitions are sorted and verified after writing.
- Feature state is updated only after complete same-date batches.
- The approved model tuple contains exactly 52 ordered `_pre` columns.
- Every chronological fold fits a new imputer, scaler, and LR52 classifier.
- Public probabilities are explicitly ordered H, D, A.

Two invocations of `make_synthetic_fixtures` with the same arguments are
required to be frame-identical. Different seeds are required to change the
generated data while preserving its schema and validation guarantees.

## Phase 3 synthetic model path

```python
from tempfile import TemporaryDirectory
from pathlib import Path

from fbai.data.audit import audit_canonical
from fbai.data.canonical import write_canonical_partitions
from fbai.features import (
    build_feature_table,
    validate_feature_table,
    write_feature_partitions,
)
from fbai.evaluation import evaluate_lr52
from fbai.testing.synthetic import make_synthetic_canonical_matches

matches = make_synthetic_canonical_matches(seed=42)
audit_canonical(matches, input_row_count=len(matches)).raise_for_failure()

with TemporaryDirectory() as temporary:
    destination = Path(temporary)
    canonical_directory = destination / "canonical"
    feature_directory = destination / "features"
    write_canonical_partitions(matches, canonical_directory)
    features = build_feature_table(matches)
    validate_feature_table(features)
    result = write_feature_partitions(features, feature_directory)
    report = evaluate_lr52(features)
    assert report.feature_count == 52
```

This verifies the synthetic loader-to-canonical-to-feature path and Parquet
read-back without network access.

Truncation invariance is tested by building the full synthetic history and a
copy truncated at a date, aligning shared natural keys, and comparing all 52
features with identical NaN positions and `1e-12` numeric tolerance. Future
append/change, current-stat, same-date, shuffle, division, and season cases
provide stronger checks.

CI uses only invented data and never requires the historical archive.
No generated feature files remain after tests because every write uses pytest
or operating-system temporary directories.

## Offline historical record

The committed LR52 research record was produced locally by reading an
authorized, read-only canonical archive, rebuilding the exact 52 features with
the public package, and running `evaluate_lr52`. No source module or source
environment was executed. Only fold counts, date ranges, configuration,
metrics, baseline aggregates, and reference differences were retained.

Historical rows and match-level predictions are not committed and are not
needed by CI. The public API example above contains no private path.

## Reproducibility levels

- **L0:** install the package and reproduce all synthetic tests offline.
- **L1:** run a future source-current downloader with explicit source
  timestamps and checksums.
- **L2:** rebuild from an authorized frozen local input matching a recorded
  checksum.
- **L3:** inspect a historical result record whose underlying third-party
  snapshot cannot be redistributed.

The synthetic L0 path now covers canonical loading, feature engineering,
train-only LR52 fitting, H/D/A prediction, and chronological reporting. The
aggregate research record documents a separate offline historical
reproduction; it is not required for public test execution.
