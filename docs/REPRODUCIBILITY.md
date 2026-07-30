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
mypy src/fbai/core src/fbai/data
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

Two invocations of `make_synthetic_fixtures` with the same arguments are
required to be frame-identical. Different seeds are required to change the
generated data while preserving its schema and validation guarantees.

## Phase 2A synthetic smoke path

```python
from tempfile import TemporaryDirectory
from pathlib import Path

from fbai.data.audit import audit_canonical
from fbai.data.canonical import write_canonical_partitions
from fbai.data.store import query_canonical
from fbai.testing.synthetic import make_synthetic_canonical_matches

matches = make_synthetic_canonical_matches(seed=42)
audit_canonical(matches, input_row_count=len(matches)).raise_for_failure()

with TemporaryDirectory() as temporary:
    destination = Path(temporary)
    write_canonical_partitions(matches, destination)
    counts = query_canonical(
        destination,
        "SELECT Division, COUNT(*) AS n FROM matches GROUP BY Division",
    )
```

This verifies the synthetic loader-to-query path. It does not claim full
research reproduction, feature-table reconstruction, or model reproduction.

## Reproducibility levels

- **L0:** install the package and reproduce all synthetic tests offline.
- **L1:** run a future source-current downloader with explicit source
  timestamps and checksums.
- **L2:** rebuild from an authorized frozen local input matching a recorded
  checksum.
- **L3:** inspect a historical result record whose underlying third-party
  snapshot cannot be redistributed.

Only the synthetic L0 path is implemented through Phase 2A.
