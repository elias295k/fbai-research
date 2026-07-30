# Reproducibility

## Supported environment

- Python 3.11 or 3.12
- runtime: NumPy, pandas, scikit-learn
- development: pytest, Ruff, mypy

Install and validate:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
mypy src/fbai/core
```

## Determinism controls

- Synthetic generation uses a caller-visible fixed seed.
- Match ordering uses the stable key
  `MatchDate, Division, HomeTeam, AwayTeam`.
- Same-date matches stay in one information batch.
- Probability columns use the fixed order `H, D, A`.
- Fold aggregation is weighted only by recorded sample counts.

Two invocations of `make_synthetic_fixtures` with the same arguments are
required to be frame-identical. Different seeds are required to change the
generated data while preserving its schema and validation guarantees.

## Reproducibility levels

- **L0:** install the package and reproduce all synthetic tests offline.
- **L1:** run a future source-current downloader with explicit source
  timestamps and checksums.
- **L2:** rebuild from an authorized frozen local input matching a recorded
  checksum.
- **L3:** inspect a historical result record whose underlying third-party
  snapshot cannot be redistributed.

Only L0 is implemented in Phase 0–1.
