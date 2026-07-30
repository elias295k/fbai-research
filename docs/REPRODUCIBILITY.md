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
mypy src/fbai/core src/fbai/data src/fbai/features src/fbai/models src/fbai/evaluation src/fbai/research
```

Match2Vec requires the isolated CPU research extra:

```bash
python -m pip install -e ".[dev,match2vec]"
pytest tests/research
```

The normal core environment does not install or import PyTorch.

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
- Synthetic closing odds remain a separate table keyed to invented matches.
- Market normalization, de-vigging, and alignment are deterministic.

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

## Phase 4 synthetic market path

```python
from fbai.evaluation import evaluate_lr52_vs_closing_market
from fbai.features import build_feature_table
from fbai.testing import make_synthetic_canonical_matches, make_synthetic_closing_market

matches = make_synthetic_canonical_matches(seed=42)
features = build_feature_table(matches)
market = make_synthetic_closing_market(features, seed=43)
report = evaluate_lr52_vs_closing_market(features, market)

assert report.feature_count == 52
assert report.development.aligned_lr52.n_samples == report.development.market.n_samples
```

Negative synthetic modes deterministically remove rows, omit an odds
component, duplicate a key, insert an invalid odd, or shuffle rows. This
exercises coverage and failure behavior without real provider data, network
access, secrets, or live prices.

## Offline historical record

The committed LR52 research record was produced locally by reading an
authorized, read-only canonical archive, rebuilding the exact 52 features with
the public package, and running `evaluate_lr52`. No source module or source
environment was executed. Only fold counts, date ranges, configuration,
metrics, baseline aggregates, and reference differences were retained.

Historical rows and match-level predictions are not committed and are not
needed by CI. The public API example above contains no private path.

The closing-market reproduction separately reads the authorized FBAI_NEW
closing-average columns, normalizes them through the public market code, and
aligns them to rebuilt feature rows by exact natural key. It compares
fold-level Log Loss, Brier score, and ECE with the strongest direct committed
market result. The odds file is not copied because the public code licence
does not grant redistribution rights over third-party data.

`research/closing_market_benchmark/result.json` is independently inspectable:
it records the schema, transformation, timing, coverage, same-match metrics,
reference values, absolute differences, tolerances, and verdict. It contains
no raw prices, team rows, match-level predictions, or machine paths.

## Reproducibility levels

- **L0:** install the package and reproduce all synthetic tests offline.
- **L1:** run a future source-current downloader with explicit source
  timestamps and checksums.
- **L2:** rebuild from an authorized frozen local input matching a recorded
  checksum.
- **L3:** inspect a historical result record whose underlying third-party
  snapshot cannot be redistributed.

The synthetic L0 path now covers canonical loading, feature engineering,
train-only LR52 fitting, separate market validation/de-vigging, exact-key
alignment, H/D/A evaluation, and chronological reporting. Aggregate research
records document separate offline historical reproductions; they are not
required for public test execution.

## Phase 5A synthetic Match2Vec path

The research tests build invented multi-season, multi-division canonical rows,
the exact 52-feature table, fold-local vocabularies, strictly-prior sequences,
and fresh CPU networks. They check OOV mapping, same-date exclusion, future
invariance, deterministic repeated fitting, finite 100-value representations,
H/D/A probabilities, identical-row LR52 comparison, weighted aggregates, and
the frozen gate. No model download, checkpoint, or provider request occurs.

The local historical reproduction read the authorized FBAI canonical
partitions, selected only the public canonical fields, rebuilt all 52 features,
and ran `evaluate_match2vec_candidate`. The public inner validation split keeps
whole dates together, while the older authority selected the last ten percent
of rows and could divide a date. That mandated safety adaptation is recorded
with exact fit/validation counts and reference differences.

The committed Match2Vec record contains only aggregate/fold metrics,
configuration, vocabulary sizes, OOV occurrence counts, and reference
differences. Historical rows, natural keys, team tokens, vocabularies, learned
representations, model weights, and predictions are not committed.

## Phase 5B1 synthetic and historical pseudo-xG paths

Synthetic tests build invented multi-season, multi-division canonical rows
and prove the exact estimator configuration, fold-local fitting, train/test
separation, current/future/same-date exclusion, season and division
isolation, keyed shuffle invariance, H/D/A ordering, identical LR52 test
keys, weighted aggregation, stable 52-feature fingerprint, and exact gate.

The historical reproduction reads authorized local canonical partitions,
selects only the public canonical fields, rebuilds the exact 52 stable
features, and runs the public pseudo-xG evaluator. It makes no provider
request and retains no generated data, predictions, parameters, or models.
`research/pseudo_xg/result.json` contains only aggregate/fold metadata and
explicit comparisons with the committed authority.

## Phase 5B2 synthetic and historical Understat xG paths

Synthetic tests construct a separate invented external table and exercise
schema failures, explicit aliases, duplicate detection, exact one-to-one
alignment, coverage views, quality gates, the exact 16-feature order, and all
four truncation invariants. Evaluation tests prove train-only preprocessing,
identical LR52/candidate keys, H/D/A ordering, weighted aggregation, and the
frozen gate. No network operation or external artifact is used by tests.

The L3 historical reproduction reads the validated local xG parquet and local
canonical/feature partitions in place. It does not execute a downloader,
retain a generated table, or write into either source repository. The
aggregate-only record reports artifact checksum, schema and coverage counts,
fold metrics, gate values, and reference differences. It contains no raw
rows, match predictions, fitted objects, or machine paths.

## Phase 5C1 synthetic and historical deep-capacity paths

Synthetic tests exercise the exact architecture grid and parameter counts,
CPU same-seed determinism, H/D/A normalization, fresh fold-local training,
inner-fit preprocessing, complete-date validation boundaries, keyed shuffle
equivalence, development-only selection, final-fold isolation, weighted
aggregation, the exact gate, stable feature fingerprint, and preservation of
prior research records.

The historical reproduction reads the authorized local 52-feature archive
in place and retains the authority’s published 22,616-row comparison scope.
Only the 52 internal fields enter either model. The public inner split keeps
dates whole, unlike the source row-count split; selection and numerical
differences are recorded explicitly. No learned state, tensors, predictions,
or generated dataset is retained.
