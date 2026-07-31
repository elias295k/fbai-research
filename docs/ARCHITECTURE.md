# Architecture

The package now has a canonical historical-match layer and a closed,
leakage-safe pre-match feature layer alongside the Phase 1 evaluation
contracts, the Phase 3 probabilistic baseline, and a separate Phase 4
closing-market benchmark. Phase 5A adds an optional research namespace without
changing the stable path.

```text
src/fbai/
├── research/
│   ├── common.py         immutable candidate/gate reporting
│   └── match2vec/        optional sequence representation and evaluation
├── models/
│   ├── preprocessing.py fixed 52-column selection and train-only transforms
│   └── logistic.py      source-verified LR52 fit and H/D/A prediction
├── evaluation/
│   ├── baselines.py     uniform and train-fitted class-prior references
│   ├── report.py        immutable JSON-safe fold and aggregate records
│   └── runner.py        expanding chronological LR52 evaluation
├── features/
│   ├── schema.py    exact 3/13/36 model-input contract
│   ├── labels.py    source-verified target construction
│   ├── elo.py       per-division date-batched Elo state
│   ├── context.py   rest, congestion, match number, progress, form
│   ├── rolling.py   prior-match team-perspective rolling statistics
│   ├── checks.py    closed schema and semantic leakage validation
│   └── build.py     orchestration and verified feature partitions
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
    -> 52-feature pre-match table
    -> train-only median imputation and scaling
    -> LR52
    -> chronological evaluation report

separate closing-odds table
    -> strict market validation
    -> reciprocal implied probabilities
    -> row-sum overround removal

LR52 probabilities + market probabilities
    -> exact natural-key intersection
    -> aligned chronological comparison report

experimental branch:
train history
    -> strictly-prior match descriptor sequences
    -> fold-local Match2Vec representation
    -> representation + exact LR52 numeric tuple
    -> identical-row LR52/candidate report
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
target domains, stable order, finite numeric values, and the exact 52-column
model schema. The feature builder and writer both invoke this semantic guard.

## Feature-state boundary

Elo carries across seasons within each division, with a 30 percent reversion
toward 1500 at a transition. Context and rolling histories reset per division,
season, and team. Rolling values are all-venue team-perspective histories with
shift-before-window semantics. Legitimate first-appearance values remain
missing; model-time imputation is outside this layer.

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

## Model boundary

`models.preprocessing` selects the fixed feature tuple without numeric-column
discovery and rejects unknown columns or infinities. `fit_lr52` repeats the
semantic guard immediately before fitting a single encapsulated
imputer/scaler/classifier pipeline. Prediction explicitly maps
scikit-learn's class order to named H/D/A columns.

`evaluation.runner` uses the existing expanding-fold generator. It creates a
new pipeline and training-prior baseline per fold, validates probabilities,
and produces immutable aggregate-only records. No fitted model or match-level
prediction is written to disk.

## Market boundary

`evaluation.market` defines a provider-neutral seven-column closing-market
table. Historical `AvgCH/AvgCD/AvgCA` aliases normalize into
`ClosingHomeOdds/ClosingDrawOdds/ClosingAwayOdds`. Decimal prices must be
finite and greater than one. Reciprocal implied probabilities are normalized
by their finite positive row sum in explicit H/D/A order; the unnormalized
sum remains a diagnostic.

`evaluation.comparison` never adds these fields to the canonical or feature
tables. It validates both natural-key sides, reports coverage, and uses a
one-to-one exact-key intersection. LR52 fits on the full valid training
history and predicts the full fold before test evaluation is restricted to
the identical market-covered matches. Full-test LR52 remains separate from
aligned LR52 in immutable aggregate-only records.

`testing.market` creates a separate deterministic invented market table,
including controlled missing, incomplete, duplicate, invalid, and shuffled
modes. It never changes the canonical synthetic-match API.

## Experimental research boundary

`research.match2vec` is not imported by the stable LR52 path and PyTorch is an
optional extra. Each outer fold creates a training-only vocabulary, inner-fit
numeric preprocessing, and a new CPU network. Its descriptor corpus uses only
matches strictly before each target date, so same-date outcomes cannot cross
the information boundary. The learned 100-value representation has a separate
closed contract and is never accepted by `fit_lr52`.

The source-selected candidate concatenates that representation with the exact
52 LR52 numeric values inside its own linear H/D/A head. It does not alter
`FEATURE_COLUMNS`, `LR52Config`, the LR52 evaluator, or the market evaluator.

`research.pseudo_xg` is a second isolated branch using only canonical
completed-match goals, shots, and shots on target. A Poisson estimator is
trained without test rows; 12 strict-prior rolling aggregates then feed a
separate LR64 pipeline. Candidate-training transforms use season-boundary
walk-forward estimator fits, while each fold's test transform uses the full
outer training partition. The research selector explicitly approves 52 plus
12 names, but the stable selector still accepts exactly 52.

`research.understat_xg` is a third isolated branch. Its six-column external
table is normalized separately and aligned one-to-one to canonical history.
Only source-verified aliases are accepted. The internal history keeps
unmatched canonical slots as missing values to preserve the authority's
trailing-ten semantics, while aggregate reports expose no raw keys.

Sixteen strict-prior, within-season xG aggregates feed a separate LR68
pipeline. All matches on the target date are excluded together. The
candidate's imputer, scaler, and classifier fit only on outer training rows;
LR52 is independently refitted on identical covered-division keys. Neither
the external table nor the 16-name contract is accepted by stable LR52.

`research.deep_capacity` is a fourth isolated branch and reuses the optional
CPU Torch extra. It accepts only the exact stable 52-feature table. Two fixed
fully connected architectures fit from scratch in each development fold;
inner-fit-only imputation/scaling and complete-date validation boundaries
precede deterministic Adam training.

Architecture selection is development-only. The historical-final fold trains
only the selected architecture. The evaluator emits aggregate records and
never writes learned state. Importing or fitting stable LR52 does not import
Torch, and the capacity branch does not alter `FEATURE_COLUMNS`,
`LR52Config`, or any prior evaluator.

`research.graph_model` is a fifth isolated branch. It consumes a separate
local fixture/appearance/lineup contract and constructs a fold-local temporal
team-player bipartite graph. Two fixed seeded `TruncatedSVD` representations
emit 40 or 72 strict-prior fields. No graph dependency, message-passing
network, pretrained representation, or CUDA path is introduced.

Natural-key and game-ID uniqueness are mandatory. Node IDs are deterministic,
and all fixtures on a date are transformed before that date updates team
history. Each outer fold creates a fresh SVD basis and separate graph-only and
LR52-plus-graph Logistic Regression pipelines. Stable LR52 never accepts
graph fields, and graph records contain counts rather than node labels, edge
lists, embeddings, or learned parameters.

## Public v1 synthesis boundary

`research.catalog` is a read-only presentation layer over an explicit allowlist
of eight committed aggregate result records. It neither fits models nor scans
the repository. Shape-specific adapters validate and normalize the baseline,
external benchmark, and six candidates; hashes bind the generated synthesis to
the exact authorities.

Population identifiers are part of the comparison contract. Raw Log Loss
ranking across different identifiers is rejected; only identical-row,
within-experiment improvement over aligned LR52 can order candidates. The
closing market retains the distinct `external_benchmark` role and cannot enter
internal-model selection.
