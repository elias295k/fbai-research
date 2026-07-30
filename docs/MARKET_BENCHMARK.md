# Closing market benchmark

## Why use the closing market

Closing three-way football prices aggregate public information and market
participants' views immediately before kickoff. De-vigged prices are therefore
a demanding external probability benchmark.

They are not an FBAI model. They are also not assumed to have been available
at the earlier time represented by every LR52 feature. The report labels them
`closing_near_kickoff` and keeps that timing asymmetry explicit.

## Separate data boundary

The architecture has two independent inputs:

```text
canonical historical matches -> 52 pre-match features -> LR52 probabilities

separate closing-odds table -> validated and de-vigged market probabilities

LR52 + market probabilities -> exact-key aligned chronological comparison
```

Odds never enter the canonical match schema, `feature_columns()`, LR52
preprocessing, or LR52 fitting. The market table attaches only during
evaluation by:

```text
MatchDate, Division, HomeTeam, AwayTeam
```

The public market schema uses provider-neutral names:

```text
ClosingHomeOdds, ClosingDrawOdds, ClosingAwayOdds
```

The verified historical aliases are the FBAI_NEW closing-average triplet
`AvgCH`, `AvgCD`, `AvgCA`. It represents final pre-kickoff average 1X2
decimal prices. Other source/provider columns are ignored.

## Validation

Market dates are parsed to timezone-naive normalized pandas datetimes.
Division and team fields must be non-empty strings. Natural keys must be
complete and unique.

Every decimal odd must be numeric, finite, and strictly greater than one.
NaN, infinity, nonnumeric values, incomplete H/D/A triplets, and duplicate
keys are never clipped, filled, or silently selected.

`normalize_closing_market` is strict and raises if any odds row is unusable.
`prepare_closing_market` is an explicit evaluation path: it returns only valid
rows together with counts of incomplete and invalid rows. Duplicate keys
still fail.

## Implied probabilities and overround removal

FBAI_NEW's executable authority computes reciprocal implied probabilities in
H, D, A order and divides them by their row sum:

```text
raw_home = 1 / home_odds
raw_draw = 1 / draw_odds
raw_away = 1 / away_odds

overround = raw_home + raw_draw + raw_away

probability_home = raw_home / overround
probability_draw = raw_draw / overround
probability_away = raw_away / overround
```

The implied sum must be finite and positive. It is not required to exceed one:
best-price constructions can produce an underround. The unnormalized sum is
retained as `market_overround` diagnostic metadata. Probability matrices must
be finite, bounded, and sum to one within the same strict H/D/A validator used
by LR52.

## Coverage and fair comparison

Both sides are validated before a one-to-one exact-key merge. The report
counts supplied and valid market rows, incomplete and invalid rows, duplicate
keys, unmatched keys in both directions, aligned rows, and coverage overall,
by division, by `SeasonStartYear`, and by fold. It does not fuzzy-match names,
join by row position, forward-fill prices, or borrow prices from another
match.

For each chronological fold:

1. LR52 fits on the complete valid training history with its unchanged 52
   non-market features.
2. LR52 predicts the complete test fold.
3. Valid closing probabilities attach to test predictions by exact key.
4. LR52, market, uniform, and training-prior probabilities are scored on the
   identical aligned targets.

Full-test LR52 is retained separately from aligned-subset LR52. Aggregates use
aligned sample counts for aligned metrics. The views remain development
(2022–2024), historically frozen final (2025), and all-historical diagnostic.

Metrics are multiclass Log Loss, multiclass Brier score, top-label ECE, and
sample count. The reported delta is:

```text
market_advantage_log_loss =
    lr52_aligned_log_loss - market_log_loss
```

Positive means the market has lower Log Loss; negative means LR52 has lower
Log Loss.

## Data, CI, and scope

Historical odds are subject to third-party data constraints and are not
committed. Local reproduction reads authorized archives without copying them
and retains only aggregate/fold metadata. CI uses deterministic invented
market records with valid, missing, incomplete, duplicate, invalid, and
shuffled modes.

This benchmark reports probability quality, not betting performance. It
contains no expected-value selection, recommendation, stake, bankroll,
settlement, profit, or ROI logic. Market-aware models, blends, recalibration,
opening prices, and odds movement are separate future research questions and
are not implemented in Phase 4.
