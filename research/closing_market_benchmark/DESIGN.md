# Closing market benchmark design

## Research question

How does the fixed LR52 baseline compare with a strong external closing 1X2
market benchmark when both are scored on exactly the same chronological test
matches?

The market is an untrained reference. It is not an FBAI model, an LR52 input,
or a claim about information available days before a match.

## Source authority and timing

The executable authority is FBAI_NEW's `implied_probs` implementation and
`run_odds_movement_eval.py`; the strongest direct structured result is
`FBAI_NEW/results/odds_movement_eval.json`. They establish:

- source aliases `AvgCH`, `AvgCD`, `AvgCA`;
- average-bookmaker closing 1X2 decimal prices;
- final pre-kickoff, near-kickoff timing;
- positional class order H, D, A;
- reciprocal implied probabilities normalized by their row sum;
- the 2022–2024 development folds and 2025 historically frozen final fold.

The public names are provider-neutral:

```text
MatchDate, Division, HomeTeam, AwayTeam,
ClosingHomeOdds, ClosingDrawOdds, ClosingAwayOdds
```

Provider extras, targets, same-match statistics, identifiers, and operational
metadata do not enter the canonical market table.

## Mechanical validation and transformation

Keys are parsed and normalized, must be complete and unique, and are stably
ordered by `MatchDate, Division, HomeTeam, AwayTeam`. Each odd must be numeric,
finite, and strictly greater than one. Duplicate keys fail. Strict
normalization also fails incomplete rows; the coverage preparation path may
exclude incomplete or invalid rows only while returning explicit counts.

For each row:

```text
raw_H = 1 / ClosingHomeOdds
raw_D = 1 / ClosingDrawOdds
raw_A = 1 / ClosingAwayOdds
overround = raw_H + raw_D + raw_A
p_i = raw_i / overround
```

The implied sum must be finite and positive, but need not exceed one. The
probability output is explicitly named `probability_home`,
`probability_draw`, `probability_away`; the unnormalized sum is retained only
as `market_overround` diagnostic metadata.

## Coverage and identical-subset comparison

Market rows attach only by the exact natural key. Both sides require unique
keys and the merge is validated one-to-one. There is no row-position join,
fuzzy team matching, forward fill, duplicate selection, or odds repair.

Coverage reports candidate, supplied, valid, incomplete, invalid, duplicate,
unmatched-both-directions, and aligned counts overall, by division, by
`SeasonStartYear`, and by evaluation fold.

For each fold, LR52 fits on the complete valid training history and never sees
market fields. LR52 predicts the complete test fold. Only then are test
predictions intersected with valid market keys; LR52, market, uniform, and
training-prior metrics use this identical aligned target vector. Full-test
LR52 remains a separate contextual metric.

Development, historical-final, and all-historical diagnostic views retain the
Phase 3 roles and use test-sample-count weighting. The comparison is:

```text
market_advantage_log_loss =
    lr52_aligned_log_loss - market_log_loss
```

A positive value means the closing market has lower Log Loss; a negative
value means LR52 has lower Log Loss.

## Reproduction criterion and scope boundary

Reproduction passes when every available closing-market fold-level Log Loss,
Brier score, and ECE differs from the direct structured authority by at most
`1e-12`. No transformation or model parameter may be changed to force parity.

No historical odds are copied into the public repository. The committed
record contains aggregate and fold metadata only. No match-level predictions,
raw prices, models, or provider rows are retained.

Closing prices form a late-information benchmark and therefore are not added
to the 52 earlier pre-match features. This phase contains no market-aware
model, blend, calibration, opening-price study, odds movement, edge,
selection, staking, bankroll, settlement, profit, or ROI diagnostic.
