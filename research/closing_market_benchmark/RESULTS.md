# Closing market benchmark results

## Reproduction verdict

**Pass** at `1e-12` tolerance for Log Loss, multiclass Brier score, and ECE.

The public code rebuilt 22,617 historical feature rows from the read-only
canonical archive and aligned all 22,617 separate `AvgCH/AvgCD/AvgCA` market
rows. There were no duplicates, incomplete triplets, invalid prices, or
unmatched keys in either direction. Coverage was 100% overall, in every
division, in every `SeasonStartYear`, and in each evaluated fold.

The direct authority prepared 22,616 rows because its broader experiment
required twelve early/closing Avg/Max fields. The excluded row precedes the
evaluated folds; all 12,978 test rows are identical in the public and
authoritative closing-market evaluations.

## Same-match fold results

| Test year | Role | n | Market Log Loss | Market Brier | Market ECE | Aligned LR52 Log Loss | Aligned LR52 Brier | Aligned LR52 ECE | Market advantage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | development | 3,296 | 0.9737595978 | 0.5794222313 | 0.0226236657 | 0.9963446047 | 0.5937116705 | 0.0125480039 | 0.0225850070 |
| 2023 | development | 3,228 | 0.9548166753 | 0.5667261805 | 0.0249689175 | 0.9820571914 | 0.5846906095 | 0.0154891642 | 0.0272405161 |
| 2024 | development | 3,228 | 0.9702627817 | 0.5775230796 | 0.0203563700 | 0.9965512557 | 0.5948841338 | 0.0190510594 | 0.0262884740 |
| 2025 | historical final | 3,226 | 0.9847704238 | 0.5876376050 | 0.0183554401 | 1.0065031448 | 0.6025104774 | 0.0239979508 | 0.0217327210 |

A positive advantage means the market has lower Log Loss. These are
historical benchmark comparisons, not claims about trading value or
profitability.

## Weighted views

| View | n | Market Log Loss | Market Brier | Market ECE | Aligned LR52 Log Loss | Market advantage | Uniform Log Loss | Training-prior Log Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development | 9,752 | 0.9663318418 | 0.5745910876 | 0.0226494699 | 0.9916837453 | 0.0253519035 | 1.0986122887 | 1.0718691334 |
| Historical final | 3,226 | 0.9847704238 | 0.5876376050 | 0.0183554401 | 1.0065031448 | 0.0217327210 | 1.0986122887 | 1.0758889787 |
| All historical diagnostic | 12,978 | 0.9709152033 | 0.5778341193 | 0.0215820835 | 0.9953674703 | 0.0244522670 | 1.0986122887 | 1.0728683645 |

Historical market coverage is complete in the evaluated folds, so full-test
and aligned-subset LR52 values happen to be equal here. The report preserves
them as distinct fields because they diverge whenever market coverage is
partial.

## Reference parity

The authoritative reference is
`FBAI_NEW/results/odds_movement_eval.json`, model `mkt_close`.

- Every fold-level Log Loss, Brier score, and ECE difference is exactly zero.
- Development weighted Log Loss differs by
  `2.220446049250313e-16`; Brier is identical and ECE differs by
  `3.469446951953614e-18`.
- The reference has no committed natural-key probability rows, so match-level
  probability parity is unavailable.

## Limitations

Closing prices are finalized near kickoff and usually contain more recent
information than LR52's historical features. The comparison is intentionally
a strong late-information benchmark, not an equal-timing contest.

The historical match and odds archives are read-only local inputs and are not
distributed. Only aggregate and fold-level metadata is committed. The 2025
historical-final fold has previously been examined and is not currently
unseen.

No market-superiority claim is made for LR52. No betting, value, edge, stake,
bankroll, settlement, profit, or ROI analysis is part of this record.
