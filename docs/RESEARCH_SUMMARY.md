# Public v1 research summary

## Decision

`DATA_CEILING_UPHELD` is the Phase 5D programme conclusion. LR52 remains the
default internal model because none of the six advanced candidates passed its
predefined development gate. The result does not assert that LR52 is globally
optimal or that future data and models cannot improve it.

## Candidate evidence

All improvements below are computed within an experiment on the exact same
rows as its independently fitted LR52 comparator. Raw Log Loss values must not
be ranked across rows with different population or coverage identifiers.

| Candidate | Evaluation population | Dev n | Aligned LR52 LL | Candidate LL | Improvement | Improved folds | Gate | Historical-final improvement | Disposition | Important coverage or timing limitation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| Match2Vec | Nine-league canonical | 9,752 | 0.991684 | 0.987920 | +0.003764 | 3 | fail | +0.006136 | `MATCH2VEC_REJECTED_FOR_NOW` | Earlier-match sequences only; the 2025 final has already been examined. |
| Graph SVD8 + LR52 | Five-division graph-covered | 5,328 | 0.995433 | 0.993977 | +0.001456 | 2 | fail | -0.000714 | `DATA_CEILING_UPHELD` | 12,452 matched historical rows; graph state uses completed earlier dates only. |
| Deep MLP 128-64 | Nine-league stable-52 | 9,752 | 0.991665 | 0.991501 | +0.000164 | 2 | fail | -0.000112 | `DATA_CEILING_UPHELD` | Public complete-date validation changed architecture selection; final is not currently unseen. |
| Pseudo-xG LR64 | Nine-league canonical | 9,752 | 0.991684 | 0.991645 | +0.000039 | 1 | fail | +0.000847 | `XG_SIGNAL_REJECTED_FOR_NOW` | Estimated from completed prior match statistics, not same-match shot quality. |
| Player availability + LR52 | Five-division availability-covered | 5,328 | 0.995430 | 0.999324 | -0.003894 | 1 | fail | -0.000974 | `PLAYER_AVAILABILITY_REJECTED_FOR_NOW` | Target lineup/bench timestamps are unknown and excluded; coverage filtering changes the comparator population. |
| Understat xG LR68 | Five-division xG-covered | 5,330 | 0.995312 | 0.999829 | -0.004517 | 2 | fail | -0.003653 | `XG_SIGNAL_REJECTED_FOR_NOW` | 8,953 aligned rows; xG coverage ends after the 2023 season start and is lagged only. |

The ordering is by development improvement over each candidate's aligned LR52,
not by raw candidate Log Loss. Each internal gate requires an improvement of at
least `0.005` and at least two improving development folds; no candidate met
both conditions. In the table, “improved folds” is the observed count out of
three; the required count is two.

Taken together, the controlled deep audit shows that additional capacity alone
did not materially improve the stable inputs. The representation, xG, graph,
and player-history signals were weak or unstable under their frozen protocols.
Match2Vec came closest, but its development improvement still fell short of the
gate. These observations motivate better timestamped pre-match information;
they do not claim that any proposed source will improve results.

## Baseline and external benchmark

The LR52 baseline recorded development Log Loss `0.991684` on 9,752 rows and
historical-final Log Loss `1.006503` on 3,226 rows. The closing market, on the
same aligned population, improved over LR52 by `0.025352` in development and
`0.021733` in historical-final evaluation. Closing prices contain later
information and are an external benchmark: the candidate gate does not apply,
and the result is not an internal-model selection or profitability claim.

## Completed source-experiment inventory

The source repositories were read-only. Paths in this table are relative to
the named source repository and are documentation references, not fields in
the machine-readable public result.

| Source experiment | Executable authority | Structured result | Public v1 status | Reason if partial or absent |
| --- | --- | --- | --- | --- |
| LR52 baseline | FBAI_NEW `scripts/run_baseline.py` | `results/baseline.json` | complete | - |
| Match2Vec sequence audit | FBAI_NEW Match2Vec development/robustness/final runners | Match2Vec development, robustness, and final JSON records | complete | Supporting stages are one candidate family. |
| Pseudo-xG audit | FBAI_NEW `scripts/run_xg_eval.py` | `results/xg_eval.json` | complete | - |
| Real historical xG audit | FBAI_NEW `scripts/run_real_xg_eval.py` | `results/real_xg_eval.json` | complete as Understat xG | - |
| Deep model audit | FBAI_NEW deep-audit runner | deep-audit JSON record | internal capacity context represented | Market and timing-uncertain availability neural contexts remain outside the catalogue. |
| Temporal graph audit | FBAI_NEW graph-audit runner | graph-audit JSON record | LR52-plus-graph context represented | Market and combined-availability contexts are outside the public graph contract. |
| Player-availability audit | FBAI_NEW availability runner | availability JSON record | strict-prior context represented | Target lineup and bench context has no verified publication timestamps. |
| Market-aware combination audit | FBAI_NEW market-aware runner | market-aware JSON record | closing benchmark only | Fitted calibration, blends, and market-plus-internal candidates need a market-derived verification phase. |
| Odds-movement audit | FBAI_NEW odds-movement runner | odds-movement JSON record | closing benchmark only | Movement and combined internal candidates are outside the six advanced candidates. |
| Odds feature and stacking audit | FBAI_NEW odds-feature runner | odds-feature JSON record | absent | Proposed later market-derived source-verification phase. |
| Open candidate discovery | FBAI_NEW open-candidate runner | open-candidate JSON record | absent | Recalibration and stacking families need a separate frozen public protocol. |
| Legacy model-family comparison | FBAI `src/fbai/models/run_model_comparison.py` | `reports/modeling/model_family_comparison_metrics.json` | absent | Legacy schema and split semantics differ. |
| Legacy exploration and tuning | FBAI `src/fbai/models/run_exploration.py` | `reports/modeling/exploration_2e1_results.json` | absent | Validation and test semantics require a separate audit. |
| Legacy static style clustering | FBAI `src/fbai/models/run_style_clustering.py` | `reports/modeling/style_clustering_2f1_metrics.json` | absent | Fold and feature contracts require source verification. |
| Legacy opponent-style interactions | FBAI `src/fbai/models/run_interaction_features.py` | `reports/modeling/interaction_features_2f2_metrics.json` | absent | Fold and feature contracts require source verification. |
| Legacy walk-forward robustness | FBAI `src/fbai/models/run_walkforward.py` | `reports/modeling/walkforward_evaluation_metrics.json` | absent | Protocol robustness study, not a normalized advanced candidate. |

## Next research priorities

1. Acquire broad, timestamp-verified pre-kickoff lineup, injury, suspension,
   and availability data.
2. Add transfer and manager-change context with explicit effective timestamps.
3. Evaluate richer event and shot-location data with broad season and division
   coverage.
4. Establish a new untouched temporal final set before selecting more
   candidates.
5. Source-verify the completed market-derived and legacy studies in separate
   phases before drawing comparisons.
