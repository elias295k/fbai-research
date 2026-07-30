# Match2Vec design

## Research question and authority

Can a train-only representation learned from historical match sequences add
stable predictive information beyond LR52?

The executable authority is the `FBAI_NEW/src/fbai_new/match2vec/` package,
`scripts/run_match2vec_dev.py`, `scripts/run_final_oos.py`, and the committed
`results/match2vec_dev.json` and `results/final_oos.json`. The development grid
selected `M2V-SEQ`, sequence length 10, with the 52-feature numeric branch. The
public phase reproduces only that frozen candidate; it performs no search.

## Corpus and vocabulary

- Team token: sorted `(Division, Team)` pair from the outer training fold.
- Minimum frequency: one; every observed training team is retained.
- Reserved token: one league-specific unknown-team ID per training league.
- OOV: a test-only team maps to its league unknown; an unseen division fails.
- Sequence group: one team within one division.
- Direction: oldest to newest among the last ten qualifying matches.
- Padding: all-zero descriptor rows with a Boolean validity mask.
- Negative samples: none; the selected model is supervised.

Each descriptor contains venue, scaled goals, goal difference, shots, shots on
target, corners, points, log recency, and same-season status. It is built from
the team's perspective using only a match strictly before the target date.
Cross-season history is allowed and explicitly flagged.

Corpus creation sorts by the stable natural key and uses a left date boundary.
Changing a current, same-date, or future result cannot alter the target's
context. Inputs are not mutated and duplicate natural keys fail.

## Train-only representation

Every outer fold builds a new vocabulary and CPU network. The network has a
shared `Linear(13, 32) + ReLU` descriptor encoder, learned scalar attention,
and a four-value league embedding. It concatenates home state, away state,
their difference, and league embedding into 100 learned values.

The source candidate jointly supplies the exact 52 LR52 numeric inputs to a
`Dropout(0.1) + Linear(152, 3)` H/D/A head. Median imputation and scaling fit
on inner-fit rows only. Adam uses learning rate `0.003`, weight decay `0.0001`,
batch 512, at most 300 epochs, patience 15, minimum delta `0.00001`, and seed
42. Python, NumPy, and Torch seeds are fixed; deterministic Torch CPU
algorithms are requested.

The public inner validation split uses the existing whole-date boundary. The
authority selected the final ten percent of rows and could divide a date.
Whole-date behavior is a required safety adaptation, not tuning.

During a test season the trained network remains fixed. A sequence for a later
test date may contain descriptors from an earlier test date because that
result was then available. Same-date results never cross.

## Evaluation and gate

- Development: expanding test seasons 2022, 2023, 2024.
- Historically frozen final: test season 2025.
- Class order: H, D, A.
- Metrics: Log Loss, multiclass Brier, ECE, sample count.
- Aggregation: test-sample-count weighted.
- Comparison: Match2Vec and LR52 on identical rows.

The predefined source gate requires both:

1. development weighted Log Loss improvement at least `0.005`;
2. at least two of three development folds improve.

Improvement is `lr52_log_loss - candidate_log_loss`; positive favors the
candidate. The historical-final result is reported after freezing but does not
select, tune, or rescue the development verdict.

## Reproduction criterion

The strongest structured fold and aggregate artifacts are compared because
no natural-key candidate predictions were committed. Tolerances account for
deterministic neural-library variation and the mandated whole-date inner
split:

- Log Loss absolute difference: `0.003` per fold and aggregate;
- Brier absolute difference: `0.003`;
- ECE absolute difference: `0.005`.

Every fold and aggregate must meet all applicable tolerances. Seeds and model
settings may not be changed to force parity.

## Isolation and scope

Match2Vec is experimental before results because representation learning adds
capacity, a heavy optional dependency, and reproducibility risk without a
guaranteed meaningful gain. LR52 remains the stable default and its exact
52-feature tuple and configuration are unchanged.

Market fields are excluded because the research question concerns historical
match information beyond LR52, not late-information price blending. No raw
historical data, team token, vocabulary, representation, model state,
prediction, or operational artifact is committed.
