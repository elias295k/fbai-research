# Pseudo-xG historical reproduction design

## Question

Can a leakage-safe estimate of historical chance quality add stable
predictive information beyond LR52?

## Frozen primary experiment

For every outer fold, completed training matches become two team-side rows.
A Poisson regression estimates goals from shots on target and shots off
target. The public experiment then creates 12 team-perspective, within-season
rolling features over the last 5 and 10 matches and fits Logistic Regression
on the exact LR52 tuple plus those features.

LR52 remains the stable model and is recomputed on identical test keys.
No external xG, provider request, odds, market probability, Match2Vec change,
or feature-contract change is permitted.

## Chronology

- Development test seasons: 2022, 2023, 2024.
- Historically frozen final test season: 2025.
- Outer training contains earlier `SeasonStartYear` values and must end
  strictly before the test partition begins.
- Training pseudo-xG transforms use season-boundary walk-forward estimator
  fits, each restricted to dates before that season starts.
- Test pseudo-xG uses one estimator fitted on the complete outer training
  partition.
- Rolling history uses `bisect_left` on match date, excluding the complete
  target-date batch.
- History is isolated by division, team, and target season.

The walk-forward training transform is the public safety adaptation to the
authority's ambiguous reuse of one outer-training estimator on earlier
training rows.

## Evaluation

Both candidates emit H/D/A probabilities on the same natural keys. Each fold
reports sample count, Log Loss, multiclass Brier score, and top-label ECE.
Development and all-historical aggregates are weighted by test sample count.

The pre-registered gate requires at least `0.005` development Log Loss
improvement and improvement in at least two of three development folds.
Improvement is `LR52 minus candidate`, so positive values favor pseudo-xG.

## Artifact policy

Only configuration, aggregate/fold metadata, reference values, differences,
and tolerances are committed. The record excludes historical rows, natural
keys, teams, predictions, fitted estimators, coefficients, arrays, external
provider data, odds, and market or staking fields.
