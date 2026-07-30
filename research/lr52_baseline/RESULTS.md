# LR52 baseline results

## Reproduction verdict

**Pass** at tolerance `1e-8`.

- Historical canonical rows rebuilt: 22,617.
- Approved model features: 52.
- Evaluated test rows across four folds: 12,978.
- Reproduced all-fold weighted log loss: `0.9953674703095913`.
- Committed reference: `0.9953674687082226`.
- Absolute difference: `1.601368704307049e-9`.
- Largest per-fold log-loss difference: `6.634087834633817e-9`
  on test year 2023.

## Fold metrics

| Test year | Role | Train rows | Test rows | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| 2022 | development | 9,639 | 3,296 | 0.9963446047 | 0.5937116705 | 0.0125480039 |
| 2023 | development | 12,935 | 3,228 | 0.9820571914 | 0.5846906095 | 0.0154891642 |
| 2024 | development | 16,163 | 3,228 | 0.9965512557 | 0.5948841338 | 0.0190510594 |
| 2025 | historical final | 19,391 | 3,226 | 1.0065031448 | 0.6025104774 | 0.0239979508 |

## Weighted views

| View | n | Model log loss | Brier | ECE | Uniform log loss | Training-prior log loss |
|---|---:|---:|---:|---:|---:|---:|
| Development | 9,752 | 0.9916837453 | 0.5911137138 | 0.0156741245 | 1.0986122887 | 1.0718691334 |
| Historical final | 3,226 | 1.0065031448 | 0.6025104774 | 0.0239979508 | 1.0986122887 | 1.0758889787 |
| All historical diagnostic | 12,978 | 0.9953674703 | 0.5939466588 | 0.0177432156 | 1.0986122887 | 1.0728683645 |

Uniform and training-prior rows are non-market statistical references. No
market comparison was performed.

## Method and limitations

The public Phase 2B builder reconstructed features from the read-only
historical canonical archive, after which the public Phase 3 evaluator fitted
an independent preprocessing/model pipeline in each fold. Synthetic CI covers
the same end-to-end path without requiring historical data.

The committed reference contains fold-level and aggregate metrics but no
match-level prediction arrays, so natural-key probability parity could not be
performed. The aggregate and all four per-fold log losses are within the
declared tolerance.

Raw historical data, team names, match predictions, and model binaries are not
included in this repository. This is an offline historical reproduction, not
a currently untouched final estimate or a deployed forecasting system.
