# Understat xG historical reproduction results

## Verdict

**Gate failed. Disposition: `XG_SIGNAL_REJECTED_FOR_NOW`.**

The candidate is worse by `0.004517` weighted development Log Loss. It
improves two folds, but improvement magnitude must also be at least `0.005`.
The 2025 historical-final result is also unfavorable.

## Public reproduction

| Test season | n | xG feature coverage | LR52 Log Loss | candidate Log Loss | improvement | LR52 Brier | candidate Brier | LR52 ECE | candidate ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 1,826 | 0.9732 | 1.006417 | 1.003709 | +0.002708 | 0.601160 | 0.599335 | 0.034954 | 0.029645 |
| 2023 | 1,752 | 0.9726 | 0.983743 | 0.983464 | +0.000278 | 0.585705 | 0.585680 | 0.010073 | 0.016108 |
| 2024 | 1,752 | 0.0000 | 0.995307 | 1.012149 | -0.016842 | 0.594009 | 0.605383 | 0.032568 | 0.023559 |
| 2025 historical-final | 1,751 | 0.0000 | 0.999847 | 1.003499 | -0.003653 | 0.596213 | 0.598447 | 0.009672 | 0.012444 |
| Development weighted | 5,330 | — | 0.995312 | 0.999829 | -0.004517 | 0.593729 | 0.596835 | 0.025991 | 0.023195 |
| All-fold diagnostic | 7,081 | — | 0.996433 | 1.000736 | -0.004303 | 0.594343 | 0.597233 | 0.021956 | 0.020536 |

Improvement is always LR52 Log Loss minus candidate Log Loss.

## Coverage and validation

The validated artifact supplies 8,955 rows in five divisions and 25 claimed
league-seasons. There are no duplicate keys or invalid xG values. Exact
normalized alignment produces 8,953 rows, with two unmatched external rows.
Minimum claimed league-season coverage is `0.997368`; value coverage is 1.0,
xG spans `0.0` to `6.88189`, league-season side means span `1.060398` to
`1.922032`, and correlation with goals is `0.633083`. Every source quality
gate passes.

## Authority comparison

Against `FBAI_NEW/results/real_xg_eval.json`, the largest fold difference is:

- Log Loss: `0.0000063183`;
- Brier: `0.0000031356`;
- ECE: `0.000571175`.

These pass the recorded tolerances of `0.00001`, `0.00001`, and `0.001`.
The gate verdict, improving-fold count, and disposition reproduce exactly.
No setting was changed to improve parity or force the gate to pass.

## Interpretation

The covered 2022 and 2023 folds show small favorable Log Loss changes. The
artifact has no 2024 or 2025 season history, and the source-defined season
reset makes candidate test features entirely missing in those folds. The
large 2024 regression dominates the development aggregate. This historical
artifact does not support promoting external xG into the stable model.
