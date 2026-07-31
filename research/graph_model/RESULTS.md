# Temporal graph-model historical reproduction

## Verdict

**Gate failed. Disposition: `DATA_CEILING_UPHELD`.**

The selected `svd8_recent10` representation improves LR52 development Log Loss
by only `0.001456`, below the required `0.005`. It improves two of three
development folds but reverses on the 2025 historical-final fold.

## Graph and model scale

| Test year | Training fixtures | Team nodes | Player nodes | Nonzero edges |
|---|---:|---:|---:|---:|
| 2022 | 5,373 | 120 | 5,293 | 6,449 |
| 2023 | 7,198 | 127 | 6,205 | 8,085 |
| 2024 | 8,949 | 133 | 7,109 | 9,690 |
| 2025 | 10,701 | 137 | 7,989 | 11,289 |

`svd8_recent10` emits 40 graph features. Its graph-only logistic classifier
has 123 parameters; LR52 plus graph has 92 inputs and 279 parameters.
`svd16_recent10` emits 72 graph features, with 219 and 375 parameters in the
same two contexts.

## Development

Metrics are Log Loss / Brier / ECE.

| Model | 2022 | 2023 | 2024 | Weighted development | Improvement | Improved folds |
|---|---|---|---|---|---:|---:|
| LR52 | 1.006867 / .601473 / .036148 | .983545 / .585517 / .012964 | .995405 / .594075 / .033379 | .995433 / .593797 / .027618 | — | — |
| SVD8 graph only | 1.041823 / .626271 / .033895 | 1.036645 / .623098 / .015552 | 1.027332 / .616445 / .018532 | 1.035356 / .621997 / .022815 | -.039923 | 0/3 |
| LR52 + SVD8 | 1.006967 / .601095 / .023738 | .982853 / .585429 / .020503 | .991564 / .591351 / .020372 | .993977 / .592743 / .021568 | +.001456 | 2/3 |
| SVD16 graph only | 1.038648 / .623767 / .021908 | 1.037294 / .623634 / .016999 | 1.028909 / .617183 / .027271 | 1.035000 / .621558 / .022058 | -.039567 | 0/3 |
| LR52 + SVD16 | 1.008309 / .601629 / .034522 | .985851 / .587502 / .016897 | .995602 / .593573 / .013643 | .996750 / .594338 / .021864 | -.001317 | 0/3 |

Positive improvement means lower Log Loss than identical-row LR52.

## Historical final and diagnostic

| View | n | LR52 LL | LR52 + SVD8 LL | Improvement | Candidate Brier | Candidate ECE |
|---|---:|---:|---:|---:|---:|---:|
| 2025 historical final | 1,751 | .999817 | 1.000531 | -.000714 | .596511 | .017045 |
| All-fold diagnostic | 7,079 | .996518 | .995598 | +.000919 | .593675 | .020449 |

The selected graph-only 2025 diagnostic has Log Loss `1.036995`, Brier
`.622458`, and ECE `.015516`.

## Authority comparison

The public complete-date batching adaptation does not change selection,
fold counts, or disposition. Maximum absolute differences across internal
source candidates are:

- Log Loss: `7.63395e-11`;
- Brier: `3.89954e-11`;
- ECE: `5.45771e-11`.

All are within the declared `1e-8` reproduction tolerance. No setting was
changed to improve parity or manufacture a passing result.
