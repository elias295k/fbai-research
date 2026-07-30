# Deep-capacity historical reproduction results

## Verdict

**Gate failed. Disposition: `DATA_CEILING_UPHELD`.**

Keeping dates indivisible selects the deeper `52 → 128 → 64 → 3` architecture
on development. Its improvement over LR52 is only `0.000164`, far below the
required `0.005`, and it is slightly worse on the historical-final fold.

## Development architecture audit

| Architecture | Parameters | 2022 LL | 2023 LL | 2024 LL | Development weighted LL | Improvement | Improved folds |
|---|---:|---:|---:|---:|---:|---:|---:|
| LR52 | — | 0.996313 | 0.982046 | 0.996539 | 0.991665 | — | — |
| `h64_do10_wd1e3` | 3,587 | 0.994711 | 0.983656 | 0.997537 | 0.991987 | -0.000322 | 1/3 |
| `h128_64_do20_wd1e3` | 15,235 | 0.996094 | 0.981305 | 0.997009 | 0.991501 | +0.000164 | 2/3 |

Positive improvement means the candidate has lower Log Loss.

The selected deeper MLP has development Brier `0.591258` and ECE `0.015722`;
LR52 has Brier `0.591101` and ECE `0.016157`.

## Historical-final and diagnostic

| View | n | LR52 LL | Candidate LL | Improvement | LR52 Brier | Candidate Brier | LR52 ECE | Candidate ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025 historical-final | 3,226 | 1.006522 | 1.006634 | -0.000112 | 0.602523 | 0.601988 | 0.023798 | 0.019225 |
| All-fold diagnostic | 12,978 | 0.995358 | 0.995263 | +0.000095 | 0.593941 | 0.593925 | 0.018056 | 0.016593 |

The final fold cannot rescue the failed development gate.

## Authority comparison

The source’s row-count validation split selected the shallow model, which
reported development Log Loss `0.990449`, improvement `0.001216`, and three
improving folds. It still failed the same gate and reported
`DATA_CEILING_UPHELD`.

The complete-date public adaptation changes both training membership and the
selected architecture. Across both development candidates, maximum absolute
differences from the source are:

- Log Loss: `0.005265`;
- Brier: `0.003089`;
- ECE: `0.010238`.

These pass the declared safety-adapted tolerances of `0.006`, `0.004`, and
`0.011`. Configuration, architecture family, gate, and final disposition
remain source faithful. No parameter was changed to manufacture parity or a
passing result.
