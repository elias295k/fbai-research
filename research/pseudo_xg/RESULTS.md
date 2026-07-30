# Pseudo-xG historical reproduction results

## Verdict

**Gate failed. Disposition: `XG_SIGNAL_REJECTED_FOR_NOW`.**

The leakage-adapted public candidate improved sample-count-weighted
development Log Loss by only `0.000039`, far below the predefined `0.005`
threshold, and improved one of three development folds. The 2025
historical-final fold was slightly favorable (`0.000847`) but cannot rescue
the failed development gate.

## Public safety-adapted result

| Test season | n | LR52 Log Loss | LR52+pxg Log Loss | improvement | LR52 Brier | candidate Brier | LR52 ECE | candidate ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 3,296 | 0.996345 | 0.996570 | -0.000225 | 0.593712 | 0.593783 | 0.012548 | 0.009123 |
| 2023 | 3,228 | 0.982057 | 0.982663 | -0.000606 | 0.584691 | 0.585232 | 0.015489 | 0.010993 |
| 2024 | 3,228 | 0.996551 | 0.995598 | +0.000953 | 0.594884 | 0.594390 | 0.019051 | 0.019551 |
| 2025 historical-final | 3,226 | 1.006503 | 1.005657 | +0.000847 | 0.602510 | 0.602095 | 0.023998 | 0.021229 |
| Development weighted | 9,752 | 0.991684 | 0.991645 | +0.000039 | 0.591114 | 0.591153 | 0.015674 | 0.013194 |
| All-fold diagnostic | 12,978 | 0.995367 | 0.995128 | +0.000240 | 0.593947 | 0.593873 | 0.017743 | 0.015191 |

Improvement is always `LR52 Log Loss - candidate Log Loss`; positive favors
pseudo-xG.

## Authority comparison

The authoritative primary experiment reported development Log Loss
`0.991475`, improvement `0.000209`, and two improving development folds.
The public safety adaptation reports `0.991645`, `0.000039`, and one improving
fold. Per-fold candidate differences from the authority remain within the
declared tolerances of `0.001` Log Loss, `0.001` Brier, and `0.005` ECE.

The material design difference is deliberate. The authority used one
outer-training Poisson estimator to transform that same training partition.
The public path instead refits at season boundaries using only earlier dates,
so later training outcomes cannot influence earlier candidate-training
features through fitted Poisson coefficients. Test features still use a
single estimator fitted only on the complete outer training window.

Before applying that adaptation, the source-equivalent public port reproduced
the authority with only numerical/order drift: the largest candidate fold
Log Loss difference was approximately `0.000050`.

## Interpretation

The available shot-volume/accuracy proxy does not provide stable incremental
information beyond LR52 at the predefined meaningfulness threshold. The
result does not test positional shot quality, player identity, or a true
external xG feed. No settings were changed to improve parity or force a gate
pass.
