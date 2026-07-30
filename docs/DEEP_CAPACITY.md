# Source-verified deep-capacity experiment

## Status and boundary

The deep-capacity audit asks whether a larger nonlinear classifier can extract
stable signal from the same exact 52 leakage-safe inputs used by LR52. It is
an optional CPU Torch experiment under `fbai.research`; LR52 remains the
default.

The authority, in descending order, is:

1. `FBAI_NEW/src/fbai_new/deep.py` and
   `FBAI_NEW/scripts/run_deep_model_audit.py`;
2. `FBAI_NEW/tests/test_deep_audit.py`;
3. `FBAI_NEW/results/deep_model_audit.json`;
4. `FBAI_NEW/reports/DEEP_MODEL_AUDIT.md`.

The original audit also contained market-input and availability-input
families. Phase 5C1 ports only its internal 52-feature MLP family. It does not
add market, availability, Match2Vec, pseudo-xG, Understat, graph, ensemble, or
search inputs.

## Architectures

Both candidates consume the ordered stable `FEATURE_COLUMNS` tuple and emit
three logits in `H, D, A` order.

| Name | Network | Dropout | Parameters |
|---|---|---:|---:|
| `h64_do10_wd1e3` | `52 → 64 → 3` | 0.10 | 3,587 |
| `h128_64_do20_wd1e3` | `52 → 128 → 64 → 3` | 0.20 | 15,235 |

Every hidden layer is `Linear → ReLU → Dropout`. There is no batch
normalization, layer normalization, residual path, embedding, or learned
feature transformation outside the MLP.

The shared training configuration is:

```text
optimizer                 Adam
learning rate             0.003
weight decay              0.001
batch size                512
maximum epochs            220
early-stopping patience   18
early-stopping min delta  1e-5
inner validation fraction 0.1
seed                      42
device                    CPU
```

## Training and selection safety

Every outer fold uses `expanding_folds`. LR52 fits on the complete outer
training window. Each MLP uses the existing `inner_time_split`:

- the chronologically latest ten percent of distinct dates is validation;
- a calendar date can never be divided between fit and validation;
- median imputation and standard scaling fit only on the inner-fit rows;
- a fresh network and Adam optimizer are created per architecture and fold;
- CPU deterministic algorithms and fixed NumPy, Python, and Torch seeds are
  enabled;
- probabilities are validated and kept in explicit `H, D, A` order;
- candidate and LR52 metrics use identical test rows.

Both architectures are evaluated only on development seasons 2022–2024.
Minimum sample-count-weighted development Log Loss selects one architecture.
Only that selected architecture is trained and evaluated on the historical
2025 fold.

## Same-date safety adaptation

The source implementation sorted the outer training window by date and used
the final ten percent of rows for early stopping. That can put matches from
one date on both sides of the split. The public implementation uses the
repository’s date-batch split, as required by the stable chronology contract.

This changes the historical development selection:

- source selection: `h64_do10_wd1e3`, development Log Loss `0.990449`;
- safe public selection: `h128_64_do20_wd1e3`, development Log Loss
  `0.991501`.

The public selection is not altered to force source parity. Per-fold
differences for both architectures remain within the declared safety-adapted
tolerances in the research record.

## Gate and result

The selected candidate uses:

```text
candidate_improvement_log_loss =
    lr52_log_loss - candidate_log_loss

pass only if:
    development improvement >= 0.005
    and at least 2 of 3 development folds improve
```

The safe selected MLP improves two folds, but weighted development improvement
is only `0.000164`. It is worse on the 2025 historical-final fold by
`0.000112`. The gate fails and the source verdict remains:

```text
DATA_CEILING_UPHELD
```

The complete aggregate-only result is in
`research/deep_capacity/result.json`. No checkpoint, tensor, learned weight,
or match-level prediction is retained.
