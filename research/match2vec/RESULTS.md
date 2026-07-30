# Match2Vec results

## Verdict

**Gate failed — `MATCH2VEC_REJECTED_FOR_NOW`.**

Match2Vec improved all three development folds, but its weighted improvement
was `0.0037639961`, below the predefined `0.005` threshold. The historically
frozen 2025 result is favorable context, not a basis for reversing the failed
development gate. LR52 remains the public default.

## Historical fold results

| Test year | Role | Train | Inner fit | Inner val | Test | LR52 Log Loss | M2V Log Loss | Improvement | M2V Brier | M2V ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | development | 9,639 | 8,615 | 1,024 | 3,296 | 0.9963446047 | 0.9909105383 | +0.0054340664 | 0.5905107013 | 0.0131439203 |
| 2023 | development | 12,935 | 11,483 | 1,452 | 3,228 | 0.9820571914 | 0.9798512076 | +0.0022059839 | 0.5831707260 | 0.0163579899 |
| 2024 | development | 16,163 | 14,553 | 1,610 | 3,228 | 0.9965512557 | 0.9929344989 | +0.0036167569 | 0.5929322867 | 0.0139572539 |
| 2025 | historical final | 19,391 | 17,469 | 1,922 | 3,226 | 1.0065031448 | 1.0003672515 | +0.0061358933 | 0.5983010453 | 0.0199498894 |

Positive improvement means lower Match2Vec Log Loss.

## Weighted views

| View | n | LR52 Log Loss | M2V Log Loss | Improvement | M2V Brier | M2V ECE |
|---|---:|---:|---:|---:|---:|---:|
| Development | 9,752 | 0.9916837453 | 0.9879197492 | +0.0037639961 | 0.5888826699 | 0.0144770271 |
| Historical final | 3,226 | 1.0065031448 | 1.0003672515 | +0.0061358933 | 0.5983010453 | 0.0199498894 |
| All historical diagnostic | 12,978 | 0.9953674703 | 0.9910138810 | +0.0043535893 | 0.5912238380 | 0.0158374412 |

Development folds improved: `3/3`. Required: `2/3`.
Development improvement: `0.0037639961`. Required: `0.005`.

## Vocabulary and OOV audit

| Test year | Training team tokens | League UNK tokens | Total | Home OOV | Away OOV |
|---|---:|---:|---:|---:|---:|
| 2022 | 217 | 9 | 226 | 264 | 264 |
| 2023 | 231 | 9 | 240 | 252 | 252 |
| 2024 | 244 | 9 | 253 | 172 | 172 |
| 2025 | 253 | 9 | 262 | 166 | 165 |

OOV team identities are diagnostic for the fold-local vocabulary. The selected
M2V-SEQ consumes league IDs and descriptor histories rather than team identity
IDs, so promoted teams still use their available strictly-prior descriptors.

## Reference comparison

The committed authority reports development Match2Vec Log Loss
`0.9883866511`, historical-final `0.9991134405`, and all-fold diagnostic
`0.9910530575`. The public values differ by `0.0004669018`, `0.0012538110`,
and `0.0000391765`, respectively.

The largest fold Log Loss difference is `0.0023654617` on 2022. The dominant
protocol difference is the public whole-date inner validation boundary:
2022 uses 8,615/1,024 fit/validation rows rather than the authority's
8,676/963 row-count split. All fold and aggregate differences are within the
declared Log Loss `0.003`, Brier `0.003`, and ECE `0.005` tolerances.
Reproduction verdict: **pass with the documented same-date safety
adaptation**.

Match-level probability parity is unavailable because the authority contains
only structured fold and aggregate results.

## Coverage and limitations

Synthetic CI covers raw-to-canonical conversion, all 52 features, fold-local
vocabulary, OOV mapping, strict sequences, deterministic CPU fitting, learned
feature shape, H/D/A probabilities, identical-row LR52 comparison, weighted
aggregation, and the gate. Engineering tests do not claim synthetic research
quality.

No historical data, team tokens, vocabulary mappings, representations, model
weights, checkpoints, or match predictions are committed. PyTorch/library
variation and the required inner-split adaptation preclude bitwise parity.
No market input or operational functionality is part of this candidate.
