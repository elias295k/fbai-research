# Deep-capacity historical reproduction design

## Question

Does fixed additional neural-network capacity produce a stable improvement
beyond LR52 when both consume the same exact 52 approved inputs?

## Frozen scope

- candidate family: source `internal_mlp_lr52`;
- inputs: exact ordered LR52 52-feature tuple;
- architectures: one 64-unit hidden layer and a 128/64 two-hidden-layer MLP;
- activation: ReLU;
- dropout: 0.10 and 0.20 respectively;
- normalization: none;
- optimizer: Adam with learning rate `0.003` and weight decay `0.001`;
- maximum epochs: 220, patience: 18, minimum delta: `1e-5`;
- batch size: 512, seed: 42, CPU only;
- folds: development 2022–2024 and historical-final 2025;
- selection: minimum development sample-count-weighted Log Loss;
- gate: at least `0.005` development improvement and at least two improving
  development folds.

No architecture or training parameter is searched. The source audit’s other
input families are out of scope.

## Safety adaptation

The authority’s row-count inner split can divide a calendar date. The public
port uses `fbai.core.splits.inner_time_split`, which selects complete dates
for validation. Imputation and scaling fit on inner-fit rows only. The final
fold is unavailable during architecture selection and is evaluated only for
the selected safe-public architecture.

## Output boundary

The evaluator returns immutable fold and aggregate records. It writes no
checkpoint or prediction. The committed record includes configurations,
parameter counts, row counts, epochs, metrics, gate observations, and
reference differences only.
