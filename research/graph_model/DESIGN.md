# Temporal graph-model audit design

## Research question

Can graph structure derived only from historical matches add stable predictive
signal beyond LR52?

This is an isolated historical experiment. LR52 remains unchanged and remains
the default model.

## Authority

The implementation follows the source authority in descending order:

1. `graph_embeddings.py` and `run_temporal_graph_embedding_audit.py`;
2. `test_graph_embeddings.py`;
3. `temporal_graph_embedding_eval.json`;
4. `TEMPORAL_GRAPH_EMBEDDING_RESULTS.md`;
5. the preregistered `TEMPORAL_GRAPH_EMBEDDING_PLAN.md`.

The source also evaluated market and player-availability contexts. Phase 5C2
explicitly excludes those inputs, so this port retains only `graph_only` and
`lr52_graph`. Both source-defined embedding dimensions are retained. No model
or representation setting is tuned.

## Graph contract

The representation is a dependency-light temporal team-player bipartite graph,
not a message-passing GNN.

- A team node is a canonical division-team pair.
- A player node is a source player identifier.
- A past player-team match relation contributes
  `minutes_played + 30*starter_flag + 5*bench_flag`.
- The fold matrix contains `log1p` of accumulated team-player edge weights.
- `TruncatedSVD` is fit with seed 42 on the outer-fold training graph.
- Player embeddings are right singular vectors scaled by the square root of
  their singular values.
- An unseen player has a zero embedding and lowers the known-weight coverage
  diagnostic.

For each fixture side, the last ten same-season team matches strictly before
the fixture date form a weighted player pool. The emitted representation
contains home and away vectors, differences, absolute differences, cosine
similarity, L2 distance, known-weight coverage, and player-pool counts.

The fixed configurations are `svd8_recent10` with 40 graph features and
`svd16_recent10` with 72.

## Temporal safety

Outer folds are the repository's predefined expanding 2022–2025 folds. Within
each fold:

- the SVD representation is rebuilt from scratch;
- no graph state is carried from another fold;
- test matches and later matches cannot enter the fitted SVD basis;
- history is queried before the target date is added;
- every date is computed as one indivisible batch, so one match on a date
  cannot affect another match on that date;
- candidate preprocessing and logistic regression fit on outer-train rows
  only;
- LR52 and the candidate are scored on identical natural keys;
- probabilities are explicitly reordered to `H, D, A`.

Both graph dimensions are evaluated on development folds. Selection uses the
lowest sample-count-weighted development Log Loss in the `lr52_graph` context.
Only the selected graph method is built for the historical-final fold, where
both source-defined internal contexts are reported once.

## Classifier and gate

Both contexts use median imputation, standard scaling, and multinomial
Logistic Regression with `lbfgs`, L2, `C=1`, at most 2,000 iterations,
tolerance `1e-4`, seed 42, and CPU execution.

The gate is:

```text
candidate_improvement_log_loss =
    lr52_log_loss - graph_candidate_log_loss

development improvement >= 0.005
and at least 2 of 3 development folds improve
```

The historical-final result is confirmation evidence and cannot rescue a
failed development gate.

## Retained record

The JSON record contains configuration, graph-size, fold, aggregate, gate, and
reproduction metadata only. It excludes node labels, edge lists, embeddings,
learned parameters, match predictions, data files, paths, market fields, and
betting fields.
