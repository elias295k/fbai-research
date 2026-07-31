# Source-verified temporal graph-model audit

## Status

The graph audit asks whether a temporal team-player representation adds stable
predictive signal beyond the exact LR52 inputs. It is an isolated research API
under `fbai.research.graph_model`; LR52 remains unchanged and remains the
default.

This is not a GCN, GAT, GraphSAGE model, or other message-passing network. The
source deliberately used a dependency-light bipartite graph plus
`TruncatedSVD` and regularized Logistic Regression. No graph library or new
optional dependency is required.

## Source boundary

The executable authority is `graph_embeddings.py` and
`run_temporal_graph_embedding_audit.py`, supported by
`test_graph_embeddings.py`, `temporal_graph_embedding_eval.json`, and
`TEMPORAL_GRAPH_EMBEDDING_RESULTS.md`.

The source audit also evaluated closing-market and player-availability
contexts. Phase 5C2 excludes both, so the public audit contains only:

- graph embeddings alone, as a diagnostic;
- the stable 52 features plus graph embeddings, as the candidate against
  identical-row LR52.

It adds no xG, Match2Vec, odds, availability, search, blending, operational, or
betting behavior.

## Graph definition

Nodes are canonical division-team pairs and source player identifiers. Each
past player-team match relation contributes:

```text
minutes_played + 30 * starter_flag + 5 * bench_flag
```

For every outer fold, the team-player matrix is rebuilt from that fold's
training fixtures. Matrix values are `log1p` accumulated edge weights. A
seeded `TruncatedSVD` produces player embeddings from right singular vectors
scaled by square-root singular values.

Target-side features aggregate player embeddings from the team's last ten
same-season matches strictly before the target date. They include:

- home and away vectors;
- signed and absolute vector differences;
- cosine similarity and L2 distance;
- known-player weight coverage;
- historical player-pool counts.

The fixed representations are:

| Configuration | Dimension | Graph features | Graph-only parameters | LR52 + graph parameters |
|---|---:|---:|---:|---:|
| `svd8_recent10` | 8 | 40 | 123 | 279 |
| `svd16_recent10` | 16 | 72 | 219 | 375 |

Classifier parameter counts include three multinomial intercepts.

## Temporal and fitting safeguards

- Fixtures use deterministic natural-key ordering.
- Duplicate fixture keys and game IDs fail.
- Node IDs are deterministic under shuffled inputs.
- The graph basis is independently fitted inside every fold.
- Test or future fixtures cannot enter that fold's basis.
- All fixtures on one date are computed before the date batch updates
  histories.
- Current-date appearances and lineups cannot affect that date's features.
- Median imputation, scaling, and Logistic Regression fit on outer-train rows
  only.
- Candidate and LR52 probabilities use identical test natural keys.
- Output probabilities are finite, normalized, and explicitly ordered
  `H, D, A`.
- Development folds alone select the graph configuration; the selected method
  is evaluated once on the historical-final fold.

The classifier is the source configuration: multinomial `lbfgs` Logistic
Regression, L2, `C=1`, 2,000 maximum iterations, tolerance `1e-4`, seed 42,
and CPU execution.

## Historical result

`svd8_recent10` is selected. On 5,328 development matches:

- LR52 Log Loss: `0.995433`;
- LR52 + graph Log Loss: `0.993977`;
- improvement: `0.001456`;
- improving folds: `2/3`.

The predefined gate requires at least `0.005` development improvement and
two improving folds. The magnitude condition fails. On 1,751 historical-final
matches, LR52 + graph is worse by `0.000714`.

The disposition is:

```text
DATA_CEILING_UPHELD
```

The aggregate-only record is
`research/graph_model/result.json`. No node labels, edge list, embedding,
learned parameter, match prediction, or external data file is retained.
