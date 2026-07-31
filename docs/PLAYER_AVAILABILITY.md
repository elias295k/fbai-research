# Source-verified player-availability audit

## Boundary

This optional experiment tests whether completed-prior squad usage and
continuity add signal beyond LR52. It is not an injury feed, a lineup predictor,
or a live service. It performs no scraping or provider request, and it does not
change the stable 52-feature model.

The source experiment mixes two timing contracts. Its 63 `*_pre` fields are
reproducible from completed matches. Its 48 `*_near_kickoff` fields consume the
actual target lineup and bench, but the raw artifact has no publication
timestamps. Those 48 fields are therefore timing-unknown and excluded. Target
minutes, substitutions, and participation are post-match and forbidden.

## Information audit

| Input | Timing | Public use |
|---|---|---|
| Fixture identity and date | Known before kickoff | Join and chronology only |
| Appearance and minutes rows | Known after their match | Completed prior dates only |
| Starter roles | Known after/around their match | Completed prior dates only |
| Player valuations | Timestamped | Only dates strictly before target |
| Target official XI and bench | Publication time absent | Forbidden |
| Target minutes/participation/substitutions | After target starts/ends | Forbidden |
| Injury/suspension intervals | Clean table absent | Not inferred or used |

Missing lineup data is not interpreted as an injury. No player-level row or
identifier is emitted from the public builder or retained in the result.

## Features and temporal behavior

The source's 21 base semantics cover recent match/minute load, unique-player
and starter breadth, minute concentration, starter continuity/rotation, and
strictly earlier valuation summaries. Home, away, and home-minus-away versions
produce 63 experimental inputs. Histories are isolated by division/team,
reset by season, and queried strictly before the target date. Same-date matches
form one indivisible batch.

The feature frame joins the canonical table by exact natural key. Duplicate
match, game, player-relation, or valuation identifiers fail. Invalid minutes,
roles, dates, and values fail. Fuzzy matching is not supported.

## Evaluation

LR52 and the 115-input candidate are independently fit on identical covered
keys in expanding 2022–2025 folds. Every pipeline learns median imputation and
standard scaling from its training fold only. The candidate is fixed L2
`lbfgs` Logistic Regression (`C=1`, `max_iter=2000`, tolerance `1e-4`, seed 42,
CPU), with probabilities explicitly reordered to `H, D, A`.

The development gate requires at least `0.005` weighted Log Loss improvement
and improvement in at least two of three development folds. The historical
reproduction yields `-0.003894` and 1/3, then `-0.000974` on 2025. The gate
fails and the disposition is `PLAYER_AVAILABILITY_REJECTED_FOR_NOW`.

See the aggregate-only [design](../research/player_availability/DESIGN.md),
[results](../research/player_availability/RESULTS.md), and
[record](../research/player_availability/result.json).
