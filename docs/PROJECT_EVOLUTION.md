# Project evolution

This repository is a curated successor to two private-development codebases:

- `FBAI`, which contained a canonical data layer, a 52-feature pre-match
  builder, operational database/API code, and several generations of legacy
  experiments;
- `FBAI_NEW`, which contained stronger fit-time leakage guards,
  chronological evaluation splits, order-safe metrics, and research-cycle
  harnesses.

Phase 0–1 deliberately ports only the trust foundation. The operational
surface, legacy betting logic, third-party data, and research models remain
outside this repository.

## What changed in Phase 0–1

- Build-time and fit-time leakage concerns were rewritten as one semantic
  column-classification API.
- Feature approval is explicit; `_pre` naming alone does not prove safety.
- Chronological splits sort on a stable four-column key and hard-fail on
  date overlap.
- Same-date matches are grouped into indivisible batches.
- Probability metrics validate their inputs and use the explicit class order
  H, D, A.
- Tests use invented teams, leagues, dates, outcomes, and features generated
  from a fixed seed.

## Future phases

Canonical ingestion and the verified 52-feature builder may be added after
the Phase 0–1 contracts are reviewed. Historical experiment records may be
added later as compact, provenance-tagged evidence. Model families and
external data adapters are not part of the current scope.

## Phase 5B1: historical pseudo-xG

Later phases added the canonical/feature path, LR52 reproduction, closing
market benchmark, and isolated Match2Vec experiment. Phase 5B1 now adds the
authoritative primary pseudo-xG question: a fold-local Poisson estimate of
completed-match chance quality, 12 strict-prior rolling features, and an
LR52-plus-12 comparison.

The authority's outer-train transformation ambiguity is tightened with
season-boundary walk-forward training transforms. LR52, Match2Vec, market
records, and the stable 52-feature contract remain unchanged.

## Phase 5B2: historical Understat xG

Phase 5B2 adds the authoritative external aggregate-xG question without
changing the canonical table or LR52. A separate six-column schema,
deterministic source aliases, exact one-to-one alignment, aggregate quality
gates, and 16 strict-prior rolling fields feed an isolated LR68 candidate.

The historical artifact aligns 8,953 rows across five divisions and passes
all source quality gates, but ends after the 2023 season start. The candidate
is worse on weighted development Log Loss and retains the source disposition
`XG_SIGNAL_REJECTED_FOR_NOW`. No external rows are committed or needed by CI.

## Phase 5C1: controlled deep capacity

Phase 5C1 holds the stable input constant and ports the source’s two internal
Torch MLPs. The public evaluator strengthens chronological early stopping by
keeping dates indivisible, selects capacity only on development folds, and
evaluates 2025 once for the selected architecture.

That adaptation changes the selected network but not the conclusion:
development improvement is far below `0.005`, the final fold is not
confirmatory, and `DATA_CEILING_UPHELD` remains. Torch stays optional, LR52
stays default, and no prior research record changes.

## Phase 5C2: temporal graph representation

Phase 5C2 ports the source's fixed team-player bipartite SVD experiment rather
than introducing a GNN search. Two dimensions, one ten-match same-season
window, frozen edge weights, and regularized Logistic Regression are
evaluated on the matched 12,452-row scope.

The public version strengthens same-date behavior by applying all graph
history updates after the complete date batch. It reproduces the internal
authority within `1e-8`: SVD8 plus LR52 improves development Log Loss by only
`0.001456` and reverses on final. The gate fails and
`DATA_CEILING_UPHELD` remains. Market and availability branches are excluded,
LR52 remains default, and no graph artifact or prior result is changed.

## Phase 5C3: player availability

Phase 5C3 ports only source fields that can be reconstructed from completed
prior matches and strictly earlier valuations. It excludes target official
lineups, bench membership, participation, minutes, and substitutions because
their pre-kickoff timing is not established by the source artifact.

The safe 63-feature candidate is worse than aligned LR52 in development and
historical-final evaluation. It fails the frozen gate, remains research-only,
and leaves the stable 52-feature model unchanged.

## Phase 5D: public v1 closure

Phase 5D catalogues the baseline, the external closing-market benchmark, and
all six public advanced candidates without rerunning or rewriting them. It
validates the authorities, ranks only within-experiment improvement, records
the remaining completed source experiments for later verification, and makes
the programme limitations explicit.

No advanced candidate passed its development gate. Public v1 therefore closes
with `DATA_CEILING_UPHELD` and LR52 as the default. This is a bounded research
decision, not proof that better information or future methods cannot improve
the task.
