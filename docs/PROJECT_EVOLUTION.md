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
