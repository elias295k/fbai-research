# Public v1 research synthesis design

## Purpose

Phase 5D closes the public v1 research programme without changing any model,
feature, split, gate, or previously committed result. The synthesis is a
catalogue over eight immutable JSON authorities:

- the LR52 baseline;
- the closing-market external benchmark;
- Match2Vec;
- pseudo-xG;
- Understat xG;
- controlled deep capacity;
- temporal graph SVD;
- player availability.

`src/fbai/research/catalog.py` contains the explicit allowlist and one adapter
per result shape. It performs no discovery, provider access, fitting, or
import-time I/O. `load_research_catalog()` reads only the allowlisted files.

## Validation contract

Catalogue construction fails on missing required fields, duplicate experiment
identifiers, invalid sample counts, non-finite metrics, mismatched candidate
and aligned-LR52 row counts, or inconsistent improvement signs. It recomputes
each candidate's development improvement as:

```text
aligned LR52 Log Loss - candidate Log Loss
```

It also validates the stored threshold, observed improvement, improved-fold
count, required-fold count, and gate verdict. SHA-256 digests bind the
synthesis to the exact eight source records.

## Comparison rules

Candidate raw Log Loss is meaningful only on its own aligned evaluation
population. The public ranking therefore uses within-experiment improvement
over the independently fitted, identical-row LR52 comparator. The catalogue
API deliberately raises `CrossPopulationComparisonError` when asked to rank
raw Log Loss across population identifiers.

The closing market is an external late-information benchmark. It is not an
internal candidate, is not eligible for selection, and is not subject to the
internal `0.005` advancement gate.

## Closure rule

LR52 remains the default because none of the six advanced internal candidates
passed its predefined development gate. That rule, and only that rule,
supports `DATA_CEILING_UPHELD`. The conclusion is a bounded programme result,
not a proof of a universal modelling or information ceiling.

## Source-inventory method

Both authorized source repositories were searched for executable experiment
entry points, structured result records, and written protocols. Completed
experiments absent from public v1 were catalogued but not ported. Market-derived
combinations and legacy tree, cluster, interaction, and walk-forward studies
need later source-verification phases because their timing, schemas, split
semantics, or candidate roles do not match the normalized public contract.

## Safe publication

The generated result contains aggregate metrics, logical authority labels,
coverage descriptions, dispositions, limitations, and hashes only. It contains
no raw football or player rows, match identifiers, predictions, lineups,
embeddings, learned parameters, credentials, or machine-specific paths.

