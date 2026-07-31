# Player-availability historical reproduction

## Verdict

**Gate failed. Disposition: `PLAYER_AVAILABILITY_REJECTED_FOR_NOW`.**

The strictly-prior candidate worsens development Log Loss by `0.003894` and
improves only one of three development folds. It also trails aligned LR52 on
the 2025 historical-final fold. LR52 remains the default.

## Coverage and quality

Exact-key alignment covers 12,452 of 12,458 Big-5 canonical matches
(`99.9518%`). The six exclusions mean the LR52 comparison population changes;
both models use the same 12,452 keys. Fold coverage is 1,825 / 1,826 in 2022,
1,751 / 1,752 in 2023, and complete in 2024 and 2025.

The local source supplies 374,319 appearance rows, 513,250 lineup-role rows,
and 656,301 valuations. Duplicate match/game/player-relation identifiers,
invalid minutes, invalid roles, invalid valuations, and unmatched relation
games are all zero. Availability-cell coverage is between `96.52%` and
`96.77%` across evaluation folds.

## Fold results

Metrics are Log Loss / Brier / ECE.

| Test year | n | LR52 | LR52 + availability | Improvement |
|---|---:|---|---|---:|
| 2022 | 1,825 | 1.006867 / .601473 / .036148 | 1.017803 / .604623 / .034346 | -.010936 |
| 2023 | 1,751 | .983545 / .585517 / .012964 | .980406 / .583204 / .024054 | +.003139 |
| 2024 | 1,752 | .995394 / .594068 / .034068 | .998981 / .596236 / .022828 | -.003587 |
| 2025 final | 1,751 | .999817 / .596200 / .009221 | 1.000792 / .595981 / .019317 | -.000974 |

Development contains 5,328 matches: LR52 Log Loss is `.995430`, candidate Log
Loss is `.999324`, and the improvement is `-.003894` with 1/3 folds improved.
Across all 7,079 diagnostic rows, the values are `.996515` and `.999687`, for
an improvement of `-.003171`.

## Timing limitation and authority comparison

The authority's 48 target-lineup/bench fields are not ported: their source rows
have no publication timestamps establishing pre-kickoff availability. Target
minutes, participation, and substitutions are post-match and forbidden. The
valid 63-feature table uses completed prior matches and strictly earlier
valuations only.

The rebuilt 63-feature table matches the source cache in every cell. Residual
metric drift is bounded numerical-library drift: maximum absolute differences
are `0.000282` Log Loss, `0.000087` Brier, and `0.001173` ECE, within declared
tolerances of `0.0003`, `0.0001`, and `0.00125`. Gate and disposition are
unchanged.
