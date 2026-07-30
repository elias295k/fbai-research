# Historical Understat xG experiment

## Status and boundary

Understat xG is an optional offline research input. It is not part of the
canonical match table, the stable 52-feature tuple, normal LR52 fitting, or CI
data. The public package does not scrape, download, or refresh it.

The authority, in descending order, is:

1. `FBAI_NEW/src/fbai_new/external.py`,
   `FBAI_NEW/scripts/convert_kaggle_understat_to_xg_contract.py`, and
   `FBAI_NEW/scripts/run_real_xg_eval.py`;
2. `FBAI_NEW/tests/test_external.py` and
   `FBAI_NEW/tests/test_understat_conversion.py`;
3. `FBAI_NEW/results/external_validation.json` and
   `FBAI_NEW/results/real_xg_eval.json`;
4. `FBAI_NEW/reports/EXTERNAL_SIGNAL_CONTRACT.md` and
   `FBAI_NEW/reports/XG_RESULTS.md`.

The source artifact has one completed-match row with these columns:

```text
Division, MatchDate, HomeTeam, AwayTeam, home_xg, away_xg
```

It contains 8,955 rows across `D1`, `E0`, `F1`, `I1`, and `SP1`, for season
starts 2019 through 2023. It is read only for local historical reproduction
and is not redistributed.

## Schema, names, and alignment

xG values must be numeric, finite, and non-negative. Division codes must be
from the source-verified five-division set. Dates are normalized, accents and
case are normalized deterministically, and only mappings found in the
executable source are accepted. There is no fuzzy match or row-position join.

External and canonical rows align one-to-one on the normalized form of:

```text
MatchDate, Division, HomeTeam, AwayTeam
```

The schema audit reports supplied rows, invalid keys and values, unsupported
divisions, duplicate keys, and valid rows. The alignment report adds unmatched
rows in both directions, aligned rows, and coverage by division,
league-season, and evaluation fold. It never serializes raw match rows.

The source plausibility gates require:

- at least 98% join coverage in each claimed league-season;
- at least 98% present xG values among aligned rows;
- individual xG in `[0, 10]`;
- league-season home and away means in `[0.7, 2.3]`;
- xG-to-goals correlation in `[0.45, 0.95]`.

The historical artifact passes: 8,953 rows align, two external rows and 3,505
covered-division canonical rows are unmatched, and the minimum claimed
league-season coverage is `0.997368`. All 25 claimed league-seasons pass.

## Exact feature contract

Final xG from a completed match is a post-match observation. It can affect
only later fixtures. For each target side, the source takes the last ten
canonical team matches strictly before the target date, restricts to the same
division and `SeasonStartYear`, and then filters to slots with present paired
xG. Partial windows are retained.

Each window of 5 and 10 produces xG for, xG against, their difference, and
goals minus xG-for:

```text
XgForAvg5_H_xg_pre
XgAgainstAvg5_H_xg_pre
XgDiffAvg5_H_xg_pre
GminusXgForAvg5_H_xg_pre
XgForAvg10_H_xg_pre
XgAgainstAvg10_H_xg_pre
XgDiffAvg10_H_xg_pre
GminusXgForAvg10_H_xg_pre
XgForAvg5_A_xg_pre
XgAgainstAvg5_A_xg_pre
XgDiffAvg5_A_xg_pre
GminusXgForAvg5_A_xg_pre
XgForAvg10_A_xg_pre
XgAgainstAvg10_A_xg_pre
XgDiffAvg10_A_xg_pre
GminusXgForAvg10_A_xg_pre
```

`bisect_left` makes each calendar date an indivisible information batch.
Current-match, same-date, and future xG cannot enter a target feature. Stable
natural-key sorting makes shuffled inputs reproduce the same keyed output.
The season restriction prevents carryover.

## Candidate and evaluation

The primary candidate is the exact 52 LR52 features plus the 16 fields above:

```text
68 numeric features
    -> SimpleImputer(strategy="median")
    -> StandardScaler()
    -> LogisticRegression(
           solver="lbfgs",
           effective penalty="l2",
           C=1.0,
           max_iter=2000,
           fit_intercept=true,
           class_weight=None,
           random_state=42,
           tol=1e-4)
```

Every fold fits preprocessing and the classifier only on candidate training
rows. LR52 is refitted independently on the same covered-division training
rows and compared on identical test keys. The class order is explicitly
`H, D, A`.

The gate uses the sample-count-weighted 2022–2024 development result:

```text
candidate_improvement_log_loss =
    lr52_log_loss - candidate_log_loss

pass only if:
    development improvement >= 0.005
    and at least 2 of 3 development folds improve
```

The 2025 fold is historical-final. It cannot select configuration or rescue a
failed development gate.

## Result and limitation

Development LR52 Log Loss is `0.995312`; candidate Log Loss is `0.999829`.
Improvement is therefore `-0.004517`, although two development folds improve
slightly. The 2025 improvement is `-0.003653`. The gate fails and the source
disposition remains `XG_SIGNAL_REJECTED_FOR_NOW`.

The artifact stops after the 2023 season start. Because history resets by
season, xG feature coverage is about 97.3% in the 2022 and 2023 folds and zero
in the 2024 and 2025 folds. This experiment therefore cannot establish
stability in uncovered seasons.

See the aggregate-only record in `research/understat_xg/result.json`.
