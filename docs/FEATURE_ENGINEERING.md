# Feature engineering

Phase 2B turns the validated Phase 2A canonical match table into one
deterministic feature row per match. It does not fit a model.

## Input and output contracts

The input is the 20-column canonical DataFrame, or a directory containing its
per-division Parquet partitions. The output retains:

```text
MatchDate, SeasonStartYear, Division, HomeTeam, AwayTeam, FTR
```

It then carries the five source-verified labels:

```text
target_1x2
target_home_win
target_draw
target_away_win
target_ou25
```

`target_1x2` is H/D/A. The next three fields are its one-hot representation.
`target_ou25` is one when full-time total goals exceed 2.5. Labels and metadata
are never model inputs.

The natural key is `MatchDate, Division, HomeTeam, AwayTeam`; rows use that
same stable order. Validation requires exact row preservation, unique and
complete keys, valid targets, an exact column order, numeric finite-or-missing
features, and the closed 52-name model tuple.

## Exact 52-feature inventory

The model-input groups contain 3 Elo, 13 context, and 36 rolling features.

### Elo (3)

```text
HomeElo_pre, AwayElo_pre, EloDiff_pre
```

Elo starts at 1500, uses K=40 and a 60-point home advantage, and is independent
per division. At each season transition, every known rating moves 30 percent
back toward 1500. The ratings attached to a fixture are always the pre-match
ratings.

### Rest, context, and form (13)

```text
DaysSinceLast_H_pre, DaysSinceLast_A_pre, RestDiff_pre
Matches14d_H_pre, Matches14d_A_pre
TeamMatchNum_H_pre, TeamMatchNum_A_pre, SeasonProgress_pre
Form3Home_pre, Form5Home_pre, Form3Away_pre, Form5Away_pre, Form5Diff_pre
```

Team histories are isolated by division, season start year, and team. Rest is
measured from the previous appearance in the season. `Matches14d` counts prior
matches strictly later than `current date - 14 days`, so a match exactly 14
days old is excluded. Team match number is the zero-based number of prior
season appearances.

Form is all-venue average points over up to the last three or five prior
matches: win=3, draw=1, loss=0. `SeasonProgress_pre` divides the home team's
prior match count by a fixed league schedule size (38 for E0/SP1/I1, 34 for
D1/F1/N1/P1/B1, and 46 for E1). Unknown divisions retain a missing value.

### Rolling match statistics (36)

Each of these 18 bases produces both `_H_pre` and `_A_pre`:

```text
GoalsForAvg3, GoalsAgainstAvg3, GoalDiffAvg3
GoalsForAvg5, GoalsAgainstAvg5, GoalDiffAvg5
ShotsForAvg5, ShotsAgainstAvg5
TargetForAvg5, TargetAgainstAvg5
CornersForAvg5, CornersAgainstAvg5
FoulsForAvg5, FoulsAgainstAvg5
YellowForAvg5, YellowAgainstAvg5
RedForAvg5, RedAgainstAvg5
```

Matches are converted conceptually to a long team-match history. “For” and
“Against” follow the team regardless of whether its historical appearance was
home or away. The current fixture's H/A role only selects which team's state
receives the `_H_pre` or `_A_pre` suffix.

Rolling state is isolated by division, season start year, and team. It has
`shift(1)` semantics: the current match is excluded before applying the
three- or five-appearance window. Early-season windows use at least one prior
valid value. Missing raw statistics are skipped within a window; if none are
valid, the feature remains missing.

## Information timing and missing values

Every `MatchDate` is one information batch. All rows on the date are calculated
from state available before that date. Only after the whole batch has been
featurized are its outcomes and statistics appended to future state. Sorting
inside a date is deterministic and does not imply intra-day information.

First appearances legitimately have missing rest, form, and rolling values.
This layer does not impute, backward-fill, or derive any full-dataset
statistics. Phase 3 will fit imputation on training rows only.

## Leakage proof

The test suite rebuilds a complete synthetic history and a copy truncated at a
cutoff, aligns every shared natural key, and compares all 52 features with a
`1e-12` floating tolerance and identical NaN positions. It also checks future
append/change invariance, current-match-stat exclusion, same-date exclusion,
shuffled inputs, duplicate rejection, and division/season isolation.

The `_pre` suffix is useful documentation but is not trusted as proof. Every
public build and write path passes the exact model tuple through the semantic
Phase 1 guard, which rejects labels, metadata, same-match fields, odds, and
unapproved aliases.

Feature partitions are written to an explicit destination, one deterministic
file per division, through temporary files with read-back validation before
atomic file replacement. No generated features or third-party match data are
committed; tests write only to temporary directories.
