"""Closed schema for the Phase 2B pre-match feature table."""

from __future__ import annotations

from collections.abc import Iterable

from fbai.features.labels import TARGET_COLUMNS

PRE_FEATURE_SUFFIX = "_pre"

METADATA_COLUMNS: tuple[str, ...] = (
    "MatchDate",
    "SeasonStartYear",
    "Division",
    "HomeTeam",
    "AwayTeam",
    "FTR",
)

ELO_FEATURES: tuple[str, ...] = (
    "HomeElo_pre",
    "AwayElo_pre",
    "EloDiff_pre",
)

CONTEXT_FEATURES: tuple[str, ...] = (
    "DaysSinceLast_H_pre",
    "DaysSinceLast_A_pre",
    "RestDiff_pre",
    "Matches14d_H_pre",
    "Matches14d_A_pre",
    "TeamMatchNum_H_pre",
    "TeamMatchNum_A_pre",
    "SeasonProgress_pre",
    "Form3Home_pre",
    "Form5Home_pre",
    "Form3Away_pre",
    "Form5Away_pre",
    "Form5Diff_pre",
)

ROLLING_BASE_FEATURES: tuple[str, ...] = (
    "GoalsForAvg3",
    "GoalsAgainstAvg3",
    "GoalDiffAvg3",
    "GoalsForAvg5",
    "GoalsAgainstAvg5",
    "GoalDiffAvg5",
    "ShotsForAvg5",
    "ShotsAgainstAvg5",
    "TargetForAvg5",
    "TargetAgainstAvg5",
    "CornersForAvg5",
    "CornersAgainstAvg5",
    "FoulsForAvg5",
    "FoulsAgainstAvg5",
    "YellowForAvg5",
    "YellowAgainstAvg5",
    "RedForAvg5",
    "RedAgainstAvg5",
)

ROLLING_FEATURES: tuple[str, ...] = tuple(
    f"{name}_H_pre" for name in ROLLING_BASE_FEATURES
) + tuple(f"{name}_A_pre" for name in ROLLING_BASE_FEATURES)

FEATURE_COLUMNS: tuple[str, ...] = (*ELO_FEATURES, *CONTEXT_FEATURES, *ROLLING_FEATURES)
FEATURE_TABLE_COLUMNS: tuple[str, ...] = (*METADATA_COLUMNS, *TARGET_COLUMNS, *FEATURE_COLUMNS)

assert len(ELO_FEATURES) == 3
assert len(CONTEXT_FEATURES) == 13
assert len(ROLLING_FEATURES) == 36
assert len(FEATURE_COLUMNS) == 52
assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
assert all(column.endswith(PRE_FEATURE_SUFFIX) for column in FEATURE_COLUMNS)


def feature_columns() -> tuple[str, ...]:
    """Return the immutable, ordered model-input feature contract."""

    return FEATURE_COLUMNS


def feature_output_columns(
    feature_names: Iterable[str] = FEATURE_COLUMNS,
) -> tuple[str, ...]:
    """Return metadata, labels, then the requested feature names."""

    return (*METADATA_COLUMNS, *TARGET_COLUMNS, *tuple(feature_names))
