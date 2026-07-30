from __future__ import annotations

from fbai.features.labels import TARGET_COLUMNS
from fbai.features.schema import (
    CONTEXT_FEATURES,
    ELO_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_TABLE_COLUMNS,
    METADATA_COLUMNS,
    ROLLING_FEATURES,
    feature_columns,
)


def test_exact_ordered_52_feature_contract() -> None:
    expected = (
        "HomeElo_pre",
        "AwayElo_pre",
        "EloDiff_pre",
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
        "GoalsForAvg3_H_pre",
        "GoalsAgainstAvg3_H_pre",
        "GoalDiffAvg3_H_pre",
        "GoalsForAvg5_H_pre",
        "GoalsAgainstAvg5_H_pre",
        "GoalDiffAvg5_H_pre",
        "ShotsForAvg5_H_pre",
        "ShotsAgainstAvg5_H_pre",
        "TargetForAvg5_H_pre",
        "TargetAgainstAvg5_H_pre",
        "CornersForAvg5_H_pre",
        "CornersAgainstAvg5_H_pre",
        "FoulsForAvg5_H_pre",
        "FoulsAgainstAvg5_H_pre",
        "YellowForAvg5_H_pre",
        "YellowAgainstAvg5_H_pre",
        "RedForAvg5_H_pre",
        "RedAgainstAvg5_H_pre",
        "GoalsForAvg3_A_pre",
        "GoalsAgainstAvg3_A_pre",
        "GoalDiffAvg3_A_pre",
        "GoalsForAvg5_A_pre",
        "GoalsAgainstAvg5_A_pre",
        "GoalDiffAvg5_A_pre",
        "ShotsForAvg5_A_pre",
        "ShotsAgainstAvg5_A_pre",
        "TargetForAvg5_A_pre",
        "TargetAgainstAvg5_A_pre",
        "CornersForAvg5_A_pre",
        "CornersAgainstAvg5_A_pre",
        "FoulsForAvg5_A_pre",
        "FoulsAgainstAvg5_A_pre",
        "YellowForAvg5_A_pre",
        "YellowAgainstAvg5_A_pre",
        "RedForAvg5_A_pre",
        "RedAgainstAvg5_A_pre",
    )

    assert FEATURE_COLUMNS == expected
    assert feature_columns() == expected
    assert len(FEATURE_COLUMNS) == 52
    assert all(column.endswith("_pre") for column in FEATURE_COLUMNS)


def test_feature_group_counts_and_output_shape() -> None:
    assert (len(ELO_FEATURES), len(CONTEXT_FEATURES), len(ROLLING_FEATURES)) == (3, 13, 36)
    assert FEATURE_COLUMNS == (*ELO_FEATURES, *CONTEXT_FEATURES, *ROLLING_FEATURES)
    assert FEATURE_TABLE_COLUMNS == (*METADATA_COLUMNS, *TARGET_COLUMNS, *FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
