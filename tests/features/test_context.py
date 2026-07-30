from __future__ import annotations

import pandas as pd
import pytest

from fbai.features.context import add_context_features
from fbai.features.schema import CONTEXT_FEATURES

KEY = ["MatchDate", "Division", "HomeTeam", "AwayTeam"]


def context_frame(rows: list[tuple[str, int, str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "SeasonStartYear",
            "Division",
            "HomeTeam",
            "AwayTeam",
            "FTR",
        ],
    ).assign(MatchDate=lambda frame: pd.to_datetime(frame.pop("date")))


def aligned(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(KEY).reset_index(drop=True).loc[:, [*KEY, *CONTEXT_FEATURES]]


def test_first_appearance_rest_form_and_counts_are_explicit() -> None:
    frame = context_frame([("2023-08-01", 2023, "E0", "A", "B", "H")])

    result = add_context_features(frame)

    assert pd.isna(result.loc[0, "DaysSinceLast_H_pre"])
    assert pd.isna(result.loc[0, "DaysSinceLast_A_pre"])
    assert result.loc[0, "Matches14d_H_pre"] == 0
    assert result.loc[0, "TeamMatchNum_H_pre"] == 0
    assert result.loc[0, "SeasonProgress_pre"] == 0.0
    assert pd.isna(result.loc[0, "Form3Home_pre"])


def test_rest_congestion_boundary_match_number_and_all_venue_form() -> None:
    frame = context_frame(
        [
            ("2023-01-01", 2023, "E0", "A", "B", "H"),
            ("2023-01-14", 2023, "E0", "C", "A", "A"),
            ("2023-01-15", 2023, "E0", "A", "D", "D"),
        ]
    )

    result = add_context_features(frame)
    second = result.loc[1]
    third = result.loc[2]

    assert second["DaysSinceLast_A_pre"] == 13
    assert second["Matches14d_A_pre"] == 1
    assert second["TeamMatchNum_A_pre"] == 1
    assert second["Form3Away_pre"] == 3.0
    assert third["DaysSinceLast_H_pre"] == 1
    assert third["Matches14d_H_pre"] == 1  # Jan 1 is exactly 14 days old and excluded.
    assert third["TeamMatchNum_H_pre"] == 2
    assert third["Form5Home_pre"] == 3.0
    assert third["SeasonProgress_pre"] == pytest.approx(2 / 38)


def test_same_date_state_is_not_visible_and_season_resets() -> None:
    frame = context_frame(
        [
            ("2023-08-01", 2023, "E0", "A", "B", "H"),
            ("2023-08-01", 2023, "E0", "A", "C", "A"),
            ("2024-08-01", 2024, "E0", "A", "D", "D"),
        ]
    )

    result = add_context_features(frame)

    assert result.loc[0, "TeamMatchNum_H_pre"] == 0
    assert result.loc[1, "TeamMatchNum_H_pre"] == 0
    assert pd.isna(result.loc[1, "Form5Home_pre"])
    assert result.loc[2, "TeamMatchNum_H_pre"] == 0
    assert pd.isna(result.loc[2, "DaysSinceLast_H_pre"])


def test_shuffle_and_future_truncation_leave_context_unchanged() -> None:
    frame = context_frame(
        [
            ("2023-08-01", 2023, "E0", "A", "B", "H"),
            ("2023-08-01", 2023, "E0", "C", "D", "A"),
            ("2023-08-08", 2023, "E0", "A", "C", "D"),
            ("2023-08-15", 2023, "E0", "B", "D", "H"),
        ]
    )
    full = add_context_features(frame)
    shuffled = add_context_features(frame.sample(frac=1.0, random_state=8))
    truncated = add_context_features(frame.loc[frame["MatchDate"] <= "2023-08-08"])

    pd.testing.assert_frame_equal(aligned(full), aligned(shuffled))
    pd.testing.assert_frame_equal(
        aligned(full.loc[full["MatchDate"] <= "2023-08-08"]),
        aligned(truncated),
    )
