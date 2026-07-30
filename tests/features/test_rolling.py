from __future__ import annotations

import pandas as pd
import pytest

from fbai.features.rolling import add_rolling_features
from fbai.features.schema import ROLLING_FEATURES

KEY = ["MatchDate", "Division", "HomeTeam", "AwayTeam"]


def rolling_frame(
    rows: list[tuple[str, int, str, str, str, int, int]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for date, season, division, home, away, home_value, away_value in rows:
        record: dict[str, object] = {
            "MatchDate": pd.Timestamp(date),
            "SeasonStartYear": season,
            "Division": division,
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHome": home_value,
            "FTAway": away_value,
        }
        for home_column, away_column in (
            ("HomeShots", "AwayShots"),
            ("HomeTarget", "AwayTarget"),
            ("HomeCorners", "AwayCorners"),
            ("HomeFouls", "AwayFouls"),
            ("HomeYellow", "AwayYellow"),
            ("HomeRed", "AwayRed"),
        ):
            record[home_column] = home_value
            record[away_column] = away_value
        records.append(record)
    return pd.DataFrame(records)


def aligned(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(KEY).reset_index(drop=True).loc[:, [*KEY, *ROLLING_FEATURES]]


def test_shift_before_rolling_and_first_match_missing_policy() -> None:
    frame = rolling_frame(
        [
            ("2023-08-01", 2023, "E0", "A", "B", 2, 1),
            ("2023-08-08", 2023, "E0", "A", "C", 9, 7),
        ]
    )

    result = add_rolling_features(frame)

    assert result.loc[0, list(ROLLING_FEATURES)].isna().all()
    assert result.loc[1, "GoalsForAvg3_H_pre"] == 2.0
    assert result.loc[1, "GoalsAgainstAvg3_H_pre"] == 1.0
    assert result.loc[1, "ShotsForAvg5_H_pre"] == 2.0
    assert result.loc[1, "ShotsAgainstAvg5_H_pre"] == 1.0


def test_three_and_five_match_windows_and_team_perspective_are_exact() -> None:
    frame = rolling_frame(
        [
            ("2023-08-01", 2023, "E0", "A", "B", 1, 0),
            ("2023-08-08", 2023, "E0", "C", "A", 2, 3),
            ("2023-08-15", 2023, "E0", "A", "D", 5, 4),
            ("2023-08-22", 2023, "E0", "E", "A", 7, 6),
            ("2023-08-29", 2023, "E0", "A", "F", 9, 8),
            ("2023-09-05", 2023, "E0", "G", "A", 11, 10),
        ]
    )

    result = add_rolling_features(frame)
    current = result.iloc[-1]

    assert current["GoalsForAvg3_A_pre"] == pytest.approx((5 + 6 + 9) / 3)
    assert current["GoalsAgainstAvg3_A_pre"] == pytest.approx((4 + 7 + 8) / 3)
    assert current["GoalsForAvg5_A_pre"] == pytest.approx((1 + 3 + 5 + 6 + 9) / 5)
    assert current["GoalsAgainstAvg5_A_pre"] == pytest.approx((0 + 2 + 4 + 7 + 8) / 5)
    assert current["GoalDiffAvg5_A_pre"] == pytest.approx(0.6)


def test_current_and_same_date_statistics_are_excluded() -> None:
    frame = rolling_frame(
        [
            ("2023-08-01", 2023, "E0", "A", "B", 8, 1),
            ("2023-08-01", 2023, "E0", "A", "C", 2, 9),
        ]
    )
    changed = frame.copy()
    changed.loc[0, ["FTHome", "HomeShots", "HomeTarget"]] = [99, 99, 99]

    result = add_rolling_features(frame)
    changed_result = add_rolling_features(changed)

    assert result.loc[:, list(ROLLING_FEATURES)].isna().all().all()
    pd.testing.assert_frame_equal(
        result.loc[:, list(ROLLING_FEATURES)],
        changed_result.loc[:, list(ROLLING_FEATURES)],
    )


def test_histories_are_isolated_by_division_season_and_team() -> None:
    frame = rolling_frame(
        [
            ("2022-08-01", 2022, "E0", "A", "B", 4, 0),
            ("2022-08-01", 2022, "D1", "A", "B", 1, 3),
            ("2023-08-01", 2023, "E0", "A", "C", 2, 1),
        ]
    )

    result = add_rolling_features(frame)

    assert result.loc[2, list(ROLLING_FEATURES)].isna().all()


def test_shuffle_and_future_rows_leave_prior_rolling_features_unchanged() -> None:
    frame = rolling_frame(
        [
            ("2023-08-01", 2023, "E0", "A", "B", 1, 0),
            ("2023-08-01", 2023, "E0", "C", "D", 2, 3),
            ("2023-08-08", 2023, "E0", "A", "C", 4, 2),
            ("2023-08-15", 2023, "E0", "B", "D", 1, 1),
        ]
    )
    full = add_rolling_features(frame)
    shuffled = add_rolling_features(frame.sample(frac=1.0, random_state=9))
    truncated = add_rolling_features(frame.loc[frame["MatchDate"] <= "2023-08-08"])

    pd.testing.assert_frame_equal(aligned(full), aligned(shuffled))
    pd.testing.assert_frame_equal(
        aligned(full.loc[full["MatchDate"] <= "2023-08-08"]),
        aligned(truncated),
    )


def test_missing_prior_stat_is_skipped_and_all_missing_window_stays_missing() -> None:
    frame = rolling_frame(
        [
            ("2023-08-01", 2023, "E0", "A", "B", 1, 0),
            ("2023-08-08", 2023, "E0", "A", "C", 2, 1),
            ("2023-08-15", 2023, "E0", "A", "D", 3, 2),
        ]
    ).astype({"HomeShots": "Float64", "HomeRed": "Float64"})
    frame.loc[0, "HomeShots"] = pd.NA
    frame.loc[:1, "HomeRed"] = pd.NA

    result = add_rolling_features(frame)

    assert pd.isna(result.loc[1, "ShotsForAvg5_H_pre"])
    assert result.loc[2, "ShotsForAvg5_H_pre"] == 2.0
    assert pd.isna(result.loc[2, "RedForAvg5_H_pre"])
