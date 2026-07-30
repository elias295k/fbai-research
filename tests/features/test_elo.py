from __future__ import annotations

import pandas as pd
import pytest

from fbai.features.elo import (
    DEFAULT_HOME_ADVANTAGE,
    DEFAULT_K_FACTOR,
    DEFAULT_SEASON_REVERSION,
    DEFAULT_START_ELO,
    add_elo_features,
    expected_home_score,
)
from fbai.features.schema import ELO_FEATURES

KEY = ["MatchDate", "Division", "HomeTeam", "AwayTeam"]


def elo_frame(rows: list[tuple[str, int, str, str, str, str]]) -> pd.DataFrame:
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
    return frame.sort_values(KEY).reset_index(drop=True).loc[:, [*KEY, *ELO_FEATURES]]


def test_initial_ratings_and_pre_match_update_arithmetic() -> None:
    frame = elo_frame(
        [
            ("2023-08-01", 2023, "E0", "A", "B", "H"),
            ("2023-08-08", 2023, "E0", "A", "C", "D"),
        ]
    )

    result = add_elo_features(frame)
    expected = expected_home_score(1500.0, 1500.0)
    rating_after_win = DEFAULT_START_ELO + DEFAULT_K_FACTOR * (1.0 - expected)

    assert result.loc[0, "HomeElo_pre"] == DEFAULT_START_ELO
    assert result.loc[0, "AwayElo_pre"] == DEFAULT_START_ELO
    assert result.loc[1, "HomeElo_pre"] == pytest.approx(rating_after_win)
    assert result.loc[1, "AwayElo_pre"] == DEFAULT_START_ELO
    assert result.loc[1, "EloDiff_pre"] == pytest.approx(rating_after_win - 1500.0)


def test_home_advantage_changes_expected_score_not_stored_rating() -> None:
    assert expected_home_score(1500.0, 1500.0) > 0.5
    assert expected_home_score(1500.0, 1500.0, home_advantage=0.0) == 0.5
    assert DEFAULT_HOME_ADVANTAGE == 60.0


def test_season_transition_reverts_known_ratings_toward_start() -> None:
    frame = elo_frame(
        [
            ("2022-08-01", 2022, "E0", "A", "B", "H"),
            ("2023-08-01", 2023, "E0", "A", "C", "D"),
        ]
    )
    result = add_elo_features(frame)
    post_win = 1500.0 + DEFAULT_K_FACTOR * (1.0 - expected_home_score(1500.0, 1500.0))
    expected_reverted = (
        post_win * (1.0 - DEFAULT_SEASON_REVERSION) + DEFAULT_START_ELO * DEFAULT_SEASON_REVERSION
    )

    assert result.loc[1, "HomeElo_pre"] == pytest.approx(expected_reverted)


def test_same_date_matches_share_pre_date_state() -> None:
    frame = elo_frame(
        [
            ("2023-08-01", 2023, "E0", "A", "B", "H"),
            ("2023-08-01", 2023, "E0", "A", "C", "A"),
            ("2023-08-08", 2023, "E0", "A", "D", "D"),
        ]
    )
    result = add_elo_features(frame)

    assert result.loc[0, "HomeElo_pre"] == 1500.0
    assert result.loc[1, "HomeElo_pre"] == 1500.0


def test_shuffle_division_isolation_and_future_truncation() -> None:
    frame = elo_frame(
        [
            ("2023-08-01", 2023, "E0", "A", "B", "H"),
            ("2023-08-01", 2023, "D1", "A", "B", "A"),
            ("2023-08-08", 2023, "E0", "A", "C", "D"),
            ("2023-08-08", 2023, "D1", "A", "C", "D"),
            ("2023-08-15", 2023, "E0", "C", "B", "A"),
        ]
    )
    full = add_elo_features(frame)
    shuffled = add_elo_features(frame.sample(frac=1.0, random_state=7))
    truncated = add_elo_features(frame.loc[frame["MatchDate"] <= "2023-08-08"])

    pd.testing.assert_frame_equal(aligned(full), aligned(shuffled))
    pd.testing.assert_frame_equal(
        aligned(full.loc[full["MatchDate"] <= "2023-08-08"]),
        aligned(truncated),
    )
    e0 = full.loc[(full["Division"] == "E0") & (full["MatchDate"] == "2023-08-08")]
    d1 = full.loc[(full["Division"] == "D1") & (full["MatchDate"] == "2023-08-08")]
    assert e0["HomeElo_pre"].iloc[0] > 1500.0
    assert d1["HomeElo_pre"].iloc[0] < 1500.0
