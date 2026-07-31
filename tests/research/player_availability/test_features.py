from __future__ import annotations

import pandas as pd
import pytest

from fbai.core.leakage import NATURAL_KEY
from fbai.research.player_availability.alignment import prepare_availability_sources
from fbai.research.player_availability.features import build_prior_availability_features
from fbai.research.player_availability.schema import PRIOR_AVAILABILITY_FEATURES


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixtures = pd.DataFrame(
        [
            ("E0", 2023, "2023-08-01", "A", "B", 1, 10, 20),
            ("E0", 2023, "2023-08-08", "C", "A", 2, 30, 10),
            ("E0", 2023, "2023-08-15", "A", "B", 3, 10, 20),
            ("E0", 2023, "2023-08-15", "A", "C", 4, 10, 30),
            ("E0", 2023, "2023-08-22", "B", "A", 5, 20, 10),
            ("E0", 2024, "2024-08-01", "A", "B", 6, 10, 20),
            ("F1", 2023, "2023-08-22", "A", "B", 7, 110, 120),
        ],
        columns=(
            "Division",
            "SeasonStartYear",
            "MatchDate",
            "HomeTeam",
            "AwayTeam",
            "game_id",
            "home_club_id",
            "away_club_id",
        ),
    )
    appearances = pd.DataFrame(
        [
            {
                "game_id": int(row.game_id),
                "player_id": club * 100 + offset,
                "player_club_id": club,
                "minutes_played": 90 - offset,
            }
            for row in fixtures.itertuples(index=False)
            for club in (int(row.home_club_id), int(row.away_club_id))
            for offset in range(1, 4)
        ]
    )
    lineups = appearances.loc[:, ["game_id", "player_id", "player_club_id"]].rename(
        columns={"player_club_id": "club_id"}
    )
    lineups["type"] = "starting_lineup"
    player_ids = sorted(set(appearances["player_id"]))
    valuations = pd.DataFrame(
        [
            {
                "player_id": player_id,
                "date": date,
                "market_value_in_eur": value,
            }
            for player_id in player_ids
            for date, value in (("2023-01-01", 1_000_000), ("2023-08-15", 9_000_000))
        ]
    )
    return fixtures, appearances, lineups, valuations


def _build(
    fixtures: pd.DataFrame,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
    valuations: pd.DataFrame,
) -> pd.DataFrame:
    sources = prepare_availability_sources(fixtures, appearances, lineups, valuations)
    return build_prior_availability_features(sources).features


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index(list(NATURAL_KEY)).sort_index()


def test_exact_prior_definitions_and_strict_valuation_timestamp() -> None:
    frame = _build(*_inputs())
    target = frame.loc[frame["game_id"].eq(3)].iloc[0]

    assert tuple(frame.columns[-63:]) == PRIOR_AVAILABILITY_FEATURES
    assert target["avail_H_matches_30d_pre"] == 2.0
    assert target["avail_H_minutes_7d_pre"] == pytest.approx(264.0)
    assert target["avail_H_unique_players_5_pre"] == 3.0
    assert target["avail_H_recent_players_value_sum_5_pre"] == 3_000_000.0


def test_current_same_date_and_future_rows_cannot_change_target_features() -> None:
    fixtures, appearances, lineups, valuations = _inputs()
    original = _build(fixtures, appearances, lineups, valuations)
    changed = appearances.copy(deep=True)
    changed.loc[changed["game_id"].isin([3, 4, 5, 6, 7]), "minutes_played"] = 1
    changed_lineups = lineups.loc[~lineups["game_id"].isin([3, 4])].copy()
    rebuilt = _build(fixtures, changed, changed_lineups, valuations)

    pd.testing.assert_series_equal(
        original.loc[original["game_id"].eq(3)].iloc[0],
        rebuilt.loc[rebuilt["game_id"].eq(3)].iloc[0],
    )
    pd.testing.assert_series_equal(
        original.loc[original["game_id"].eq(4)].iloc[0],
        rebuilt.loc[rebuilt["game_id"].eq(4)].iloc[0],
    )


def test_shuffling_is_deterministic_and_seasons_divisions_reset() -> None:
    fixtures, appearances, lineups, valuations = _inputs()
    expected = _build(fixtures, appearances, lineups, valuations)
    actual = _build(
        fixtures.sample(frac=1.0, random_state=1),
        appearances.sample(frac=1.0, random_state=2),
        lineups.sample(frac=1.0, random_state=3),
        valuations.sample(frac=1.0, random_state=4),
    )

    pd.testing.assert_frame_equal(_keyed(expected), _keyed(actual))
    resets = actual.loc[actual["game_id"].isin([6, 7]), list(PRIOR_AVAILABILITY_FEATURES)]
    assert resets.isna().all().all()
