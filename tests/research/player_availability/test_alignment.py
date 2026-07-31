from __future__ import annotations

import pandas as pd
import pytest

from fbai.features import build_feature_table
from fbai.research.player_availability.alignment import (
    AvailabilityAlignmentError,
    align_availability_scope,
    prepare_availability_sources,
)
from fbai.testing.synthetic import make_synthetic_canonical_matches


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical = make_synthetic_canonical_matches(
        seed=73,
        season_start_years=(2020, 2021),
        divisions=("SYN1",),
        teams_per_division=4,
    )
    features = build_feature_table(canonical)
    fixtures = features.loc[
        :, ["Division", "SeasonStartYear", "MatchDate", "HomeTeam", "AwayTeam"]
    ].copy()
    fixtures["game_id"] = range(1, len(fixtures) + 1)
    teams = sorted(set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))
    clubs = {team: index + 10 for index, team in enumerate(teams)}
    fixtures["home_club_id"] = fixtures["HomeTeam"].map(clubs)
    fixtures["away_club_id"] = fixtures["AwayTeam"].map(clubs)
    appearances = pd.DataFrame(
        [
            {
                "game_id": int(row.game_id),
                "player_id": club * 100 + 1,
                "player_club_id": club,
                "minutes_played": 90,
            }
            for row in fixtures.itertuples(index=False)
            for club in (int(row.home_club_id), int(row.away_club_id))
        ]
    )
    lineups = appearances.rename(columns={"player_club_id": "club_id"}).drop(
        columns="minutes_played"
    )
    lineups["type"] = "starting_lineup"
    valuations = pd.DataFrame(
        [
            {
                "player_id": club * 100 + 1,
                "date": pd.Timestamp("2019-01-01"),
                "market_value_in_eur": 1_000_000,
            }
            for club in clubs.values()
        ]
    )
    return features, fixtures, appearances, lineups, valuations


def test_exact_alignment_reports_population_and_quality() -> None:
    features, fixtures, appearances, lineups, valuations = _frames()
    sources = prepare_availability_sources(fixtures, appearances, lineups, valuations)
    alignment = align_availability_scope(features.iloc[:-1], sources)

    assert alignment.report.base_match_rows == len(features) - 1
    assert alignment.report.supplied_match_rows == len(fixtures)
    assert alignment.report.covered_match_rows == len(features) - 1
    assert alignment.report.unmatched_supplied_matches == 1
    assert not alignment.report.coverage_filter_changes_lr52_population
    assert alignment.report.exact_key_only
    assert sources.quality.appearance_rows == len(appearances)
    assert sources.quality.invalid_minutes_rows == 0


@pytest.mark.parametrize(
    ("frame_name", "column", "value"),
    (
        ("appearances", "minutes_played", -1),
        ("lineups", "type", "captain"),
    ),
)
def test_invalid_minutes_and_roles_fail(
    frame_name: str,
    column: str,
    value: object,
) -> None:
    _, fixtures, appearances, lineups, valuations = _frames()
    frames = {"appearances": appearances, "lineups": lineups}
    changed = frames[frame_name].copy(deep=True)
    changed.loc[changed.index[0], column] = value
    with pytest.raises(AvailabilityAlignmentError):
        prepare_availability_sources(
            fixtures,
            changed if frame_name == "appearances" else appearances,
            changed if frame_name == "lineups" else lineups,
            valuations,
        )


def test_duplicate_match_game_and_player_identifiers_fail() -> None:
    _, fixtures, appearances, lineups, valuations = _frames()
    duplicate_fixture = pd.concat([fixtures, fixtures.iloc[[0]]], ignore_index=True)
    duplicate_appearance = pd.concat([appearances, appearances.iloc[[0]]], ignore_index=True)

    with pytest.raises(AvailabilityAlignmentError):
        prepare_availability_sources(duplicate_fixture, appearances, lineups, valuations)
    with pytest.raises(AvailabilityAlignmentError):
        prepare_availability_sources(fixtures, duplicate_appearance, lineups, valuations)
