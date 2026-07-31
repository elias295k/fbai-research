"""Strictly-prior player-availability feature construction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fbai.core.leakage import NATURAL_KEY, assert_natural_key_valid
from fbai.core.splits import STABLE_ORDER_KEY
from fbai.research.player_availability.alignment import AvailabilitySourceFrames
from fbai.research.player_availability.schema import (
    PRIOR_AVAILABILITY_FEATURES,
    PRIOR_BASE_FEATURES,
)


class AvailabilityFeatureError(ValueError):
    """Raised when strict-prior availability construction is unsafe."""


@dataclass(frozen=True, slots=True)
class AvailabilityFeatureBuild:
    """Aggregate-only metadata and the anonymous per-match feature frame."""

    features: pd.DataFrame
    match_rows: int
    feature_count: int
    same_date_batches: bool
    season_resets: bool


def _appearance_maps(
    appearances: pd.DataFrame,
) -> dict[tuple[int, int], dict[int, float]]:
    maps: dict[tuple[int, int], dict[int, float]] = {}
    for key, group in appearances.groupby(["game_id", "player_club_id"], sort=False, observed=True):
        minutes = group.groupby("player_id", sort=True)["minutes_played"].sum(min_count=1)
        maps[(int(key[0]), int(key[1]))] = {
            int(player_id): float(value) for player_id, value in minutes.items()
        }
    return maps


def _starter_maps(lineups: pd.DataFrame) -> dict[tuple[int, int], set[int]]:
    starters: dict[tuple[int, int], set[int]] = defaultdict(set)
    selected = lineups.loc[lineups["type"].eq("starting_lineup")]
    for row in selected.itertuples(index=False):
        starters[(int(row.game_id), int(row.club_id))].add(int(row.player_id))
    return starters


def _valuation_lookup(
    valuations: pd.DataFrame,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for player_id, group in valuations.groupby("player_id", sort=False, observed=True):
        lookup[int(player_id)] = (
            group["date"].to_numpy(dtype="datetime64[ns]"),
            group["market_value_in_eur"].to_numpy(dtype=float),
        )
    return lookup


def _latest_value(
    lookup: dict[int, tuple[np.ndarray, np.ndarray]],
    player_id: int,
    match_date: pd.Timestamp,
) -> float:
    item = lookup.get(player_id)
    if item is None:
        return float("nan")
    dates, values = item
    position = int(np.searchsorted(dates, np.datetime64(match_date, "ns"), side="left")) - 1
    return float(values[position]) if position >= 0 else float("nan")


def _value_summary(
    lookup: dict[int, tuple[np.ndarray, np.ndarray]],
    players: list[int] | set[int],
    match_date: pd.Timestamp,
) -> tuple[float, float, float]:
    player_ids = [int(player_id) for player_id in players]
    if not player_ids:
        return float("nan"), float("nan"), float("nan")
    values = np.asarray(
        [_latest_value(lookup, player_id, match_date) for player_id in player_ids],
        dtype=float,
    )
    known = np.isfinite(values)
    if not known.any():
        return float("nan"), float("nan"), 0.0
    return (
        float(values[known].sum()),
        float(values[known].mean()),
        float(known.mean()),
    )


def _combine_minutes(matches: list[dict[str, Any]]) -> dict[int, float]:
    minutes: dict[int, float] = {}
    for match in matches:
        for player_id, value in match["appearances"].items():
            resolved = int(player_id)
            minutes[resolved] = minutes.get(resolved, 0.0) + float(value)
    return minutes


def _most_common(minutes: dict[int, float], count: int) -> list[tuple[int, float]]:
    return sorted(minutes.items(), key=lambda item: item[1], reverse=True)[:count]


def _share_top(minutes: dict[int, float], count: int) -> float:
    total = float(sum(minutes.values()))
    if total <= 0.0:
        return float("nan")
    return float(sum(value for _, value in _most_common(minutes, count)) / total)


def _starter_overlap(first: set[int], second: set[int]) -> float:
    if not first or not second:
        return float("nan")
    return float(len(first & second) / 11.0)


def _starter_jaccard(first: set[int], second: set[int]) -> float:
    if not first or not second:
        return float("nan")
    return float(len(first & second) / len(first | second))


def _side_features(
    record: dict[str, Any],
    history: list[dict[str, Any]],
    valuations: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    match_date = pd.Timestamp(record["MatchDate"])
    season = int(record["SeasonStartYear"])
    season_history = [match for match in history if int(match["SeasonStartYear"]) == season]
    output = {name: float("nan") for name in PRIOR_BASE_FEATURES}
    if not season_history:
        return output

    for days in (7, 14, 21, 30):
        start = match_date - pd.Timedelta(days=days)
        recent = [match for match in season_history if match["MatchDate"] >= start]
        output[f"minutes_{days}d"] = float(
            sum(sum(match["appearances"].values()) for match in recent)
        )
        if days == 30:
            output["matches_30d"] = float(len(recent))
    output["days_since_prev_match"] = float((match_date - season_history[-1]["MatchDate"]).days)

    last_five = season_history[-5:]
    last_ten = season_history[-10:]
    minutes_five = _combine_minutes(last_five)
    minutes_ten = _combine_minutes(last_ten)
    output["unique_players_5"] = float(len(minutes_five))
    output["unique_players_10"] = float(len(minutes_ten))
    output["top5_minutes_share_5"] = _share_top(minutes_five, 5)
    output["top11_minutes_share_5"] = _share_top(minutes_five, 11)
    output["top11_minutes_share_10"] = _share_top(minutes_ten, 11)
    output["avg_player_minutes_5"] = (
        float(sum(minutes_five.values()) / len(minutes_five)) if minutes_five else float("nan")
    )

    starters_five = [match["starters"] for match in last_five if match["starters"]]
    output["unique_starters_5"] = (
        float(len(set().union(*starters_five))) if starters_five else float("nan")
    )
    if len(starters_five) >= 2:
        output["starter_overlap_last"] = _starter_overlap(starters_five[-1], starters_five[-2])
        output["starter_jaccard_last"] = _starter_jaccard(starters_five[-1], starters_five[-2])
        output["rotation_index_last"] = 1.0 - output["starter_overlap_last"]
        changes = [
            11.0 - 11.0 * _starter_overlap(current, previous)
            for previous, current in zip(starters_five[:-1], starters_five[1:], strict=True)
        ]
        output["avg_starter_changes_5"] = float(np.mean(changes))

    players_five = list(minutes_five)
    top_eleven = [player_id for player_id, _ in _most_common(minutes_five, 11)]
    last_starters = list(season_history[-1]["starters"])
    output["recent_players_value_sum_5"] = _value_summary(valuations, players_five, match_date)[0]
    output["recent_top11_value_sum_5"] = _value_summary(valuations, top_eleven, match_date)[0]
    output["last_starters_value_sum"] = _value_summary(valuations, last_starters, match_date)[0]
    output["recent_value_coverage_5"] = _value_summary(valuations, players_five, match_date)[2]
    return output


def _side_records(
    fixtures: pd.DataFrame,
    appearances: dict[tuple[int, int], dict[int, float]],
    starters: dict[tuple[int, int], set[int]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for position, row in fixtures.iterrows():
        for side, team_column, club_column in (
            ("H", "HomeTeam", "home_club_id"),
            ("A", "AwayTeam", "away_club_id"),
        ):
            game_id = int(row["game_id"])
            club_id = int(row[club_column])
            key = (game_id, club_id)
            records.append(
                {
                    "fixture_position": int(position),
                    "side": side,
                    "Division": str(row["Division"]),
                    "SeasonStartYear": int(row["SeasonStartYear"]),
                    "MatchDate": pd.Timestamp(row["MatchDate"]),
                    "team": str(row[team_column]),
                    "game_id": game_id,
                    "appearances": appearances.get(key, {}),
                    "starters": starters.get(key, set()),
                }
            )
    return records


def build_prior_availability_features(
    sources: AvailabilitySourceFrames,
) -> AvailabilityFeatureBuild:
    """Build the 63 frozen features without observing any target-date row."""

    fixtures = sources.fixtures.sort_values(list(STABLE_ORDER_KEY), kind="mergesort").reset_index(
        drop=True
    )
    appearances = _appearance_maps(sources.appearances)
    starters = _starter_maps(sources.lineups)
    valuations = _valuation_lookup(sources.valuations)
    records = _side_records(fixtures, appearances, starters)
    records.sort(
        key=lambda record: (
            record["MatchDate"],
            record["Division"],
            record["team"],
            record["game_id"],
            record["side"],
        )
    )

    side_features: dict[tuple[int, str], dict[str, float]] = {}
    histories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    position = 0
    while position < len(records):
        match_date = records[position]["MatchDate"]
        end = position
        while end < len(records) and records[end]["MatchDate"] == match_date:
            end += 1
        batch = records[position:end]
        for record in batch:
            team_key = (record["Division"], record["team"])
            side_features[(record["fixture_position"], record["side"])] = _side_features(
                record, histories[team_key], valuations
            )
        for record in batch:
            histories[(record["Division"], record["team"])].append(record)
        position = end

    rows: list[dict[str, object]] = []
    for fixture_position, row in fixtures.iterrows():
        output: dict[str, object] = {
            column: row[column] for column in (*NATURAL_KEY, "SeasonStartYear", "game_id")
        }
        home = side_features[(fixture_position, "H")]
        away = side_features[(fixture_position, "A")]
        for feature in PRIOR_BASE_FEATURES:
            home_value = home[feature]
            away_value = away[feature]
            output[f"avail_H_{feature}_pre"] = home_value
            output[f"avail_A_{feature}_pre"] = away_value
            output[f"avail_diff_{feature}_pre"] = (
                float(home_value - away_value)
                if np.isfinite(home_value) and np.isfinite(away_value)
                else float("nan")
            )
        rows.append(output)

    columns = [
        *NATURAL_KEY,
        "SeasonStartYear",
        "game_id",
        *PRIOR_AVAILABILITY_FEATURES,
    ]
    features = pd.DataFrame(rows).loc[:, columns]
    assert_natural_key_valid(features, key_columns=NATURAL_KEY)
    numeric = features.loc[:, list(PRIOR_AVAILABILITY_FEATURES)].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise AvailabilityFeatureError("availability features contain infinite values")
    if any(column in {"player_id", "player_name"} for column in features):
        raise AvailabilityFeatureError("availability output exposed a player identifier")
    return AvailabilityFeatureBuild(
        features=features,
        match_rows=len(features),
        feature_count=len(PRIOR_AVAILABILITY_FEATURES),
        same_date_batches=True,
        season_resets=True,
    )
