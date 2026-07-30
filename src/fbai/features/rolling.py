"""Date-batched rolling match statistics from each team's perspective."""

from __future__ import annotations

from collections import defaultdict
from typing import cast

import numpy as np
import pandas as pd

from fbai.features.schema import ROLLING_BASE_FEATURES, ROLLING_FEATURES

# (display base, canonical home field, canonical away field, windows)
STAT_SPECS: tuple[tuple[str, str, str, tuple[int, ...]], ...] = (
    ("Goals", "FTHome", "FTAway", (3, 5)),
    ("Shots", "HomeShots", "AwayShots", (5,)),
    ("Target", "HomeTarget", "AwayTarget", (5,)),
    ("Corners", "HomeCorners", "AwayCorners", (5,)),
    ("Fouls", "HomeFouls", "AwayFouls", (5,)),
    ("Yellow", "HomeYellow", "AwayYellow", (5,)),
    ("Red", "HomeRed", "AwayRed", (5,)),
)

_SORT_KEYS = ["MatchDate", "Division", "HomeTeam", "AwayTeam"]


class RollingBuildError(ValueError):
    """Raised when rolling state cannot be built from canonical rows."""


def _intermediate_names() -> tuple[str, ...]:
    names: list[str] = []
    for base, _home, _away, windows in STAT_SPECS:
        for window in windows:
            names.extend((f"{base}ForAvg{window}", f"{base}AgainstAvg{window}"))
            if base == "Goals":
                names.append(f"GoalDiffAvg{window}")
    return tuple(names)


_INTERMEDIATE = _intermediate_names()
assert _INTERMEDIATE == ROLLING_BASE_FEATURES


def _rolling_mean(values: list[float], window: int) -> float:
    window_values = values[-window:]
    present = [value for value in window_values if not pd.isna(value)]
    if not present:
        return float("nan")
    return float(np.mean(present))


def _as_float(value: object) -> float:
    return float("nan") if pd.isna(value) else float(cast(float | int, value))


def _team_pre_features(
    histories: defaultdict[tuple[object, object, object, str], list[float]],
    key: tuple[object, object, object],
) -> dict[str, float]:
    features: dict[str, float] = {}
    for base, _home, _away, windows in STAT_SPECS:
        for_values = histories[(*key, f"{base}_for")]
        against_values = histories[(*key, f"{base}_against")]
        for window in windows:
            for_average = _rolling_mean(for_values, window)
            against_average = _rolling_mean(against_values, window)
            features[f"{base}ForAvg{window}"] = for_average
            features[f"{base}AgainstAvg{window}"] = against_average
            if base == "Goals":
                features[f"GoalDiffAvg{window}"] = for_average - against_average
    return features


def add_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach 36 all-venue, prior-match rolling features.

    Histories are isolated by ``(Division, SeasonStartYear, team)``. Windows
    contain the last three or five completed team appearances, skip missing
    raw values, and require at least one valid prior observation. State from a
    date is appended only after every row on that date has been featurized.
    """

    required = {
        "Division",
        "MatchDate",
        "SeasonStartYear",
        "HomeTeam",
        "AwayTeam",
        *(home for _base, home, _away, _windows in STAT_SPECS),
        *(away for _base, _home, away, _windows in STAT_SPECS),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RollingBuildError(f"Missing rolling source columns: {', '.join(missing)}")

    out = frame.copy(deep=True)
    values = {column: pd.Series(index=out.index, dtype="float64") for column in ROLLING_FEATURES}
    histories: defaultdict[tuple[object, object, object, str], list[float]] = defaultdict(list)
    ordered = out.sort_values(_SORT_KEYS, kind="mergesort")

    for _date, date_rows in ordered.groupby("MatchDate", sort=True):
        pending: list[tuple[tuple[object, object, object], dict[str, float]]] = []
        for row in date_rows.itertuples(index=True):
            home_key = (row.Division, row.SeasonStartYear, row.HomeTeam)
            away_key = (row.Division, row.SeasonStartYear, row.AwayTeam)
            home_features = _team_pre_features(histories, home_key)
            away_features = _team_pre_features(histories, away_key)
            for name in ROLLING_BASE_FEATURES:
                values[f"{name}_H_pre"].at[row.Index] = home_features[name]
                values[f"{name}_A_pre"].at[row.Index] = away_features[name]

            home_raw: dict[str, float] = {}
            away_raw: dict[str, float] = {}
            for base, home_column, away_column, _windows in STAT_SPECS:
                home_value = _as_float(getattr(row, home_column))
                away_value = _as_float(getattr(row, away_column))
                home_raw[f"{base}_for"] = home_value
                home_raw[f"{base}_against"] = away_value
                away_raw[f"{base}_for"] = away_value
                away_raw[f"{base}_against"] = home_value
            pending.append((home_key, home_raw))
            pending.append((away_key, away_raw))

        for key, raw_values in pending:
            for name, value in raw_values.items():
                histories[(*key, name)].append(value)

    for column in ROLLING_FEATURES:
        out[column] = values[column]
    return out
