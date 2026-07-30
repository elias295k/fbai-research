"""Date-batched rest, congestion, season-progress, and points-form features."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fbai.features.schema import CONTEXT_FEATURES

EXPECTED_TEAM_MATCHES: dict[str, int] = {
    "E0": 38,
    "SP1": 38,
    "I1": 38,
    "D1": 34,
    "F1": 34,
    "N1": 34,
    "P1": 34,
    "E1": 46,
    "B1": 34,
}

_HOME_POINTS = {"H": 3.0, "D": 1.0, "A": 0.0}
_AWAY_POINTS = {"H": 0.0, "D": 1.0, "A": 3.0}
_SORT_KEYS = ["MatchDate", "Division", "HomeTeam", "AwayTeam"]


class ContextBuildError(ValueError):
    """Raised when context state cannot be built from canonical rows."""


@dataclass
class _TeamContext:
    dates: list[pd.Timestamp] = field(default_factory=list)
    points: list[float] = field(default_factory=list)


def _mean_tail(values: list[float], window: int) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values[-window:]))


def _pre_context(
    state: _TeamContext,
    match_date: pd.Timestamp,
) -> tuple[float, int, int, float, float]:
    days_since = float((match_date - state.dates[-1]).days) if state.dates else float("nan")
    lower_bound = match_date - pd.Timedelta(days=14)
    matches_14d = sum(previous > lower_bound for previous in state.dates)
    return (
        days_since,
        matches_14d,
        len(state.dates),
        _mean_tail(state.points, 3),
        _mean_tail(state.points, 5),
    )


def add_context_features(
    frame: pd.DataFrame,
    *,
    expected_team_matches: dict[str, int] = EXPECTED_TEAM_MATCHES,
) -> pd.DataFrame:
    """Attach the 13 verified context features using prior dates only.

    Form is all-venue points per match (win=3, draw=1, loss=0). Histories reset
    for each ``(Division, SeasonStartYear, team)``. The 14-day count includes
    prior dates strictly greater than ``current_date - 14 days``.
    """

    required = {
        "Division",
        "MatchDate",
        "SeasonStartYear",
        "HomeTeam",
        "AwayTeam",
        "FTR",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ContextBuildError(f"Missing context source columns: {', '.join(missing)}")

    out = frame.copy(deep=True)
    values = {column: pd.Series(index=out.index, dtype="float64") for column in CONTEXT_FEATURES}
    states: defaultdict[tuple[object, object, object], _TeamContext] = defaultdict(_TeamContext)
    ordered = out.sort_values(_SORT_KEYS, kind="mergesort")

    for match_date, date_rows in ordered.groupby("MatchDate", sort=True):
        pending: list[tuple[tuple[object, object, object], pd.Timestamp, float]] = []
        for row in date_rows.itertuples(index=True):
            try:
                home_points = _HOME_POINTS[row.FTR]
                away_points = _AWAY_POINTS[row.FTR]
            except KeyError as exc:
                raise ContextBuildError(f"Invalid FTR value for form: {row.FTR!r}") from exc

            home_key = (row.Division, row.SeasonStartYear, row.HomeTeam)
            away_key = (row.Division, row.SeasonStartYear, row.AwayTeam)
            home = _pre_context(states[home_key], pd.Timestamp(match_date))
            away = _pre_context(states[away_key], pd.Timestamp(match_date))

            values["DaysSinceLast_H_pre"].at[row.Index] = home[0]
            values["DaysSinceLast_A_pre"].at[row.Index] = away[0]
            values["RestDiff_pre"].at[row.Index] = home[0] - away[0]
            values["Matches14d_H_pre"].at[row.Index] = home[1]
            values["Matches14d_A_pre"].at[row.Index] = away[1]
            values["TeamMatchNum_H_pre"].at[row.Index] = home[2]
            values["TeamMatchNum_A_pre"].at[row.Index] = away[2]
            expected = expected_team_matches.get(str(row.Division))
            values["SeasonProgress_pre"].at[row.Index] = (
                home[2] / expected if expected is not None else float("nan")
            )
            values["Form3Home_pre"].at[row.Index] = home[3]
            values["Form5Home_pre"].at[row.Index] = home[4]
            values["Form3Away_pre"].at[row.Index] = away[3]
            values["Form5Away_pre"].at[row.Index] = away[4]
            values["Form5Diff_pre"].at[row.Index] = home[4] - away[4]

            pending.append((home_key, pd.Timestamp(match_date), home_points))
            pending.append((away_key, pd.Timestamp(match_date), away_points))

        for key, date, points in pending:
            states[key].dates.append(date)
            states[key].points.append(points)

    for column in CONTEXT_FEATURES:
        if column.startswith(("Matches14d_", "TeamMatchNum_")):
            out[column] = values[column].astype("int64")
        else:
            out[column] = values[column]
    return out
