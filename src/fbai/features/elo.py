"""Date-batched, walk-forward Elo features rebuilt from canonical outcomes."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from fbai.features.schema import ELO_FEATURES

DEFAULT_START_ELO = 1500.0
DEFAULT_K_FACTOR = 40.0
DEFAULT_HOME_ADVANTAGE = 60.0
DEFAULT_SEASON_REVERSION = 0.30

_RESULT_SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}
_SORT_KEYS = ["MatchDate", "HomeTeam", "AwayTeam"]


class EloBuildError(ValueError):
    """Raised when Elo state cannot be built deterministically."""


def expected_home_score(
    home_elo: float,
    away_elo: float,
    *,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> float:
    """Return the standard Elo expected score for the home side."""

    return float(1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + home_advantage)) / 400.0)))


def add_elo_features(
    frame: pd.DataFrame,
    *,
    start_elo: float = DEFAULT_START_ELO,
    k_factor: float = DEFAULT_K_FACTOR,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    season_reversion: float = DEFAULT_SEASON_REVERSION,
) -> pd.DataFrame:
    """Attach three pre-match Elo columns without exposing same-date results.

    Ratings are independent per division. At a season transition every known
    rating moves ``season_reversion`` of the distance back toward ``start_elo``.
    All rows on one date read the same pre-date state; their deltas are applied
    together only after the complete date batch has been scored.
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
        raise EloBuildError(f"Missing Elo source columns: {', '.join(missing)}")
    if not 0.0 <= season_reversion <= 1.0:
        raise EloBuildError("season_reversion must lie in [0, 1]")
    if k_factor < 0.0:
        raise EloBuildError("k_factor must be non-negative")

    out = frame.copy(deep=True)
    home_pre = pd.Series(index=out.index, dtype="float64")
    away_pre = pd.Series(index=out.index, dtype="float64")

    for _division, indexes in out.groupby("Division", sort=True).groups.items():
        division_rows = out.loc[indexes].sort_values(_SORT_KEYS, kind="mergesort")
        ratings: dict[str, float] = {}
        current_season: object | None = None

        for _date, date_rows in division_rows.groupby("MatchDate", sort=True):
            seasons = date_rows["SeasonStartYear"].drop_duplicates().tolist()
            if len(seasons) != 1:
                raise EloBuildError(
                    "A division/date batch must belong to exactly one SeasonStartYear"
                )
            season = seasons[0]
            if current_season is not None and season != current_season:
                ratings = {
                    team: rating * (1.0 - season_reversion) + start_elo * season_reversion
                    for team, rating in ratings.items()
                }
            current_season = season

            deltas: defaultdict[str, float] = defaultdict(float)
            for row in date_rows.itertuples(index=True):
                home_rating = ratings.get(row.HomeTeam, start_elo)
                away_rating = ratings.get(row.AwayTeam, start_elo)
                home_pre.at[row.Index] = home_rating
                away_pre.at[row.Index] = away_rating

                try:
                    actual_home = _RESULT_SCORE[row.FTR]
                except KeyError as exc:
                    raise EloBuildError(f"Invalid FTR value for Elo: {row.FTR!r}") from exc
                expected_home = expected_home_score(
                    home_rating,
                    away_rating,
                    home_advantage=home_advantage,
                )
                change = k_factor * (actual_home - expected_home)
                deltas[row.HomeTeam] += change
                deltas[row.AwayTeam] -= change

            for team, delta in deltas.items():
                ratings[team] = ratings.get(team, start_elo) + delta

    out[ELO_FEATURES[0]] = home_pre
    out[ELO_FEATURES[1]] = away_pre
    out[ELO_FEATURES[2]] = home_pre - away_pre
    return out


# Source-private name retained for focused arithmetic tests.
_expected_home = expected_home_score
