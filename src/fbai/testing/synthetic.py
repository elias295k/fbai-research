"""Deterministic, invented football fixtures for tests and examples."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from fbai.core.splits import sort_matches

SYNTHETIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "HomeRating_pre",
    "AwayRating_pre",
    "RatingDiff_pre",
    "HomeForm_pre",
    "AwayForm_pre",
)


def _round_robin_rounds(teams: Sequence[str]) -> list[list[tuple[str, str]]]:
    rotation: list[str | None] = list(teams)
    if len(rotation) % 2:
        rotation.append(None)
    rounds: list[list[tuple[str, str]]] = []
    for round_index in range(len(rotation) - 1):
        fixtures: list[tuple[str, str]] = []
        half = len(rotation) // 2
        for pair_index in range(half):
            first = rotation[pair_index]
            second = rotation[-(pair_index + 1)]
            if first is None or second is None:
                continue
            if (round_index + pair_index) % 2:
                fixtures.append((second, first))
            else:
                fixtures.append((first, second))
        rounds.append(fixtures)
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return rounds


def make_synthetic_fixtures(
    *,
    seed: int = 42,
    season_start_years: Sequence[int] = (2021, 2022, 2023, 2024, 2025),
    divisions: Sequence[str] = ("SYN1", "SYN2"),
    teams_per_division: int = 6,
    include_forbidden_same_match: bool = False,
) -> pd.DataFrame:
    """Create a reproducible fictional fixture table with no external data."""

    if teams_per_division < 2:
        raise ValueError("teams_per_division must be at least two")
    if not season_start_years:
        raise ValueError("At least one season is required")
    if not divisions:
        raise ValueError("At least one division is required")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for season in season_start_years:
        for division_index, division in enumerate(divisions):
            teams = [
                f"{division} Synthetic Club {team_number:02d}"
                for team_number in range(1, teams_per_division + 1)
            ]
            strengths = {team: float(rng.normal(loc=0.0, scale=0.7)) for team in teams}
            pairings = list(combinations(teams, 2))
            for round_index, (first_team, second_team) in enumerate(pairings):
                reverse = (round_index + season + division_index) % 2 == 1
                home_team, away_team = (
                    (second_team, first_team) if reverse else (first_team, second_team)
                )
                match_date = pd.Timestamp(season, 8, 1) + pd.Timedelta(
                    days=7 * (round_index // max(1, teams_per_division // 2))
                )
                home_rating = strengths[home_team] + 0.25
                away_rating = strengths[away_team]
                home_form = float(rng.normal(loc=0.0, scale=0.5))
                away_form = float(rng.normal(loc=0.0, scale=0.5))
                advantage = home_rating - away_rating + 0.25 * (home_form - away_form)
                logits = np.array(
                    [0.85 * advantage, 0.15 - 0.2 * abs(advantage), -0.85 * advantage]
                )
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                result = str(rng.choice(np.array(["H", "D", "A"]), p=probabilities))
                row: dict[str, object] = {
                    "Division": division,
                    "MatchDate": match_date,
                    "HomeTeam": home_team,
                    "AwayTeam": away_team,
                    "SeasonStartYear": season,
                    "HomeRating_pre": home_rating,
                    "AwayRating_pre": away_rating,
                    "RatingDiff_pre": home_rating - away_rating,
                    "HomeForm_pre": home_form,
                    "AwayForm_pre": away_form,
                    "FTR": result,
                }
                if include_forbidden_same_match:
                    home_goals = int(rng.poisson(1.2 + max(advantage, 0.0)))
                    away_goals = int(rng.poisson(1.0 + max(-advantage, 0.0)))
                    row.update(
                        {
                            "FTHG": home_goals,
                            "FTAG": away_goals,
                            "HomeShots": int(rng.integers(4, 20)),
                            "AwayShots": int(rng.integers(4, 20)),
                        }
                    )
                rows.append(row)

    return sort_matches(pd.DataFrame(rows)).reset_index(drop=True)


def make_synthetic_raw_matches(
    *,
    seed: int = 42,
    season_start_years: Sequence[int] = (2021, 2022, 2023, 2024, 2025),
    divisions: Sequence[str] = ("SYN1", "SYN2"),
    teams_per_division: int = 6,
) -> pd.DataFrame:
    """Create complete fictional source-shaped rows for the Phase 2A loader."""

    if teams_per_division < 2:
        raise ValueError("teams_per_division must be at least two")
    if not season_start_years:
        raise ValueError("At least one season is required")
    if not divisions:
        raise ValueError("At least one division is required")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for season in season_start_years:
        for division in divisions:
            teams = [
                f"{division} Synthetic Club {team_number:02d}"
                for team_number in range(1, teams_per_division + 1)
            ]
            strengths = {team: float(rng.normal(loc=0.0, scale=0.55)) for team in teams}
            first_leg = _round_robin_rounds(teams)
            schedule = [
                fixtures if leg == 0 else [(away, home) for home, away in fixtures]
                for leg in range(2)
                for fixtures in first_leg
            ]
            for round_index, fixtures in enumerate(schedule):
                match_date = pd.Timestamp(season, 8, 1) + pd.Timedelta(days=7 * round_index)
                for home_team, away_team in fixtures:
                    strength_delta = strengths[home_team] - strengths[away_team]
                    home_rate = float(np.exp(np.clip(0.25 + 0.45 * strength_delta, -1, 1)))
                    away_rate = float(np.exp(np.clip(-0.05 - 0.4 * strength_delta, -1, 1)))
                    home_goals = int(rng.poisson(home_rate))
                    away_goals = int(rng.poisson(away_rate))
                    result = "H" if home_goals > away_goals else "A"
                    if home_goals == away_goals:
                        result = "D"

                    home_target = home_goals + int(rng.integers(0, 6))
                    away_target = away_goals + int(rng.integers(0, 6))
                    home_shots = home_target + int(rng.integers(1, 11))
                    away_shots = away_target + int(rng.integers(1, 11))
                    rows.append(
                        {
                            "Div": division,
                            "Date": match_date.strftime("%d/%m/%Y"),
                            "Season": f"{season}-{season + 1}",
                            "HomeTeam": home_team,
                            "AwayTeam": away_team,
                            "FTHG": home_goals,
                            "FTAG": away_goals,
                            "FTR": result,
                            "HS": home_shots,
                            "AS": away_shots,
                            "HST": home_target,
                            "AST": away_target,
                            "HC": int(rng.poisson(5.0)),
                            "AC": int(rng.poisson(4.5)),
                            "HF": int(rng.integers(6, 18)),
                            "AF": int(rng.integers(6, 18)),
                            "HY": int(rng.integers(0, 6)),
                            "AY": int(rng.integers(0, 6)),
                            "HR": int(rng.binomial(1, 0.06)),
                            "AR": int(rng.binomial(1, 0.06)),
                        }
                    )
    return pd.DataFrame(rows)


def make_synthetic_canonical_matches(
    *,
    seed: int = 42,
    season_start_years: Sequence[int] = (2021, 2022, 2023, 2024, 2025),
    divisions: Sequence[str] = ("SYN1", "SYN2"),
    teams_per_division: int = 6,
) -> pd.DataFrame:
    """Generate source-shaped rows and pass them through the public loader."""

    from fbai.data.loader import canonicalize_source_frame

    raw = make_synthetic_raw_matches(
        seed=seed,
        season_start_years=season_start_years,
        divisions=divisions,
        teams_per_division=teams_per_division,
    )
    return canonicalize_source_frame(raw)
