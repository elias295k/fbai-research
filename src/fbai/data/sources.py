"""Supported public CSV sources and deterministic source paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"

SUPPORTED_LEAGUES: dict[str, str] = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "B1": "Belgian Pro League",
    "N1": "Eredivisie",
    "E1": "Championship",
    "P1": "Primeira Liga",
}


@dataclass(frozen=True)
class SeasonSpec:
    """One supported football-data season."""

    code: str
    start_year: int

    @property
    def label(self) -> str:
        return f"{self.start_year}-{self.start_year + 1}"


SUPPORTED_SEASONS: tuple[SeasonSpec, ...] = (
    SeasonSpec("1920", 2019),
    SeasonSpec("2021", 2020),
    SeasonSpec("2122", 2021),
    SeasonSpec("2223", 2022),
    SeasonSpec("2324", 2023),
    SeasonSpec("2425", 2024),
    SeasonSpec("2526", 2025),
)
_SEASONS_BY_CODE = {season.code: season for season in SUPPORTED_SEASONS}


class UnsupportedDivisionError(ValueError):
    """Raised when acquisition is requested for an unsupported division."""


class UnsupportedSeasonError(ValueError):
    """Raised when acquisition is requested for an unsupported source season."""


def validate_division(division: str) -> str:
    """Return a supported division code or raise."""

    if division not in SUPPORTED_LEAGUES:
        supported = ", ".join(SUPPORTED_LEAGUES)
        raise UnsupportedDivisionError(
            f"Unsupported division {division!r}; supported divisions: {supported}"
        )
    return division


def season_spec(season_code: str) -> SeasonSpec:
    """Return configuration for an exact football-data season code."""

    try:
        return _SEASONS_BY_CODE[season_code]
    except KeyError as exc:
        supported = ", ".join(_SEASONS_BY_CODE)
        raise UnsupportedSeasonError(
            f"Unsupported season code {season_code!r}; supported codes: {supported}"
        ) from exc


def season_start_year(season_code: str) -> int:
    """Return the configured four-digit start year for a source season code."""

    return season_spec(season_code).start_year


def season_label(season_code: str) -> str:
    """Return the configured long season label."""

    return season_spec(season_code).label


def build_source_url(season_code: str, division: str) -> str:
    """Build a deterministic public CSV URL after validating both dimensions."""

    season_spec(season_code)
    validate_division(division)
    return f"{FOOTBALL_DATA_BASE_URL}/{season_code}/{division}.csv"


def source_csv_path(root: Path, season_code: str, division: str) -> Path:
    """Return the deterministic local path for one downloaded source CSV."""

    season_spec(season_code)
    validate_division(division)
    return Path(root) / season_code / f"{division}.csv"
