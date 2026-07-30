"""Focused schema and deterministic naming for historical external xG rows."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

UNDERSTAT_XG_COLUMNS: tuple[str, ...] = (
    "Division",
    "MatchDate",
    "HomeTeam",
    "AwayTeam",
    "home_xg",
    "away_xg",
)
UNDERSTAT_XG_KEY: tuple[str, ...] = ("MatchDate", "Division", "HomeTeam", "AwayTeam")
SUPPORTED_DIVISIONS: tuple[str, ...] = ("D1", "E0", "F1", "I1", "SP1")

# Union of the executable external-signal contract and executable Understat
# conversion script. No heuristic or fuzzy mapping is applied.
TEAM_ALIASES: dict[str, str] = {
    "ac milan": "milan",
    "arminia": "bielefeld",
    "arminia bielefeld": "bielefeld",
    "athletic club": "ath bilbao",
    "atletico madrid": "ath madrid",
    "bayer leverkusen": "leverkusen",
    "borussia dortmund": "dortmund",
    "borussia mgladbach": "mgladbach",
    "borussia monchengladbach": "mgladbach",
    "celta vigo": "celta",
    "clermont foot": "clermont",
    "darmstadt 98": "darmstadt",
    "dusseldorf": "fortuna dusseldorf",
    "eintracht frankfurt": "ein frankfurt",
    "espanyol": "espanol",
    "fc cologne": "fc koln",
    "fc heidenheim": "heidenheim",
    "fortuna duesseldorf": "fortuna dusseldorf",
    "gladbach": "mgladbach",
    "greuther fuerth": "greuther furth",
    "hamburger sv": "hamburg",
    "hellas verona": "verona",
    "hertha berlin": "hertha",
    "hertha bsc": "hertha",
    "ipswich town": "ipswich",
    "koln": "fc koln",
    "leeds united": "leeds",
    "leicester city": "leicester",
    "luton town": "luton",
    "mainz 05": "mainz",
    "manchester city": "man city",
    "manchester united": "man united",
    "manchester utd": "man united",
    "newcastle united": "newcastle",
    "norwich city": "norwich",
    "nottingham forest": "nottm forest",
    "paderborn 07": "paderborn",
    "paris saint germain": "paris sg",
    "parma calcio 1913": "parma",
    "rasenballsport leipzig": "rb leipzig",
    "rayo vallecano": "vallecano",
    "real betis": "betis",
    "real sociedad": "sociedad",
    "real valladolid": "valladolid",
    "saint etienne": "st etienne",
    "sd huesca": "huesca",
    "spal 2013": "spal",
    "tottenham hotspur": "tottenham",
    "vfb stuttgart": "stuttgart",
    "west bromwich albion": "west brom",
    "west ham united": "west ham",
    "wolverhampton wanderers": "wolves",
}


class UnderstatXGSchemaError(ValueError):
    """Raised when an external xG frame violates its isolated contract."""


def normalize_team_name(name: object) -> str:
    """Apply the exact source normalization and explicit alias table."""

    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower()
    text = "".join(
        character if character.isalnum() or character.isspace() else " " if character == "-" else ""
        for character in text
    )
    text = " ".join(text.split())
    return TEAM_ALIASES.get(text, text)


@dataclass(frozen=True, slots=True)
class UnderstatXGSchemaAudit:
    """Aggregate-only external schema diagnostics."""

    supplied_rows: int
    valid_rows: int
    invalid_key_rows: int
    invalid_xg_rows: int
    invalid_xg_values: int
    unsupported_division_rows: int
    duplicate_keys: int
    duplicate_rows: int

    @property
    def passed(self) -> bool:
        return (
            self.supplied_rows > 0
            and self.valid_rows == self.supplied_rows
            and self.invalid_key_rows == 0
            and self.invalid_xg_rows == 0
            and self.unsupported_division_rows == 0
            and self.duplicate_keys == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "supplied_rows": self.supplied_rows,
            "valid_rows": self.valid_rows,
            "invalid_key_rows": self.invalid_key_rows,
            "invalid_xg_rows": self.invalid_xg_rows,
            "invalid_xg_values": self.invalid_xg_values,
            "unsupported_division_rows": self.unsupported_division_rows,
            "duplicate_keys": self.duplicate_keys,
            "duplicate_rows": self.duplicate_rows,
            "passed": self.passed,
        }


def _coerce_external_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise UnderstatXGSchemaError("external xG input must be a pandas DataFrame")
    missing = [column for column in UNDERSTAT_XG_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in UNDERSTAT_XG_COLUMNS]
    if missing:
        raise UnderstatXGSchemaError(f"Missing external xG columns: {', '.join(missing)}")
    if extra:
        raise UnderstatXGSchemaError(f"Unexpected external xG columns: {', '.join(extra)}")

    normalized = frame.loc[:, list(UNDERSTAT_XG_COLUMNS)].copy(deep=True)
    normalized["Division"] = normalized["Division"].astype("string").str.strip()
    normalized["MatchDate"] = pd.to_datetime(
        normalized["MatchDate"], errors="coerce"
    ).dt.normalize()
    for column in ("HomeTeam", "AwayTeam"):
        present = normalized[column].notna()
        normalized[column] = normalized[column].map(
            lambda value: normalize_team_name(value) if pd.notna(value) else ""
        )
        normalized.loc[~present, column] = ""
    for column in ("home_xg", "away_xg"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def audit_understat_xg_frame(frame: pd.DataFrame) -> UnderstatXGSchemaAudit:
    """Audit external rows without exposing raw keys or values."""

    normalized = _coerce_external_frame(frame)
    invalid_key = (
        normalized["MatchDate"].isna()
        | normalized["Division"].isna()
        | normalized["Division"].eq("")
        | normalized["HomeTeam"].eq("")
        | normalized["AwayTeam"].eq("")
        | normalized["HomeTeam"].eq(normalized["AwayTeam"])
    )
    xg = normalized.loc[:, ["home_xg", "away_xg"]].to_numpy(dtype=np.float64)
    invalid_cells = ~np.isfinite(xg) | (xg < 0.0)
    invalid_xg = pd.Series(invalid_cells.any(axis=1), index=normalized.index)
    unsupported = ~normalized["Division"].isin(SUPPORTED_DIVISIONS)
    duplicate_mask = normalized.duplicated(list(UNDERSTAT_XG_KEY), keep=False)
    duplicate_keys = int(
        normalized.loc[duplicate_mask, list(UNDERSTAT_XG_KEY)].drop_duplicates().shape[0]
    )
    invalid_rows = invalid_key | invalid_xg | unsupported | duplicate_mask
    return UnderstatXGSchemaAudit(
        supplied_rows=len(normalized),
        valid_rows=int((~invalid_rows).sum()),
        invalid_key_rows=int(invalid_key.sum()),
        invalid_xg_rows=int(invalid_xg.sum()),
        invalid_xg_values=int(invalid_cells.sum()),
        unsupported_division_rows=int(unsupported.sum()),
        duplicate_keys=duplicate_keys,
        duplicate_rows=int(duplicate_mask.sum()),
    )


def validate_understat_xg_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy after all external schema gates pass."""

    normalized = _coerce_external_frame(frame)
    audit = audit_understat_xg_frame(frame)
    if not audit.passed:
        raise UnderstatXGSchemaError(
            "external xG schema failed: "
            f"invalid_key_rows={audit.invalid_key_rows}, "
            f"invalid_xg_rows={audit.invalid_xg_rows}, "
            f"unsupported_division_rows={audit.unsupported_division_rows}, "
            f"duplicate_keys={audit.duplicate_keys}"
        )
    normalized["Division"] = normalized["Division"].astype(str)
    return normalized.sort_values(list(UNDERSTAT_XG_KEY), kind="mergesort").reset_index(drop=True)
