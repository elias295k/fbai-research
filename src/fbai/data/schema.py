"""Explicit canonical match schema and integrity validation."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import date, datetime
from numbers import Integral, Real

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype

NATURAL_KEY: tuple[str, ...] = ("MatchDate", "Division", "HomeTeam", "AwayTeam")
STABLE_SORT_KEY: tuple[str, ...] = NATURAL_KEY

IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "MatchDate",
    "SeasonStartYear",
    "Division",
    "HomeTeam",
    "AwayTeam",
)
OUTCOME_COLUMNS: tuple[str, ...] = ("FTR", "FTHome", "FTAway")
MATCH_STAT_COLUMNS: tuple[str, ...] = (
    "HomeShots",
    "AwayShots",
    "HomeTarget",
    "AwayTarget",
    "HomeCorners",
    "AwayCorners",
    "HomeFouls",
    "AwayFouls",
    "HomeYellow",
    "AwayYellow",
    "HomeRed",
    "AwayRed",
)
NUMERIC_MATCH_COLUMNS: tuple[str, ...] = ("FTHome", "FTAway", *MATCH_STAT_COLUMNS)
CANONICAL_COLUMNS: tuple[str, ...] = (
    *IDENTIFIER_COLUMNS,
    *OUTCOME_COLUMNS,
    *MATCH_STAT_COLUMNS,
)

SOURCE_COLUMN_ALIASES: dict[str, str] = {
    "Date": "MatchDate",
    "Div": "Division",
    "FTHG": "FTHome",
    "FTAG": "FTAway",
    "FTR": "FTR",
    "HS": "HomeShots",
    "AS": "AwayShots",
    "HST": "HomeTarget",
    "AST": "AwayTarget",
    "HC": "HomeCorners",
    "AC": "AwayCorners",
    "HF": "HomeFouls",
    "AF": "AwayFouls",
    "HY": "HomeYellow",
    "AY": "AwayYellow",
    "HR": "HomeRed",
    "AR": "AwayRed",
}

_DATE_FORMATS: tuple[str, ...] = ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y")
_LONG_SEASON_PATTERN = re.compile(r"^(\d{4})\s*[-/]\s*(\d{4})$")


class CanonicalSchemaError(ValueError):
    """Raised when source or canonical data violates the Phase 2A contract."""


def parse_match_dates(values: pd.Series) -> pd.Series:
    """Parse supported source date formats into normalized, timezone-naive dates."""

    parsed: list[pd.Timestamp] = []
    invalid: list[str] = []
    for index, value in values.items():
        timestamp: pd.Timestamp | None = None
        if isinstance(value, (pd.Timestamp, datetime, date)):
            timestamp = pd.Timestamp(value)
        elif isinstance(value, str):
            text = value.strip()
            for date_format in _DATE_FORMATS:
                try:
                    timestamp = pd.Timestamp(datetime.strptime(text, date_format))
                    break
                except ValueError:
                    continue
        if timestamp is None or pd.isna(timestamp):
            invalid.append(f"{index!r}={value!r}")
            continue
        if timestamp.tzinfo is not None:
            invalid.append(f"{index!r}={value!r} (timezone-aware)")
            continue
        parsed.append(timestamp.normalize())

    if invalid:
        preview = ", ".join(invalid[:5])
        suffix = "" if len(invalid) <= 5 else f" (+{len(invalid) - 5} more)"
        raise CanonicalSchemaError(f"Unparseable MatchDate values: {preview}{suffix}")
    return pd.Series(parsed, index=values.index, dtype="datetime64[ns]", name=values.name)


def normalize_season_start_year(value: object) -> int:
    """Normalize an explicit start year or long season label."""

    year: int | None = None
    if isinstance(value, bool):
        year = None
    elif isinstance(value, Integral):
        year = int(value)
    elif isinstance(value, Real) and math.isfinite(float(value)):
        numeric = float(value)
        if numeric.is_integer():
            year = int(numeric)
    elif isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"\d{4}", text):
            year = int(text)
        else:
            match = _LONG_SEASON_PATTERN.fullmatch(text)
            if match:
                start = int(match.group(1))
                end = int(match.group(2))
                if end == start + 1:
                    year = start

    if year is None or not 1900 <= year <= 2100:
        raise CanonicalSchemaError(
            f"Malformed or ambiguous season value {value!r}; use an explicit "
            "four-digit start year or YYYY-YYYY label"
        )
    return year


def normalize_season_start_years(values: pd.Series) -> pd.Series:
    """Normalize a series of explicit season values to nullable integers."""

    normalized: list[int] = []
    invalid: list[str] = []
    for index, value in values.items():
        try:
            normalized.append(normalize_season_start_year(value))
        except CanonicalSchemaError:
            invalid.append(f"{index!r}={value!r}")
    if invalid:
        preview = ", ".join(invalid[:5])
        suffix = "" if len(invalid) <= 5 else f" (+{len(invalid) - 5} more)"
        raise CanonicalSchemaError(f"Invalid SeasonStartYear values: {preview}{suffix}")
    return pd.Series(normalized, index=values.index, dtype="Int64", name=values.name)


def _require_exact_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in CANONICAL_COLUMNS]
    if missing:
        raise CanonicalSchemaError(f"Missing canonical columns: {', '.join(missing)}")
    if extra:
        raise CanonicalSchemaError(f"Unexpected canonical columns: {', '.join(extra)}")


def _require_nonempty_strings(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        values = frame[column].astype("string")
        invalid = values.isna() | values.str.strip().eq("")
        if bool(invalid.any()):
            rows = ", ".join(str(index) for index in frame.index[invalid][:5])
            raise CanonicalSchemaError(f"{column} must contain non-empty strings; rows: {rows}")


def validate_canonical_frame(frame: pd.DataFrame) -> None:
    """Validate the complete canonical schema without mutating the frame."""

    _require_exact_columns(frame)
    if frame.empty:
        raise CanonicalSchemaError("Canonical frame must contain at least one match")
    if not is_datetime64_any_dtype(frame["MatchDate"].dtype):
        raise CanonicalSchemaError("MatchDate must use a pandas datetime dtype")
    if frame["MatchDate"].isna().any():
        raise CanonicalSchemaError("MatchDate contains null or invalid dates")
    if frame["MatchDate"].dt.tz is not None:
        raise CanonicalSchemaError("MatchDate must be timezone-naive")
    if not frame["MatchDate"].eq(frame["MatchDate"].dt.normalize()).all():
        raise CanonicalSchemaError("MatchDate values must be normalized to midnight")

    if not is_integer_dtype(frame["SeasonStartYear"].dtype):
        raise CanonicalSchemaError("SeasonStartYear must use an integer dtype")
    if frame["SeasonStartYear"].isna().any():
        raise CanonicalSchemaError("SeasonStartYear contains null values")

    _require_nonempty_strings(frame, ("Division", "HomeTeam", "AwayTeam"))
    if frame["HomeTeam"].eq(frame["AwayTeam"]).any():
        raise CanonicalSchemaError("HomeTeam and AwayTeam must differ on every row")

    invalid_results = ~frame["FTR"].isin(["H", "D", "A"])
    if bool(invalid_results.any()):
        values = sorted(set(frame.loc[invalid_results, "FTR"].astype(str)))
        raise CanonicalSchemaError(f"FTR contains values outside H/D/A: {values}")

    for column in NUMERIC_MATCH_COLUMNS:
        if not is_numeric_dtype(frame[column].dtype):
            raise CanonicalSchemaError(f"{column} must be numeric")
        present = frame[column].dropna()
        if bool((present < 0).any()):
            raise CanonicalSchemaError(f"{column} must be non-negative")
        if bool((present % 1 != 0).any()):
            raise CanonicalSchemaError(f"{column} must contain integer counts")

    null_key_cells = int(frame.loc[:, list(NATURAL_KEY)].isna().sum().sum())
    if null_key_cells:
        raise CanonicalSchemaError(f"Natural key contains {null_key_cells} null cells")
    duplicate_rows = int(frame.duplicated(list(NATURAL_KEY), keep=False).sum())
    if duplicate_rows:
        raise CanonicalSchemaError(
            f"Natural key is not unique; {duplicate_rows} rows belong to duplicate keys"
        )


def sort_canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return the stable canonical row order."""

    validate_canonical_frame(frame)
    return frame.sort_values(list(STABLE_SORT_KEY), kind="mergesort").reset_index(drop=True)
