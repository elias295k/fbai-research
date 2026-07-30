"""Normalize source-shaped CSV rows into the canonical match schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fbai.data.schema import (
    CANONICAL_COLUMNS,
    NUMERIC_MATCH_COLUMNS,
    SOURCE_COLUMN_ALIASES,
    CanonicalSchemaError,
    normalize_season_start_year,
    normalize_season_start_years,
    parse_match_dates,
    sort_canonical_frame,
)
from fbai.data.sources import season_start_year as source_season_start_year


class MissingSourceColumnsError(CanonicalSchemaError):
    """Raised when source rows cannot supply every required canonical field."""


def _source_candidates(canonical_column: str) -> tuple[str, ...]:
    aliases = [
        source
        for source, canonical in SOURCE_COLUMN_ALIASES.items()
        if canonical == canonical_column and source != canonical_column
    ]
    return (canonical_column, *aliases)


def _select_source_column(raw: pd.DataFrame, canonical_column: str) -> pd.Series | None:
    present = [column for column in _source_candidates(canonical_column) if column in raw]
    if not present:
        return None
    if len(present) > 1:
        raise CanonicalSchemaError(
            f"Ambiguous source columns for {canonical_column}: {', '.join(present)}"
        )
    return raw[present[0]].copy()


def _resolve_seasons(
    raw: pd.DataFrame,
    *,
    season_start_year: int | str | None,
    source_season_code: str | None,
) -> pd.Series | None:
    if season_start_year is not None and source_season_code is not None:
        raise CanonicalSchemaError("Provide season_start_year or source_season_code, not both")

    raw_values = _select_source_column(raw, "SeasonStartYear")
    context_year: int | None = None
    if season_start_year is not None:
        context_year = normalize_season_start_year(season_start_year)
    elif source_season_code is not None:
        context_year = source_season_start_year(source_season_code)

    if raw_values is not None:
        normalized = normalize_season_start_years(raw_values)
        if context_year is not None and not normalized.eq(context_year).all():
            raise CanonicalSchemaError(
                "Source SeasonStartYear values conflict with acquisition context"
            )
        return normalized
    if "Season" in raw:
        normalized = normalize_season_start_years(raw["Season"])
        if context_year is not None and not normalized.eq(context_year).all():
            raise CanonicalSchemaError("Source Season values conflict with acquisition context")
        return normalized
    if context_year is not None:
        return pd.Series(context_year, index=raw.index, dtype="Int64")
    return None


def _coerce_numeric_counts(frame: pd.DataFrame) -> None:
    for column in NUMERIC_MATCH_COLUMNS:
        source = frame[column]
        converted = pd.to_numeric(source, errors="coerce")
        invalid = source.notna() & converted.isna()
        if bool(invalid.any()):
            rows = ", ".join(str(index) for index in frame.index[invalid][:5])
            raise CanonicalSchemaError(f"{column} contains non-numeric values; rows: {rows}")
        present = converted.dropna()
        if bool((present % 1 != 0).any()):
            raise CanonicalSchemaError(f"{column} contains non-integer match counts")
        frame[column] = converted.astype("Int64")


def canonicalize_source_frame(
    raw: pd.DataFrame,
    *,
    season_start_year: int | str | None = None,
    source_season_code: str | None = None,
) -> pd.DataFrame:
    """Normalize source aliases and values into a validated canonical frame.

    Unrelated columns are ignored. Required match statistics are never filled
    with fabricated values.
    """

    selected: dict[str, pd.Series] = {}
    missing: list[str] = []
    for canonical_column in CANONICAL_COLUMNS:
        if canonical_column == "SeasonStartYear":
            continue
        values = _select_source_column(raw, canonical_column)
        if values is None:
            missing.append(canonical_column)
        else:
            selected[canonical_column] = values

    seasons = _resolve_seasons(
        raw,
        season_start_year=season_start_year,
        source_season_code=source_season_code,
    )
    if seasons is None:
        missing.append("SeasonStartYear")
    else:
        selected["SeasonStartYear"] = seasons
    if missing:
        ordered = [column for column in CANONICAL_COLUMNS if column in missing]
        raise MissingSourceColumnsError(
            f"Source rows missing required canonical fields: {', '.join(ordered)}"
        )

    canonical = pd.DataFrame(selected, index=raw.index).loc[:, list(CANONICAL_COLUMNS)]
    canonical["MatchDate"] = parse_match_dates(canonical["MatchDate"])
    canonical["SeasonStartYear"] = normalize_season_start_years(canonical["SeasonStartYear"])
    for column in ("Division", "HomeTeam", "AwayTeam", "FTR"):
        canonical[column] = canonical[column].astype("string").str.strip()
    _coerce_numeric_counts(canonical)
    return sort_canonical_frame(canonical.reset_index(drop=True))


def load_source_csv(
    path: Path,
    *,
    season_start_year: int | str | None = None,
    source_season_code: str | None = None,
) -> pd.DataFrame:
    """Load a local source CSV and return its canonical representation."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source CSV not found: {source_path}")
    try:
        raw = pd.read_csv(source_path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        raw = pd.read_csv(source_path, encoding="latin-1", low_memory=False)
    return canonicalize_source_frame(
        raw,
        season_start_year=season_start_year,
        source_season_code=source_season_code,
    )
