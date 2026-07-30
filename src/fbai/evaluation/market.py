"""Strict closing-market normalization and H/D/A probability construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fbai.core.metrics import CLASS_ORDER, validate_probabilities
from fbai.data.schema import CanonicalSchemaError, parse_match_dates

MARKET_NATURAL_KEY: tuple[str, ...] = (
    "MatchDate",
    "Division",
    "HomeTeam",
    "AwayTeam",
)
CANONICAL_ODDS_COLUMNS: tuple[str, str, str] = (
    "ClosingHomeOdds",
    "ClosingDrawOdds",
    "ClosingAwayOdds",
)
CANONICAL_MARKET_COLUMNS: tuple[str, ...] = (*MARKET_NATURAL_KEY, *CANONICAL_ODDS_COLUMNS)
AUTHORITATIVE_SOURCE_ODDS_COLUMNS: tuple[str, str, str] = ("AvgCH", "AvgCD", "AvgCA")
MARKET_PROBABILITY_COLUMNS: tuple[str, str, str] = (
    "probability_home",
    "probability_draw",
    "probability_away",
)
MARKET_OVERROUND_COLUMN = "market_overround"
MARKET_PROBABILITY_OUTPUT_COLUMNS: tuple[str, ...] = (
    *MARKET_NATURAL_KEY,
    *MARKET_PROBABILITY_COLUMNS,
    MARKET_OVERROUND_COLUMN,
)
MARKET_CLASS_ORDER: tuple[str, str, str] = CLASS_ORDER

_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "MatchDate": ("MatchDate", "Date"),
    "Division": ("Division", "Div"),
    "HomeTeam": ("HomeTeam",),
    "AwayTeam": ("AwayTeam",),
    "ClosingHomeOdds": ("ClosingHomeOdds", "AvgCH"),
    "ClosingDrawOdds": ("ClosingDrawOdds", "AvgCD"),
    "ClosingAwayOdds": ("ClosingAwayOdds", "AvgCA"),
}


class MarketSchemaError(ValueError):
    """Raised when closing-market records violate the public contract."""


@dataclass(frozen=True, slots=True)
class MarketInputAudit:
    """Aggregate validation counts for a supplied closing-market table."""

    supplied_market_rows: int
    valid_market_rows: int
    duplicate_market_keys: int
    incomplete_odds_rows: int
    invalid_odds_rows: int

    def to_dict(self) -> dict[str, int]:
        """Return JSON-safe aggregate counts."""

        return {
            "supplied_market_rows": self.supplied_market_rows,
            "valid_market_rows": self.valid_market_rows,
            "duplicate_market_keys": self.duplicate_market_keys,
            "incomplete_odds_rows": self.incomplete_odds_rows,
            "invalid_odds_rows": self.invalid_odds_rows,
        }


def _select_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise MarketSchemaError("Closing-market input must be a pandas DataFrame")

    selected: dict[str, pd.Series[Any]] = {}
    missing: list[str] = []
    for canonical, candidates in _COLUMN_CANDIDATES.items():
        source = next((candidate for candidate in candidates if candidate in frame.columns), None)
        if source is None:
            missing.append(canonical)
        else:
            selected[canonical] = frame[source].copy(deep=True)
    if missing:
        raise MarketSchemaError(f"Missing required market fields: {', '.join(missing)}")
    return pd.DataFrame(selected, index=frame.index.copy())


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    try:
        normalized["MatchDate"] = parse_match_dates(normalized["MatchDate"])
    except CanonicalSchemaError as exc:
        raise MarketSchemaError(str(exc)) from exc

    for column in ("Division", "HomeTeam", "AwayTeam"):
        values = normalized[column].astype("string")
        invalid = values.isna() | values.str.strip().eq("")
        if bool(invalid.any()):
            raise MarketSchemaError(f"{column} must contain non-empty strings")
        normalized[column] = values.str.strip()

    duplicate_rows = int(normalized.duplicated(subset=list(MARKET_NATURAL_KEY), keep=False).sum())
    if duplicate_rows:
        raise MarketSchemaError(
            f"Market natural key is not unique; {duplicate_rows} rows are duplicated"
        )
    return normalized


def _odds_masks(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    raw_odds = selected.loc[:, list(CANONICAL_ODDS_COLUMNS)].copy(deep=True)
    text = raw_odds.astype("string")
    missing_cells = raw_odds.isna() | text.apply(lambda values: values.str.strip().eq(""))
    incomplete_rows = missing_cells.any(axis=1)

    numeric = raw_odds.apply(pd.to_numeric, errors="coerce").astype("float64")
    numeric_values = numeric.to_numpy(dtype="float64")
    nonfinite_rows = pd.Series(
        ~np.isfinite(numeric_values).all(axis=1),
        index=numeric.index,
        dtype=bool,
    )
    invalid_rows = ~incomplete_rows & (
        numeric.isna().any(axis=1) | nonfinite_rows | numeric.le(1.0).any(axis=1)
    )
    return numeric, incomplete_rows.astype(bool), invalid_rows.astype(bool)


def _canonicalize_with_masks(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    selected = _normalize_keys(_select_columns(frame))
    numeric, incomplete_rows, invalid_rows = _odds_masks(selected)
    for column in CANONICAL_ODDS_COLUMNS:
        selected[column] = numeric[column].astype("float64")
    return selected, incomplete_rows, invalid_rows


def _stable_market_order(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.loc[:, list(CANONICAL_MARKET_COLUMNS)]
        .sort_values(list(MARKET_NATURAL_KEY), kind="mergesort")
        .reset_index(drop=True)
    )


def normalize_closing_market(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported aliases and require every closing-odds row to be valid.

    FBAI_NEW's authoritative closing-average aliases are ``AvgCH``,
    ``AvgCD``, and ``AvgCA``. Provider-specific extras are deliberately
    ignored and the input frame is never mutated.
    """

    canonical, incomplete_rows, invalid_rows = _canonicalize_with_masks(frame)
    if canonical.empty:
        raise MarketSchemaError("Closing-market table must contain at least one row")
    if bool(incomplete_rows.any()):
        count = int(incomplete_rows.sum())
        raise MarketSchemaError(f"Closing odds are incomplete on {count} rows")
    if bool(invalid_rows.any()):
        count = int(invalid_rows.sum())
        raise MarketSchemaError(
            f"Closing odds must be numeric, finite, and strictly greater than 1.0; "
            f"{count} rows are invalid"
        )
    return _stable_market_order(canonical)


def prepare_closing_market(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, MarketInputAudit]:
    """Return explicitly filtered valid rows plus complete validation counts.

    Unlike :func:`normalize_closing_market`, this evaluation preparation path
    may exclude incomplete or invalid odds. The returned audit makes every
    exclusion observable. Duplicate natural keys still fail rather than being
    selected or dropped.
    """

    canonical, incomplete_rows, invalid_rows = _canonicalize_with_masks(frame)
    valid_mask = ~(incomplete_rows | invalid_rows)
    valid = _stable_market_order(canonical.loc[valid_mask].copy(deep=True))
    audit = MarketInputAudit(
        supplied_market_rows=len(canonical),
        valid_market_rows=len(valid),
        duplicate_market_keys=0,
        incomplete_odds_rows=int(incomplete_rows.sum()),
        invalid_odds_rows=int(invalid_rows.sum()),
    )
    return valid, audit


def closing_market_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert valid decimal closing odds to de-vigged H/D/A probabilities.

    The source-verified method is reciprocal implied probability followed by
    division by the row's implied-probability sum. The unnormalized sum is
    retained as the overround diagnostic; it is not required to exceed one.
    """

    canonical = normalize_closing_market(frame)
    odds = canonical.loc[:, list(CANONICAL_ODDS_COLUMNS)].to_numpy(dtype="float64")
    raw_implied = np.reciprocal(odds)
    overround = raw_implied.sum(axis=1)
    if not np.isfinite(overround).all() or bool((overround <= 0.0).any()):
        raise MarketSchemaError("Implied-probability sums must be finite and strictly positive")
    probabilities = raw_implied / overround[:, np.newaxis]
    try:
        validated = validate_probabilities(probabilities)
    except ValueError as exc:
        raise MarketSchemaError(str(exc)) from exc

    output = canonical.loc[:, list(MARKET_NATURAL_KEY)].copy(deep=True)
    for index, column in enumerate(MARKET_PROBABILITY_COLUMNS):
        output[column] = validated[:, index]
    output[MARKET_OVERROUND_COLUMN] = overround
    return output.loc[:, list(MARKET_PROBABILITY_OUTPUT_COLUMNS)]
