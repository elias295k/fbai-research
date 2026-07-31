"""Exact-key validation and alignment for local availability source frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fbai.core.leakage import NATURAL_KEY, TableValidationError, assert_natural_key_valid
from fbai.core.splits import ALL_TEST_YEARS, STABLE_ORDER_KEY
from fbai.features.checks import validate_feature_table
from fbai.research.player_availability.schema import (
    ALLOWED_LINEUP_ROLES,
    APPEARANCE_COLUMNS,
    FIXTURE_COLUMNS,
    LINEUP_COLUMNS,
    VALUATION_COLUMNS,
)


class AvailabilityAlignmentError(ValueError):
    """Raised when availability inputs or exact-key alignment are invalid."""


@dataclass(frozen=True, slots=True)
class AvailabilityInputQuality:
    """Non-identifying source-row validation counts."""

    supplied_match_rows: int
    appearance_rows: int
    lineup_rows: int
    valuation_rows: int
    appearance_games: int
    lineup_games: int
    duplicate_match_keys: int
    duplicate_game_ids: int
    duplicate_appearance_rows: int
    duplicate_lineup_rows: int
    duplicate_valuation_rows: int
    invalid_minutes_rows: int
    invalid_role_rows: int
    invalid_valuation_rows: int
    unmatched_appearance_games: int
    unmatched_lineup_games: int

    def to_dict(self) -> dict[str, int]:
        return {
            "supplied_match_rows": self.supplied_match_rows,
            "appearance_rows": self.appearance_rows,
            "lineup_rows": self.lineup_rows,
            "valuation_rows": self.valuation_rows,
            "appearance_games": self.appearance_games,
            "lineup_games": self.lineup_games,
            "duplicate_match_keys": self.duplicate_match_keys,
            "duplicate_game_ids": self.duplicate_game_ids,
            "duplicate_appearance_rows": self.duplicate_appearance_rows,
            "duplicate_lineup_rows": self.duplicate_lineup_rows,
            "duplicate_valuation_rows": self.duplicate_valuation_rows,
            "invalid_minutes_rows": self.invalid_minutes_rows,
            "invalid_role_rows": self.invalid_role_rows,
            "invalid_valuation_rows": self.invalid_valuation_rows,
            "unmatched_appearance_games": self.unmatched_appearance_games,
            "unmatched_lineup_games": self.unmatched_lineup_games,
        }


@dataclass(frozen=True, slots=True)
class AvailabilitySourceFrames:
    """Validated, deterministic local input frames."""

    fixtures: pd.DataFrame
    appearances: pd.DataFrame
    lineups: pd.DataFrame
    valuations: pd.DataFrame
    quality: AvailabilityInputQuality


@dataclass(frozen=True, slots=True)
class CoverageRecord:
    """Base and covered match counts for one aggregate group."""

    base_rows: int
    covered_rows: int

    @property
    def covered_share(self) -> float:
        return self.covered_rows / self.base_rows if self.base_rows else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "base_rows": self.base_rows,
            "covered_rows": self.covered_rows,
            "covered_share": self.covered_share,
        }


@dataclass(frozen=True, slots=True)
class AvailabilityAlignmentReport:
    """Aggregate-only exact-key availability coverage."""

    base_match_rows: int
    supplied_match_rows: int
    covered_match_rows: int
    unmatched_base_matches: int
    unmatched_supplied_matches: int
    coverage_filter_changes_lr52_population: bool
    exact_key_only: bool
    by_division: dict[str, CoverageRecord]
    by_season: dict[str, CoverageRecord]
    by_fold: dict[str, CoverageRecord]
    input_quality: AvailabilityInputQuality

    @property
    def covered_share(self) -> float:
        return self.covered_match_rows / self.base_match_rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_match_rows": self.base_match_rows,
            "supplied_match_rows": self.supplied_match_rows,
            "covered_match_rows": self.covered_match_rows,
            "unmatched_base_matches": self.unmatched_base_matches,
            "unmatched_supplied_matches": self.unmatched_supplied_matches,
            "covered_share": self.covered_share,
            "coverage_filter_changes_lr52_population": (
                self.coverage_filter_changes_lr52_population
            ),
            "exact_key_only": self.exact_key_only,
            "by_division": {key: value.to_dict() for key, value in self.by_division.items()},
            "by_season": {key: value.to_dict() for key, value in self.by_season.items()},
            "by_fold": {key: value.to_dict() for key, value in self.by_fold.items()},
            "input_quality": self.input_quality.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AvailabilityAlignment:
    """Covered stable feature rows plus their aggregate coverage report."""

    feature_table: pd.DataFrame
    report: AvailabilityAlignmentReport


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    name: str,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise AvailabilityAlignmentError(f"{name} must be a pandas DataFrame")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AvailabilityAlignmentError(f"{name} is missing: {', '.join(missing)}")


def _numeric_ids(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise AvailabilityAlignmentError(f"{name} {column} must be complete numeric IDs")
        frame[column] = numeric.astype("int64")


def prepare_availability_sources(
    fixtures: pd.DataFrame,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
    valuations: pd.DataFrame,
) -> AvailabilitySourceFrames:
    """Validate source frames without fuzzy matching or silent row repair."""

    _require_columns(fixtures, FIXTURE_COLUMNS, "availability fixtures")
    _require_columns(appearances, APPEARANCE_COLUMNS, "availability appearances")
    _require_columns(lineups, LINEUP_COLUMNS, "availability lineups")
    _require_columns(valuations, VALUATION_COLUMNS, "availability valuations")
    fixture_frame = fixtures.loc[:, list(FIXTURE_COLUMNS)].copy(deep=True)
    appearance_frame = appearances.loc[:, list(APPEARANCE_COLUMNS)].copy(deep=True)
    lineup_frame = lineups.loc[:, list(LINEUP_COLUMNS)].copy(deep=True)
    valuation_frame = valuations.loc[:, list(VALUATION_COLUMNS)].copy(deep=True)

    fixture_frame["MatchDate"] = pd.to_datetime(fixture_frame["MatchDate"], errors="raise")
    duplicate_match_keys = int(fixture_frame.duplicated(subset=list(NATURAL_KEY), keep=False).sum())
    duplicate_game_ids = int(fixture_frame["game_id"].duplicated(keep=False).sum())
    try:
        assert_natural_key_valid(fixture_frame, key_columns=NATURAL_KEY)
    except TableValidationError as exc:
        raise AvailabilityAlignmentError(str(exc)) from exc
    if duplicate_game_ids:
        raise AvailabilityAlignmentError(
            f"availability fixtures contain {duplicate_game_ids} duplicate game IDs"
        )
    _numeric_ids(
        fixture_frame,
        ("SeasonStartYear", "game_id", "home_club_id", "away_club_id"),
        "availability fixtures",
    )
    _numeric_ids(
        appearance_frame,
        ("game_id", "player_id", "player_club_id"),
        "availability appearances",
    )
    _numeric_ids(
        lineup_frame,
        ("game_id", "player_id", "club_id"),
        "availability lineups",
    )
    _numeric_ids(valuation_frame, ("player_id",), "availability valuations")

    appearance_frame["minutes_played"] = pd.to_numeric(
        appearance_frame["minutes_played"], errors="coerce"
    )
    invalid_minutes = (
        appearance_frame["minutes_played"].isna()
        | ~np.isfinite(appearance_frame["minutes_played"].to_numpy(dtype=float))
        | appearance_frame["minutes_played"].lt(0.0)
    )
    invalid_minutes_rows = int(invalid_minutes.sum())
    if invalid_minutes_rows:
        raise AvailabilityAlignmentError(
            f"availability appearances contain {invalid_minutes_rows} invalid minutes rows"
        )
    lineup_frame["type"] = lineup_frame["type"].astype(str)
    invalid_roles = ~lineup_frame["type"].isin(ALLOWED_LINEUP_ROLES)
    invalid_role_rows = int(invalid_roles.sum())
    if invalid_role_rows:
        raise AvailabilityAlignmentError(
            f"availability lineups contain {invalid_role_rows} invalid role rows"
        )
    valuation_frame["date"] = pd.to_datetime(valuation_frame["date"], errors="coerce")
    valuation_frame["market_value_in_eur"] = pd.to_numeric(
        valuation_frame["market_value_in_eur"], errors="coerce"
    )
    invalid_valuations = (
        valuation_frame["date"].isna()
        | valuation_frame["market_value_in_eur"].isna()
        | ~np.isfinite(valuation_frame["market_value_in_eur"].to_numpy(dtype=float))
        | valuation_frame["market_value_in_eur"].lt(0.0)
    )
    invalid_valuation_rows = int(invalid_valuations.sum())
    if invalid_valuation_rows:
        raise AvailabilityAlignmentError(
            f"availability valuations contain {invalid_valuation_rows} invalid rows"
        )

    duplicate_appearances = int(
        appearance_frame.duplicated(
            subset=["game_id", "player_id", "player_club_id"], keep=False
        ).sum()
    )
    duplicate_lineups = int(
        lineup_frame.duplicated(
            subset=["game_id", "player_id", "club_id", "type"], keep=False
        ).sum()
    )
    duplicate_valuations = int(
        valuation_frame.duplicated(subset=["player_id", "date"], keep=False).sum()
    )
    if duplicate_appearances or duplicate_lineups or duplicate_valuations:
        raise AvailabilityAlignmentError(
            "availability player relations contain duplicate identifiers"
        )

    game_ids = frozenset(fixture_frame["game_id"].astype(int))
    unmatched_appearance_games = len(
        set(appearance_frame["game_id"].astype(int)).difference(game_ids)
    )
    unmatched_lineup_games = len(set(lineup_frame["game_id"].astype(int)).difference(game_ids))
    if unmatched_appearance_games or unmatched_lineup_games:
        raise AvailabilityAlignmentError(
            "availability player relations reference games outside supplied fixtures"
        )

    fixture_frame = fixture_frame.sort_values(list(STABLE_ORDER_KEY), kind="mergesort").reset_index(
        drop=True
    )
    appearance_frame = appearance_frame.sort_values(
        ["game_id", "player_club_id", "player_id"], kind="mergesort"
    ).reset_index(drop=True)
    lineup_frame = lineup_frame.sort_values(
        ["game_id", "club_id", "player_id", "type"], kind="mergesort"
    ).reset_index(drop=True)
    valuation_frame = valuation_frame.sort_values(
        ["player_id", "date"], kind="mergesort"
    ).reset_index(drop=True)
    quality = AvailabilityInputQuality(
        supplied_match_rows=len(fixture_frame),
        appearance_rows=len(appearance_frame),
        lineup_rows=len(lineup_frame),
        valuation_rows=len(valuation_frame),
        appearance_games=int(appearance_frame["game_id"].nunique()),
        lineup_games=int(lineup_frame["game_id"].nunique()),
        duplicate_match_keys=duplicate_match_keys,
        duplicate_game_ids=duplicate_game_ids,
        duplicate_appearance_rows=duplicate_appearances,
        duplicate_lineup_rows=duplicate_lineups,
        duplicate_valuation_rows=duplicate_valuations,
        invalid_minutes_rows=invalid_minutes_rows,
        invalid_role_rows=invalid_role_rows,
        invalid_valuation_rows=invalid_valuation_rows,
        unmatched_appearance_games=unmatched_appearance_games,
        unmatched_lineup_games=unmatched_lineup_games,
    )
    return AvailabilitySourceFrames(
        fixtures=fixture_frame,
        appearances=appearance_frame,
        lineups=lineup_frame,
        valuations=valuation_frame,
        quality=quality,
    )


def _coverage_map(
    base: pd.DataFrame,
    covered: pd.DataFrame,
    column: str,
) -> dict[str, CoverageRecord]:
    values = sorted(set(base[column].astype(str)).union(covered[column].astype(str)))
    return {
        value: CoverageRecord(
            base_rows=int(base[column].astype(str).eq(value).sum()),
            covered_rows=int(covered[column].astype(str).eq(value).sum()),
        )
        for value in values
    }


def align_availability_scope(
    feature_table: pd.DataFrame,
    sources: AvailabilitySourceFrames,
) -> AvailabilityAlignment:
    """Align availability fixtures to stable features by exact natural key."""

    ordered = feature_table.sort_values(list(STABLE_ORDER_KEY), kind="mergesort").reset_index(
        drop=True
    )
    validate_feature_table(ordered, expected_row_count=len(feature_table))
    fixture_keys = sources.fixtures.loc[:, list(NATURAL_KEY)]
    covered = ordered.merge(
        fixture_keys,
        on=list(NATURAL_KEY),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    covered = covered.sort_values(list(STABLE_ORDER_KEY), kind="mergesort").reset_index(drop=True)
    covered_keys = set(covered.loc[:, list(NATURAL_KEY)].itertuples(index=False, name=None))
    fixture_key_set = set(fixture_keys.itertuples(index=False, name=None))
    base_key_set = set(ordered.loc[:, list(NATURAL_KEY)].itertuples(index=False, name=None))
    by_fold = {
        str(year): CoverageRecord(
            base_rows=int(ordered["SeasonStartYear"].eq(year).sum()),
            covered_rows=int(covered["SeasonStartYear"].eq(year).sum()),
        )
        for year in ALL_TEST_YEARS
    }
    report = AvailabilityAlignmentReport(
        base_match_rows=len(ordered),
        supplied_match_rows=len(sources.fixtures),
        covered_match_rows=len(covered),
        unmatched_base_matches=len(base_key_set.difference(fixture_key_set)),
        unmatched_supplied_matches=len(fixture_key_set.difference(base_key_set)),
        coverage_filter_changes_lr52_population=len(covered) != len(ordered),
        exact_key_only=True,
        by_division=_coverage_map(ordered, covered, "Division"),
        by_season=_coverage_map(ordered, covered, "SeasonStartYear"),
        by_fold=by_fold,
        input_quality=sources.quality,
    )
    if len(covered_keys) != len(covered):
        raise AvailabilityAlignmentError("covered availability keys are not unique")
    return AvailabilityAlignment(feature_table=covered, report=report)
