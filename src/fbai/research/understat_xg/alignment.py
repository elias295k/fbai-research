"""One-to-one alignment and aggregate quality reporting for external xG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fbai.data.schema import NATURAL_KEY, validate_canonical_frame
from fbai.research.understat_xg.schema import (
    UNDERSTAT_XG_KEY,
    UnderstatXGSchemaAudit,
    audit_understat_xg_frame,
    normalize_team_name,
    validate_understat_xg_frame,
)

ALIGNED_XG_COLUMNS: tuple[str, ...] = (
    *NATURAL_KEY,
    "SeasonStartYear",
    "FTHome",
    "FTAway",
    "home_xg",
    "away_xg",
)


class UnderstatXGAlignmentError(ValueError):
    """Raised when exact external alignment or its quality gates fail."""


@dataclass(frozen=True, slots=True)
class CoverageRecord:
    """Aggregate coverage for a division, season, fold, or overall view."""

    name: str
    canonical_rows: int
    aligned_rows: int

    @property
    def coverage(self) -> float:
        return self.aligned_rows / self.canonical_rows if self.canonical_rows else 0.0

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "name": self.name,
            "canonical_rows": self.canonical_rows,
            "aligned_rows": self.aligned_rows,
            "coverage": self.coverage,
        }


@dataclass(frozen=True, slots=True)
class UnderstatXGQualityReport:
    """Source-defined plausibility gates evaluated on aligned observations."""

    covered_league_seasons: int
    minimum_claimed_league_season_coverage: float
    value_coverage: float
    minimum_xg: float | None
    maximum_xg: float | None
    minimum_side_mean: float | None
    maximum_side_mean: float | None
    goals_correlation: float | None
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": {
                "minimum_join_coverage": 0.98,
                "minimum_value_coverage": 0.98,
                "xg_range": [0.0, 10.0],
                "side_mean_range": [0.7, 2.3],
                "goals_correlation_range": [0.45, 0.95],
            },
            "covered_league_seasons": self.covered_league_seasons,
            "minimum_claimed_league_season_coverage": (self.minimum_claimed_league_season_coverage),
            "value_coverage": self.value_coverage,
            "xg_range": [self.minimum_xg, self.maximum_xg],
            "side_mean_range": [self.minimum_side_mean, self.maximum_side_mean],
            "goals_correlation": self.goals_correlation,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class UnderstatXGAlignmentReport:
    """Aggregate-only exact-join report."""

    method: str
    schema: UnderstatXGSchemaAudit
    covered_divisions: tuple[str, ...]
    canonical_rows_in_covered_divisions: int
    aligned_rows: int
    unmatched_canonical_rows: int
    unmatched_external_rows: int
    coverage_by_division: tuple[CoverageRecord, ...]
    coverage_by_league_season: tuple[CoverageRecord, ...]
    coverage_by_evaluation_fold: tuple[CoverageRecord, ...]
    quality: UnderstatXGQualityReport

    @property
    def passed(self) -> bool:
        return self.schema.passed and self.quality.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "schema": self.schema.to_dict(),
            "covered_divisions": list(self.covered_divisions),
            "canonical_rows_in_covered_divisions": self.canonical_rows_in_covered_divisions,
            "aligned_rows": self.aligned_rows,
            "unmatched_canonical_rows": self.unmatched_canonical_rows,
            "unmatched_external_rows": self.unmatched_external_rows,
            "coverage_by_division": [record.to_dict() for record in self.coverage_by_division],
            "coverage_by_league_season": [
                record.to_dict() for record in self.coverage_by_league_season
            ],
            "coverage_by_evaluation_fold": [
                record.to_dict() for record in self.coverage_by_evaluation_fold
            ],
            "quality": self.quality.to_dict(),
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class AlignedUnderstatXG:
    """Internal aligned research table plus its safe public report."""

    matches: pd.DataFrame
    report: UnderstatXGAlignmentReport


def _coverage_records(
    merged: pd.DataFrame,
    group_columns: tuple[str, ...],
    *,
    name_prefix: str = "",
) -> tuple[CoverageRecord, ...]:
    records: list[CoverageRecord] = []
    grouped = merged.groupby(list(group_columns), sort=True, observed=True)
    for group, rows in grouped:
        parts = group if isinstance(group, tuple) else (group,)
        name = name_prefix + "/".join(str(part) for part in parts)
        records.append(
            CoverageRecord(
                name=name,
                canonical_rows=len(rows),
                aligned_rows=int(rows["_aligned"].sum()),
            )
        )
    return tuple(records)


def _quality_report(merged: pd.DataFrame) -> UnderstatXGQualityReport:
    per_season = _coverage_records(merged, ("Division", "SeasonStartYear"))
    claimed = tuple(record for record in per_season if record.aligned_rows > 0)
    minimum_coverage = min((record.coverage for record in claimed), default=0.0)
    aligned = merged.loc[merged["_aligned"]].copy()
    value_coverage = (
        float(aligned[["home_xg", "away_xg"]].notna().to_numpy().mean()) if len(aligned) else 0.0
    )
    values = aligned.loc[:, ["home_xg", "away_xg"]].to_numpy(dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    minimum_xg = float(values.min()) if len(values) else None
    maximum_xg = float(values.max()) if len(values) else None
    side_means = (
        aligned.groupby(["Division", "SeasonStartYear"], sort=True, observed=True)[
            ["home_xg", "away_xg"]
        ]
        .mean()
        .to_numpy(dtype=np.float64)
        .reshape(-1)
    )
    side_means = side_means[np.isfinite(side_means)]
    minimum_side_mean = float(side_means.min()) if len(side_means) else None
    maximum_side_mean = float(side_means.max()) if len(side_means) else None
    complete = aligned.dropna(subset=["home_xg", "away_xg"])
    correlation: float | None = None
    if len(complete) >= 100:
        observed = np.concatenate(
            [
                complete["home_xg"].to_numpy(dtype=np.float64),
                complete["away_xg"].to_numpy(dtype=np.float64),
            ]
        )
        goals = np.concatenate(
            [
                complete["FTHome"].to_numpy(dtype=np.float64),
                complete["FTAway"].to_numpy(dtype=np.float64),
            ]
        )
        if float(np.std(observed)) > 0.0 and float(np.std(goals)) > 0.0:
            calculated = float(np.corrcoef(observed, goals)[0, 1])
            correlation = calculated if np.isfinite(calculated) else None
    passed = (
        bool(claimed)
        and minimum_coverage >= 0.98
        and value_coverage >= 0.98
        and minimum_xg is not None
        and maximum_xg is not None
        and 0.0 <= minimum_xg <= maximum_xg <= 10.0
        and minimum_side_mean is not None
        and maximum_side_mean is not None
        and 0.7 <= minimum_side_mean <= maximum_side_mean <= 2.3
        and correlation is not None
        and 0.45 <= correlation <= 0.95
    )
    return UnderstatXGQualityReport(
        covered_league_seasons=len(claimed),
        minimum_claimed_league_season_coverage=minimum_coverage,
        value_coverage=value_coverage,
        minimum_xg=minimum_xg,
        maximum_xg=maximum_xg,
        minimum_side_mean=minimum_side_mean,
        maximum_side_mean=maximum_side_mean,
        goals_correlation=correlation,
        passed=passed,
    )


def align_understat_xg(
    canonical_matches: pd.DataFrame,
    external_xg: pd.DataFrame,
    *,
    require_quality_pass: bool = True,
) -> AlignedUnderstatXG:
    """Normalize and exactly align external xG to canonical natural keys."""

    validate_canonical_frame(canonical_matches)
    schema_audit = audit_understat_xg_frame(external_xg)
    normalized = validate_understat_xg_frame(external_xg)
    covered_divisions = tuple(sorted(normalized["Division"].unique()))
    canonical = canonical_matches.loc[canonical_matches["Division"].isin(covered_divisions)].copy(
        deep=True
    )
    canonical["_home_normalized"] = canonical["HomeTeam"].map(normalize_team_name)
    canonical["_away_normalized"] = canonical["AwayTeam"].map(normalize_team_name)
    normalized = normalized.rename(
        columns={"HomeTeam": "_home_normalized", "AwayTeam": "_away_normalized"}
    )
    join_key = ("MatchDate", "Division", "_home_normalized", "_away_normalized")
    if canonical.duplicated(list(join_key), keep=False).any():
        raise UnderstatXGAlignmentError("canonical aliases collapse distinct natural keys")

    merged = canonical.merge(
        normalized.loc[:, [*join_key, "home_xg", "away_xg"]],
        on=list(join_key),
        how="left",
        sort=False,
        validate="one_to_one",
        indicator="_join_state",
    )
    merged["_aligned"] = merged["_join_state"].eq("both")
    external_probe = normalized.merge(
        canonical.loc[:, list(join_key)],
        on=list(join_key),
        how="left",
        sort=False,
        validate="one_to_one",
        indicator="_join_state",
    )
    unmatched_external = int(external_probe["_join_state"].eq("left_only").sum())
    coverage_by_division = _coverage_records(merged, ("Division",))
    coverage_by_season = _coverage_records(merged, ("Division", "SeasonStartYear"))
    evaluation_rows = merged.loc[merged["SeasonStartYear"].isin((2022, 2023, 2024, 2025))]
    coverage_by_fold = _coverage_records(
        evaluation_rows,
        ("SeasonStartYear",),
        name_prefix="test_",
    )
    quality = _quality_report(merged)
    report = UnderstatXGAlignmentReport(
        method=(
            "one_to_one exact MatchDate/Division/normalized HomeTeam/normalized AwayTeam; "
            "source-verified aliases only"
        ),
        schema=schema_audit,
        covered_divisions=covered_divisions,
        canonical_rows_in_covered_divisions=len(canonical),
        aligned_rows=int(merged["_aligned"].sum()),
        unmatched_canonical_rows=int((~merged["_aligned"]).sum()),
        unmatched_external_rows=unmatched_external,
        coverage_by_division=coverage_by_division,
        coverage_by_league_season=coverage_by_season,
        coverage_by_evaluation_fold=coverage_by_fold,
        quality=quality,
    )
    if require_quality_pass and not report.passed:
        raise UnderstatXGAlignmentError("external xG alignment failed aggregate quality gates")
    history = (
        merged.loc[:, list(ALIGNED_XG_COLUMNS)]
        .sort_values(list(UNDERSTAT_XG_KEY), kind="mergesort")
        .reset_index(drop=True)
    )
    return AlignedUnderstatXG(matches=history, report=report)
