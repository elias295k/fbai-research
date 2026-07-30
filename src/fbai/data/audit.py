"""Structured integrity auditing for canonical match datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from fbai.data.schema import (
    CANONICAL_COLUMNS,
    NATURAL_KEY,
    STABLE_SORT_KEY,
    CanonicalSchemaError,
    validate_canonical_frame,
)


class AuditFailure(RuntimeError):
    """Raised when a caller requires a passing canonical audit."""


@dataclass(frozen=True)
class AuditResult:
    """Machine-readable Phase 2A audit verdict."""

    passed: bool
    input_row_count: int
    output_row_count: int
    number_of_divisions: int
    number_of_seasons: int
    duplicate_natural_keys: int
    null_natural_key_cells: int
    invalid_target_labels: int
    invalid_dates: int
    invalid_home_away_rows: int
    missing_required_columns: tuple[str, ...]
    deterministic_ordering: bool
    rows_preserved: bool
    issues: tuple[str, ...]

    def raise_for_failure(self) -> None:
        """Raise with the complete issue list when the verdict failed."""

        if not self.passed:
            raise AuditFailure("Canonical audit failed: " + "; ".join(self.issues))


def _invalid_date_count(frame: pd.DataFrame) -> int:
    if "MatchDate" not in frame:
        return 0
    values = frame["MatchDate"]
    if is_datetime64_any_dtype(values.dtype):
        return int(values.isna().sum())
    parsed = pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=True)
    return int(parsed.isna().sum())


def _is_deterministically_ordered(frame: pd.DataFrame, invalid_dates: int) -> bool:
    if invalid_dates or not set(STABLE_SORT_KEY).issubset(frame.columns):
        return False
    expected = frame.sort_values(list(STABLE_SORT_KEY), kind="mergesort").reset_index(drop=True)
    current = frame.reset_index(drop=True)
    return bool(
        current.loc[:, list(STABLE_SORT_KEY)].equals(expected.loc[:, list(STABLE_SORT_KEY)])
    )


def audit_canonical(
    frame: pd.DataFrame,
    *,
    input_row_count: int | None = None,
) -> AuditResult:
    """Audit canonical integrity without printing or modifying data."""

    expected_input_rows = len(frame) if input_row_count is None else input_row_count
    missing = tuple(column for column in CANONICAL_COLUMNS if column not in frame.columns)

    present_key = [column for column in NATURAL_KEY if column in frame]
    null_key_cells = int(frame.loc[:, present_key].isna().sum().sum()) if present_key else 0
    if set(NATURAL_KEY).issubset(frame.columns):
        duplicated = frame.loc[
            frame.duplicated(list(NATURAL_KEY), keep=False),
            list(NATURAL_KEY),
        ]
        duplicate_keys = len(duplicated.drop_duplicates())
    else:
        duplicate_keys = 0
    invalid_targets = int((~frame["FTR"].isin(["H", "D", "A"])).sum()) if "FTR" in frame else 0
    invalid_dates = _invalid_date_count(frame)
    invalid_identity = (
        int(frame["HomeTeam"].eq(frame["AwayTeam"]).sum())
        if {"HomeTeam", "AwayTeam"}.issubset(frame.columns)
        else 0
    )
    number_of_divisions = int(frame["Division"].dropna().nunique()) if "Division" in frame else 0
    number_of_seasons = (
        int(frame["SeasonStartYear"].dropna().nunique()) if "SeasonStartYear" in frame else 0
    )
    deterministic_ordering = _is_deterministically_ordered(frame, invalid_dates)
    rows_preserved = len(frame) == expected_input_rows

    issues: list[str] = []
    if missing:
        issues.append(f"missing required columns: {', '.join(missing)}")
    if not rows_preserved:
        issues.append(
            f"row preservation failed: {expected_input_rows} input vs {len(frame)} output"
        )
    if duplicate_keys:
        issues.append(f"{duplicate_keys} duplicate natural keys")
    if null_key_cells:
        issues.append(f"{null_key_cells} null natural-key cells")
    if invalid_targets:
        issues.append(f"{invalid_targets} invalid FTR labels")
    if invalid_dates:
        issues.append(f"{invalid_dates} invalid MatchDate values")
    if invalid_identity:
        issues.append(f"{invalid_identity} rows have identical home and away teams")
    if not deterministic_ordering:
        issues.append("rows are not in deterministic canonical order")

    if not missing:
        try:
            validate_canonical_frame(frame)
        except CanonicalSchemaError as exc:
            message = str(exc)
            if not any(message in issue for issue in issues):
                issues.append(message)

    return AuditResult(
        passed=not issues,
        input_row_count=expected_input_rows,
        output_row_count=len(frame),
        number_of_divisions=number_of_divisions,
        number_of_seasons=number_of_seasons,
        duplicate_natural_keys=duplicate_keys,
        null_natural_key_cells=null_key_cells,
        invalid_target_labels=invalid_targets,
        invalid_dates=invalid_dates,
        invalid_home_away_rows=invalid_identity,
        missing_required_columns=missing,
        deterministic_ordering=deterministic_ordering,
        rows_preserved=rows_preserved,
        issues=tuple(issues),
    )
