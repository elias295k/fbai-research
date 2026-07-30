"""Semantic leakage guards and feature-table validation.

The guards classify a column by what it means, not merely by its suffix. A
same-match field therefore remains forbidden even if it is renamed with a
``_pre`` suffix.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum

import pandas as pd

NATURAL_KEY: tuple[str, ...] = ("Division", "MatchDate", "HomeTeam", "AwayTeam")


class LeakageViolation(ValueError):
    """Raised when model inputs contain a leakage-prone or unapproved field."""


class TableValidationError(ValueError):
    """Raised when a feature table violates its schema or key contract."""


class ColumnRole(StrEnum):
    """Semantic roles used by the leakage policy."""

    APPROVED_PRE = "approved_pre"
    UNAPPROVED_PRE = "unapproved_pre"
    SAME_MATCH = "same_match"
    LABEL = "label"
    METADATA = "metadata"
    ODDS = "odds"
    UNKNOWN = "unknown"


_SAME_MATCH_FIELDS = {
    "ac",
    "af",
    "ar",
    "as",
    "ast",
    "awaycorners",
    "awayfouls",
    "awaygoals",
    "awayred",
    "awayredcards",
    "awayshots",
    "awayshotsontarget",
    "awaytarget",
    "awayyellow",
    "awayyellowcards",
    "attendance",
    "ftaway",
    "fthome",
    "fthg",
    "ftag",
    "hc",
    "hf",
    "homecorners",
    "homefouls",
    "homegoals",
    "homered",
    "homeredcards",
    "homeshots",
    "homeshotsontarget",
    "hometarget",
    "homeyellow",
    "homeyellowcards",
    "hr",
    "hs",
    "hst",
    "htaway",
    "hthome",
    "htag",
    "hthg",
    "referee",
    "sourceawayxg",
    "sourcehomexg",
    "ay",
    "hy",
}
_LABEL_FIELDS = {
    "ftr",
    "ftresult",
    "htr",
    "htresult",
    "target1x2",
    "targeta",
    "targetawaywin",
    "targetd",
    "targetdraw",
    "targeth",
    "targetclass",
    "targethomewin",
    "targetou25",
    "targetresult",
}
_METADATA_FIELDS = {
    "awayteam",
    "date",
    "division",
    "hometeam",
    "ingestedat",
    "leaguename",
    "matchdate",
    "matchid",
    "season",
    "seasonstartyear",
    "sourceurl",
    "time",
}
_ODDS_PREFIXES = (
    "ah",
    "avg",
    "b365",
    "bet",
    "bf",
    "bw",
    "gb",
    "handi",
    "iw",
    "lb",
    "max",
    "mkt",
    "p25",
    "pinnacle",
    "ps",
    "sb",
    "sj",
    "vc",
    "wh",
)


def _normalize(column: str) -> str:
    return "".join(character for character in column.lower() if character.isalnum())


def _semantic_base(column: str) -> str:
    return column[:-4] if column.lower().endswith("_pre") else column


def _is_odds_field(normalized: str) -> bool:
    return normalized.startswith(_ODDS_PREFIXES)


def _validate_approval_set(approved_pre_features: Iterable[str]) -> frozenset[str]:
    approved = frozenset(approved_pre_features)
    invalid: list[str] = []
    for column in approved:
        if not column.lower().endswith("_pre"):
            invalid.append(column)
            continue
        base = _normalize(_semantic_base(column))
        if (
            base in _SAME_MATCH_FIELDS
            or base in _LABEL_FIELDS
            or base in _METADATA_FIELDS
            or _is_odds_field(base)
        ):
            invalid.append(column)
    if invalid:
        joined = ", ".join(sorted(invalid))
        raise LeakageViolation(f"Approval list contains semantically unsafe fields: {joined}")
    return approved


def classify_column(
    column: str,
    *,
    approved_pre_features: Iterable[str] = (),
) -> ColumnRole:
    """Classify a column while preventing suffix-based semantic bypasses."""

    approved = _validate_approval_set(approved_pre_features)
    base = _normalize(_semantic_base(column))
    if base in _SAME_MATCH_FIELDS:
        return ColumnRole.SAME_MATCH
    if base in _LABEL_FIELDS:
        return ColumnRole.LABEL
    if base in _METADATA_FIELDS:
        return ColumnRole.METADATA
    if _is_odds_field(base):
        return ColumnRole.ODDS
    if column in approved:
        return ColumnRole.APPROVED_PRE
    if column.lower().endswith("_pre"):
        return ColumnRole.UNAPPROVED_PRE
    return ColumnRole.UNKNOWN


def assert_model_inputs_safe(
    feature_columns: Sequence[str],
    *,
    approved_pre_features: Iterable[str],
) -> None:
    """Require every model input to be an explicitly approved pre-match feature."""

    if len(feature_columns) != len(set(feature_columns)):
        raise LeakageViolation("Model input list contains duplicate columns")

    approved = _validate_approval_set(approved_pre_features)
    violations: dict[str, ColumnRole] = {}
    for column in feature_columns:
        role = classify_column(column, approved_pre_features=approved)
        if role is not ColumnRole.APPROVED_PRE:
            violations[column] = role
    if violations:
        details = ", ".join(
            f"{column} ({role.value})" for column, role in sorted(violations.items())
        )
        raise LeakageViolation(f"Unsafe or unapproved model inputs: {details}")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise TableValidationError(f"Missing required columns: {', '.join(missing)}")


def assert_natural_key_valid(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str] = NATURAL_KEY,
) -> None:
    """Require complete, unique match identity columns."""

    _require_columns(frame, key_columns)
    null_columns = [column for column in key_columns if frame[column].isna().any()]
    if null_columns:
        raise TableValidationError(
            f"Natural key contains null values in: {', '.join(null_columns)}"
        )
    duplicate_mask = frame.duplicated(subset=list(key_columns), keep=False)
    if duplicate_mask.any():
        count = int(duplicate_mask.sum())
        raise TableValidationError(f"Natural key is not unique; {count} rows are duplicated")


def validate_result_targets(
    frame: pd.DataFrame,
    *,
    target_columns: Sequence[str] = ("FTR",),
) -> None:
    """Validate H/D/A labels and optional one-hot result targets."""

    _require_columns(frame, target_columns)
    for column in target_columns:
        if frame[column].isna().any():
            raise TableValidationError(f"Target column {column} contains null values")
        normalized = _normalize(column)
        if normalized in {
            "ftr",
            "ftresult",
            "target1x2",
            "targetclass",
            "targetresult",
        }:
            invalid = sorted(set(frame[column].astype(str)).difference({"H", "D", "A"}))
            if invalid:
                raise TableValidationError(
                    f"Target column {column} contains values outside H/D/A: {invalid}"
                )
        elif normalized in {
            "targeta",
            "targetawaywin",
            "targetd",
            "targetdraw",
            "targeth",
            "targethomewin",
            "targetou25",
        }:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.isna().any() or not numeric.isin([0, 1]).all():
                raise TableValidationError(f"One-hot target {column} must contain only 0/1")
        else:
            raise TableValidationError(f"Unsupported result target column: {column}")

    for group in (
        ("target_H", "target_D", "target_A"),
        ("target_home_win", "target_draw", "target_away_win"),
    ):
        one_hot = [column for column in group if column in target_columns]
        if one_hot:
            if len(one_hot) != 3:
                raise TableValidationError(
                    f"One-hot targets must include the complete group: {', '.join(group)}"
                )
            row_sums = frame[one_hot].sum(axis=1)
            if not row_sums.eq(1).all():
                raise TableValidationError("One-hot result targets must sum to one on every row")


def validate_feature_table(
    frame: pd.DataFrame,
    *,
    approved_pre_features: Iterable[str],
    key_columns: Sequence[str] = NATURAL_KEY,
    target_columns: Sequence[str] = ("FTR",),
) -> None:
    """Validate keys, targets, and the semantic role of all remaining columns."""

    approved = _validate_approval_set(approved_pre_features)
    assert_natural_key_valid(frame, key_columns=key_columns)
    validate_result_targets(frame, target_columns=target_columns)

    exempt = set(key_columns).union(target_columns)
    violations: dict[str, ColumnRole] = {}
    for column in frame.columns:
        if column in exempt:
            continue
        role = classify_column(column, approved_pre_features=approved)
        if role not in {ColumnRole.APPROVED_PRE, ColumnRole.METADATA}:
            violations[column] = role
    if violations:
        details = ", ".join(
            f"{column} ({role.value})" for column, role in sorted(violations.items())
        )
        raise LeakageViolation(f"Feature table contains unsafe or unknown columns: {details}")


def select_model_input_columns(
    frame: pd.DataFrame,
    *,
    approved_pre_features: Iterable[str],
    key_columns: Sequence[str] = NATURAL_KEY,
    target_columns: Sequence[str] = ("FTR",),
) -> list[str]:
    """Validate a feature table and return approved features in table order."""

    approved = _validate_approval_set(approved_pre_features)
    validate_feature_table(
        frame,
        approved_pre_features=approved,
        key_columns=key_columns,
        target_columns=target_columns,
    )
    return [column for column in frame.columns if column in approved]
