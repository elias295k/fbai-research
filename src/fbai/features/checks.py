"""Closed-schema integrity and semantic leakage checks for feature tables."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from fbai.core.leakage import (
    LeakageViolation,
    TableValidationError,
    assert_model_inputs_safe,
    assert_natural_key_valid,
    validate_result_targets,
)
from fbai.data.schema import NATURAL_KEY, STABLE_SORT_KEY
from fbai.features.schema import FEATURE_COLUMNS, FEATURE_TABLE_COLUMNS


class FeatureValidationError(ValueError):
    """Raised when a Phase 2B feature table violates its closed contract."""


def _raise_as_feature_error(operation: object) -> None:
    try:
        operation()  # type: ignore[operator]
    except (LeakageViolation, TableValidationError) as exc:
        raise FeatureValidationError(str(exc)) from exc


def _validate_exact_columns(frame: pd.DataFrame) -> None:
    actual = tuple(frame.columns)
    if actual == FEATURE_TABLE_COLUMNS:
        return
    missing = [column for column in FEATURE_TABLE_COLUMNS if column not in actual]
    extra = [column for column in actual if column not in FEATURE_TABLE_COLUMNS]
    order_only = not missing and not extra
    details: list[str] = []
    if missing:
        details.append(f"missing: {', '.join(missing)}")
    if extra:
        details.append(f"unexpected: {', '.join(extra)}")
    if order_only:
        details.append("column order differs from the feature-table contract")
    raise FeatureValidationError("Feature-table columns are invalid (" + "; ".join(details) + ")")


def _validate_targets(frame: pd.DataFrame) -> None:
    target_columns = (
        "FTR",
        "target_1x2",
        "target_home_win",
        "target_draw",
        "target_away_win",
    )
    _raise_as_feature_error(lambda: validate_result_targets(frame, target_columns=target_columns))
    over_under = pd.to_numeric(frame["target_ou25"], errors="coerce")
    invalid_over_under = over_under.notna() & ~over_under.isin([0, 1])
    if bool(invalid_over_under.any()):
        raise FeatureValidationError("target_ou25 must contain only 0/1 or missing values")

    if not frame["target_1x2"].eq(frame["FTR"]).all():
        raise FeatureValidationError("target_1x2 must equal FTR on every row")
    expected = {
        "target_home_win": frame["FTR"].eq("H").astype("Int8"),
        "target_draw": frame["FTR"].eq("D").astype("Int8"),
        "target_away_win": frame["FTR"].eq("A").astype("Int8"),
    }
    for column, values in expected.items():
        if not frame[column].astype("Int8").equals(values):
            raise FeatureValidationError(f"{column} is inconsistent with FTR")


def validate_feature_table(
    frame: pd.DataFrame,
    *,
    expected_row_count: int | None = None,
) -> None:
    """Require the exact ordered metadata, target, and 52-feature schema."""

    if not isinstance(frame, pd.DataFrame):
        raise FeatureValidationError("Feature table must be a pandas DataFrame")
    if frame.empty:
        raise FeatureValidationError("Feature table must contain at least one match")
    _validate_exact_columns(frame)
    if expected_row_count is not None and len(frame) != expected_row_count:
        raise FeatureValidationError(
            f"Row preservation failed: expected {expected_row_count}, found {len(frame)}"
        )

    _raise_as_feature_error(lambda: assert_natural_key_valid(frame, key_columns=NATURAL_KEY))
    _raise_as_feature_error(
        lambda: assert_model_inputs_safe(
            FEATURE_COLUMNS,
            approved_pre_features=FEATURE_COLUMNS,
        )
    )
    _validate_targets(frame)

    if frame.loc[:, list(NATURAL_KEY)].isna().any().any():
        raise FeatureValidationError("Natural-key cells must not be null")
    if frame["SeasonStartYear"].isna().any():
        raise FeatureValidationError("SeasonStartYear must not be null")
    if not is_datetime64_any_dtype(frame["MatchDate"].dtype):
        raise FeatureValidationError("MatchDate must use a pandas datetime dtype")
    if frame["MatchDate"].isna().any():
        raise FeatureValidationError("MatchDate must not contain null values")

    expected_order = frame.sort_values(list(STABLE_SORT_KEY), kind="mergesort").reset_index(
        drop=True
    )
    actual_key = frame.reset_index(drop=True).loc[:, list(STABLE_SORT_KEY)]
    expected_key = expected_order.loc[:, list(STABLE_SORT_KEY)]
    if not actual_key.equals(expected_key):
        raise FeatureValidationError(
            "Rows must be ordered by MatchDate, Division, HomeTeam, AwayTeam"
        )

    for column in FEATURE_COLUMNS:
        if not is_numeric_dtype(frame[column].dtype):
            raise FeatureValidationError(f"Model feature {column} must be numeric")
    numeric = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(numeric).any():
        raise FeatureValidationError("Model features must not contain infinite values")


def select_model_input_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Validate ``frame`` and return only the approved 52 feature names."""

    validate_feature_table(frame)
    assert_model_inputs_safe(FEATURE_COLUMNS, approved_pre_features=FEATURE_COLUMNS)
    return FEATURE_COLUMNS


def assert_selected_features_safe(feature_names: Sequence[str]) -> None:
    """Expose the semantic guard for a proposed fit-time selection."""

    assert_model_inputs_safe(feature_names, approved_pre_features=FEATURE_COLUMNS)
