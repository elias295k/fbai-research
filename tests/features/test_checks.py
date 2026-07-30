from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbai.core.leakage import LeakageViolation
from fbai.features.build import build_feature_table
from fbai.features.checks import (
    FeatureValidationError,
    assert_selected_features_safe,
    select_model_input_columns,
    validate_feature_table,
)
from fbai.features.schema import FEATURE_COLUMNS
from fbai.testing.synthetic import make_synthetic_canonical_matches


def safe_table() -> pd.DataFrame:
    canonical = make_synthetic_canonical_matches(
        seed=101,
        season_start_years=(2023,),
        divisions=("E0",),
        teams_per_division=4,
    )
    return build_feature_table(canonical)


def test_safe_table_passes_and_selector_returns_only_approved_features() -> None:
    frame = safe_table()

    validate_feature_table(frame)

    assert select_model_input_columns(frame) == FEATURE_COLUMNS


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns=FEATURE_COLUMNS[-1]), "missing"),
        (lambda frame: frame.assign(Unknown_pre=1.0), "unexpected"),
        (lambda frame: frame.assign(UnknownNumeric=1.0), "unexpected"),
        (lambda frame: frame.assign(HomeShots_pre=1.0), "unexpected"),
        (lambda frame: frame.assign(AvgH_pre=1.0), "unexpected"),
    ],
)
def test_closed_schema_rejects_missing_or_unknown_columns(
    mutation: object,
    message: str,
) -> None:
    frame = mutation(safe_table())  # type: ignore[operator]

    with pytest.raises(FeatureValidationError, match=message):
        validate_feature_table(frame)


def test_semantic_guard_rejects_label_same_match_and_odds_selections() -> None:
    for unsafe in ("target_1x2", "HomeShots_pre", "AvgH_pre"):
        with pytest.raises(LeakageViolation):
            assert_selected_features_safe([unsafe])


def test_infinite_feature_duplicate_key_null_key_and_invalid_target_fail() -> None:
    infinite = safe_table()
    infinite.loc[0, FEATURE_COLUMNS[0]] = np.inf
    with pytest.raises(FeatureValidationError, match="infinite"):
        validate_feature_table(infinite)

    duplicate = pd.concat([safe_table(), safe_table().iloc[[0]]], ignore_index=True)
    with pytest.raises(FeatureValidationError, match="not unique"):
        validate_feature_table(duplicate)

    null_key = safe_table()
    null_key.loc[0, "HomeTeam"] = pd.NA
    with pytest.raises(FeatureValidationError, match="null"):
        validate_feature_table(null_key)

    invalid_target = safe_table()
    invalid_target.loc[0, "target_1x2"] = "X"
    with pytest.raises(FeatureValidationError, match="outside H/D/A"):
        validate_feature_table(invalid_target)


def test_incorrect_row_order_and_column_order_fail() -> None:
    frame = safe_table()
    shuffled = frame.sample(frac=1.0, random_state=17).reset_index(drop=True)
    with pytest.raises(FeatureValidationError, match="Rows must be ordered"):
        validate_feature_table(shuffled)

    reordered = frame.loc[:, list(reversed(frame.columns))]
    with pytest.raises(FeatureValidationError, match="column order"):
        validate_feature_table(reordered)


def test_missing_over_under_label_is_valid_when_goals_are_missing() -> None:
    frame = safe_table()
    frame.loc[0, "target_ou25"] = pd.NA

    validate_feature_table(frame)
