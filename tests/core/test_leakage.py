from __future__ import annotations

import pandas as pd
import pytest

from fbai.core.leakage import (
    LeakageViolation,
    TableValidationError,
    assert_model_inputs_safe,
    select_model_input_columns,
    validate_feature_table,
)

APPROVED = ("HomeRating_pre", "AwayRating_pre", "RatingDiff_pre")


def feature_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Division": ["SYN1", "SYN1"],
            "MatchDate": pd.to_datetime(["2024-08-01", "2024-08-08"]),
            "HomeTeam": ["Synthetic A", "Synthetic C"],
            "AwayTeam": ["Synthetic B", "Synthetic D"],
            "SeasonStartYear": [2024, 2024],
            "HomeRating_pre": [0.5, -0.1],
            "AwayRating_pre": [0.1, 0.2],
            "RatingDiff_pre": [0.4, -0.3],
            "FTR": ["H", "A"],
        }
    )


@pytest.mark.parametrize(
    "column",
    [
        "FTHG",
        "FTAG",
        "FTHome",
        "FTAway",
        "HomeGoals",
        "AwayGoals",
        "HomeShots",
        "AwayShots",
        "HomeTarget",
        "AwayTarget",
        "source_home_xg",
        "source_away_xg",
    ],
)
def test_same_match_fields_are_rejected(column: str) -> None:
    with pytest.raises(LeakageViolation, match="same_match"):
        assert_model_inputs_safe([column], approved_pre_features=APPROVED)


@pytest.mark.parametrize(
    "column",
    ["FTR", "HTR", "target_H", "target_1x2", "target_home_win"],
)
def test_label_fields_are_rejected_as_model_inputs(column: str) -> None:
    with pytest.raises(LeakageViolation, match="label"):
        assert_model_inputs_safe([column], approved_pre_features=APPROVED)


@pytest.mark.parametrize("column", ["B365H", "AvgD", "PSA", "AHh", "HandiHome"])
def test_odds_fields_are_rejected_by_default(column: str) -> None:
    with pytest.raises(LeakageViolation, match="odds"):
        assert_model_inputs_safe([column], approved_pre_features=APPROVED)


@pytest.mark.parametrize(
    "column",
    [
        "HomeShots_pre",
        "FTR_pre",
        "AvgH_pre",
        "AHh_pre",
        "MatchDate_pre",
        "source_url_pre",
    ],
)
def test_pre_suffix_cannot_bypass_semantic_policy(column: str) -> None:
    with pytest.raises(LeakageViolation):
        assert_model_inputs_safe([column], approved_pre_features=[column])


def test_explicitly_approved_pre_match_features_are_selected_in_table_order() -> None:
    table = feature_table()

    selected = select_model_input_columns(table, approved_pre_features=APPROVED)

    assert selected == list(APPROVED)


def test_unapproved_pre_match_feature_is_rejected() -> None:
    with pytest.raises(LeakageViolation, match="unapproved_pre"):
        assert_model_inputs_safe(["RecentWins_pre"], approved_pre_features=APPROVED)


def test_unknown_unsuffixed_numeric_column_is_rejected() -> None:
    table = feature_table().assign(MysteryRating=[1.0, 2.0])

    with pytest.raises(LeakageViolation, match="unknown"):
        validate_feature_table(table, approved_pre_features=APPROVED)


def test_duplicate_natural_keys_are_rejected() -> None:
    table = pd.concat([feature_table(), feature_table().iloc[[0]]], ignore_index=True)

    with pytest.raises(TableValidationError, match="not unique"):
        validate_feature_table(table, approved_pre_features=APPROVED)


def test_null_natural_key_is_rejected() -> None:
    table = feature_table()
    table.loc[0, "HomeTeam"] = None

    with pytest.raises(TableValidationError, match="null"):
        validate_feature_table(table, approved_pre_features=APPROVED)


def test_invalid_result_domain_is_rejected() -> None:
    table = feature_table()
    table.loc[0, "FTR"] = "W"

    with pytest.raises(TableValidationError, match="outside H/D/A"):
        validate_feature_table(table, approved_pre_features=APPROVED)


def test_duplicate_model_inputs_are_rejected() -> None:
    with pytest.raises(LeakageViolation, match="duplicate"):
        assert_model_inputs_safe(
            ["HomeRating_pre", "HomeRating_pre"],
            approved_pre_features=APPROVED,
        )


def test_metadata_is_allowed_in_table_but_not_in_model_inputs() -> None:
    validate_feature_table(feature_table(), approved_pre_features=APPROVED)

    with pytest.raises(LeakageViolation, match="metadata"):
        assert_model_inputs_safe(["SeasonStartYear"], approved_pre_features=APPROVED)
