from __future__ import annotations

import pandas as pd
import pytest

from fbai.core.leakage import LeakageViolation, validate_feature_table
from fbai.data.schema import NATURAL_KEY, NUMERIC_MATCH_COLUMNS, validate_canonical_frame
from fbai.testing.synthetic import (
    SYNTHETIC_FEATURE_COLUMNS,
    make_synthetic_canonical_matches,
    make_synthetic_fixtures,
    make_synthetic_raw_matches,
)


def test_same_seed_produces_identical_fixtures() -> None:
    first = make_synthetic_fixtures(seed=31)
    second = make_synthetic_fixtures(seed=31)

    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_change_generated_values() -> None:
    first = make_synthetic_fixtures(seed=31)
    second = make_synthetic_fixtures(seed=32)

    assert not first.equals(second)


def test_synthetic_keys_are_unique_and_complete() -> None:
    fixtures = make_synthetic_fixtures()
    key = ["Division", "MatchDate", "HomeTeam", "AwayTeam"]

    assert not fixtures[key].isna().any().any()
    assert not fixtures.duplicated(key).any()


def test_synthetic_results_use_hda_domain() -> None:
    fixtures = make_synthetic_fixtures()

    assert set(fixtures["FTR"]) <= {"H", "D", "A"}


def test_clean_synthetic_table_passes_leakage_validation() -> None:
    fixtures = make_synthetic_fixtures()

    validate_feature_table(
        fixtures,
        approved_pre_features=SYNTHETIC_FEATURE_COLUMNS,
    )


def test_forbidden_same_match_columns_are_available_only_for_guard_tests() -> None:
    fixtures = make_synthetic_fixtures(include_forbidden_same_match=True)

    assert {"FTHG", "FTAG", "HomeShots", "AwayShots"} <= set(fixtures.columns)
    with pytest.raises(LeakageViolation, match="same_match"):
        validate_feature_table(
            fixtures,
            approved_pre_features=SYNTHETIC_FEATURE_COLUMNS,
        )


@pytest.mark.parametrize("teams_per_division", [0, 1])
def test_too_few_teams_are_rejected(teams_per_division: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        make_synthetic_fixtures(teams_per_division=teams_per_division)


def test_phase_2a_raw_matches_are_deterministic() -> None:
    first = make_synthetic_raw_matches(seed=71)
    second = make_synthetic_raw_matches(seed=71)

    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(make_synthetic_raw_matches(seed=72))


def test_phase_2a_canonical_matches_have_unique_keys_and_valid_schema() -> None:
    matches = make_synthetic_canonical_matches(seed=73)

    validate_canonical_frame(matches)
    assert not matches.duplicated(list(NATURAL_KEY)).any()


def test_phase_2a_goals_and_results_are_consistent() -> None:
    matches = make_synthetic_canonical_matches(seed=74)
    expected = pd.Series("D", index=matches.index, dtype="string")
    expected.loc[matches["FTHome"] > matches["FTAway"]] = "H"
    expected.loc[matches["FTHome"] < matches["FTAway"]] = "A"

    pd.testing.assert_series_equal(matches["FTR"], expected, check_names=False)


def test_phase_2a_statistics_are_non_negative() -> None:
    matches = make_synthetic_canonical_matches(seed=75)

    assert (matches.loc[:, list(NUMERIC_MATCH_COLUMNS)] >= 0).all().all()


def test_phase_2a_generator_contains_no_real_names_or_odds() -> None:
    raw = make_synthetic_raw_matches(seed=76)

    assert raw["HomeTeam"].str.contains("Synthetic Club").all()
    assert raw["AwayTeam"].str.contains("Synthetic Club").all()
    assert not any(
        column.startswith(("B365", "Avg", "Max", "PS", "BW", "WH", "VC")) for column in raw.columns
    )


def test_phase_2a_schedule_has_same_date_batches_and_multiple_team_matches() -> None:
    matches = make_synthetic_canonical_matches(
        seed=77,
        season_start_years=(2023,),
        divisions=("SYN1",),
        teams_per_division=4,
    )
    appearances = pd.concat([matches["HomeTeam"], matches["AwayTeam"]]).value_counts()

    assert matches.groupby("MatchDate").size().gt(1).any()
    assert appearances.gt(1).all()
