from __future__ import annotations

import pandas as pd
import pytest

from fbai.core.leakage import LeakageViolation, validate_feature_table
from fbai.testing.synthetic import SYNTHETIC_FEATURE_COLUMNS, make_synthetic_fixtures


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
