from __future__ import annotations

import pandas as pd
import pytest

from fbai.core.splits import (
    ALL_TEST_YEARS,
    FoldRole,
    chronological_date_batches,
    expanding_folds,
    inner_time_split,
    sort_matches,
)
from fbai.testing.synthetic import make_synthetic_fixtures


def fixtures() -> pd.DataFrame:
    return make_synthetic_fixtures(divisions=("SYN1",), teams_per_division=4)


def test_expanding_folds_have_strict_train_before_test_dates() -> None:
    matches = fixtures()

    folds = list(expanding_folds(matches))

    assert [fold.test_year for fold in folds] == list(ALL_TEST_YEARS)
    for fold in folds:
        train = matches.loc[list(fold.train_idx)]
        test = matches.loc[list(fold.test_idx)]
        assert train["MatchDate"].max() < test["MatchDate"].min()
        assert set(train.index).isdisjoint(test.index)


def test_2025_fold_is_labeled_historical_final() -> None:
    folds = list(expanding_folds(fixtures()))

    assert [fold.role for fold in folds[:-1]] == [FoldRole.DEVELOPMENT] * 3
    assert folds[-1].role is FoldRole.HISTORICAL_FINAL
    assert folds[-1].test_year == 2025


def test_season_label_cannot_hide_date_overlap() -> None:
    matches = fixtures()
    training_row = matches.index[matches["SeasonStartYear"].eq(2021)][0]
    matches.loc[training_row, "MatchDate"] = pd.Timestamp("2023-01-01")

    with pytest.raises(ValueError, match="violates chronology"):
        list(expanding_folds(matches, test_years=(2022,)))


def test_shuffled_input_produces_identical_fold_indices() -> None:
    matches = fixtures()
    shuffled = matches.sample(frac=1.0, random_state=17)

    original = list(expanding_folds(matches))
    reordered = list(expanding_folds(shuffled))

    assert original == reordered


def test_inner_split_is_strict_and_keeps_dates_indivisible() -> None:
    matches = fixtures().loc[lambda frame: frame["SeasonStartYear"] <= 2023]

    fit_idx, validation_idx = inner_time_split(matches, validation_fraction=0.25)

    fit_dates = matches.loc[list(fit_idx), "MatchDate"]
    validation_dates = matches.loc[list(validation_idx), "MatchDate"]
    assert fit_dates.max() < validation_dates.min()
    assert set(fit_dates).isdisjoint(set(validation_dates))


def test_inner_split_rejects_a_single_date() -> None:
    matches = fixtures()
    one_date = matches.loc[matches["MatchDate"].eq(matches["MatchDate"].min())]

    with pytest.raises(ValueError, match="two distinct dates"):
        inner_time_split(one_date)


def test_sort_matches_uses_stable_tie_breakers() -> None:
    matches = fixtures().sample(frac=1.0, random_state=5)

    sorted_matches = sort_matches(matches)
    expected = sorted_matches.sort_values(
        ["MatchDate", "Division", "HomeTeam", "AwayTeam"],
        kind="mergesort",
    )

    pd.testing.assert_frame_equal(sorted_matches, expected)


def test_date_batches_contain_every_match_from_the_date() -> None:
    matches = fixtures()

    batches = chronological_date_batches(matches)

    flattened = [index for batch in batches for index in batch]
    assert len(flattened) == len(matches)
    assert len(flattened) == len(set(flattened))
    for batch in batches:
        assert matches.loc[list(batch), "MatchDate"].nunique() == 1
        batch_date = matches.loc[batch[0], "MatchDate"]
        assert set(batch) == set(matches.index[matches["MatchDate"].eq(batch_date)])


def test_duplicate_input_index_is_rejected() -> None:
    matches = fixtures()
    matches.index = [0] * len(matches)

    with pytest.raises(ValueError, match="index must be unique"):
        sort_matches(matches)
