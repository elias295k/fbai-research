from __future__ import annotations

import hashlib

import pandas as pd

from fbai.data.schema import NATURAL_KEY
from fbai.features.schema import FEATURE_COLUMNS, feature_columns
from fbai.research.pseudo_xg.features import (
    PSEUDO_XG_FEATURE_COLUMNS,
    build_pseudo_xg_feature_table,
    build_walk_forward_training_feature_table,
)
from fbai.research.pseudo_xg.model import fit_pseudo_xg_estimator
from fbai.testing.synthetic import make_synthetic_canonical_matches


def _canonical() -> pd.DataFrame:
    return make_synthetic_canonical_matches(
        seed=532,
        season_start_years=(2020, 2021, 2022),
        divisions=("SYN1", "SYN2"),
        teams_per_division=4,
    )


def _keys(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, [*NATURAL_KEY, "SeasonStartYear"]].copy(deep=True)


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(list(NATURAL_KEY), kind="mergesort").reset_index(drop=True)


def test_feature_names_count_and_stable_contract_fingerprint_are_explicit() -> None:
    assert len(PSEUDO_XG_FEATURE_COLUMNS) == 12
    assert len(set(PSEUDO_XG_FEATURE_COLUMNS)) == 12
    assert all(column.endswith("_pxg_pre") for column in PSEUDO_XG_FEATURE_COLUMNS)
    assert set(PSEUDO_XG_FEATURE_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert feature_columns() == FEATURE_COLUMNS
    assert hashlib.sha256("\0".join(feature_columns()).encode()).hexdigest() == (
        "09b9204c283e8adf7e91e98cb1a547183b3f832cc95f815b9845208be822de78"
    )


def test_shuffled_inputs_produce_identical_keyed_output() -> None:
    canonical = _canonical()
    fitted = fit_pseudo_xg_estimator(canonical.loc[canonical["SeasonStartYear"].le(2021)])
    expected = build_pseudo_xg_feature_table(_keys(canonical), canonical, fitted)
    actual = build_pseudo_xg_feature_table(
        _keys(canonical).sample(frac=1, random_state=10),
        canonical.sample(frac=1, random_state=11),
        fitted,
    )

    pd.testing.assert_frame_equal(_keyed(actual), _keyed(expected))


def test_current_match_statistics_do_not_affect_own_features() -> None:
    canonical = _canonical()
    train = canonical.loc[canonical["SeasonStartYear"].le(2020)]
    fitted = fit_pseudo_xg_estimator(train)
    target = canonical.loc[canonical["SeasonStartYear"].eq(2021)].iloc[[5]]
    changed = canonical.copy(deep=True)
    target_keys = pd.MultiIndex.from_frame(target.loc[:, list(NATURAL_KEY)])
    mask = pd.MultiIndex.from_frame(changed.loc[:, list(NATURAL_KEY)]).isin(target_keys)
    changed.loc[mask, ["HomeShots", "AwayShots"]] += 100

    before = build_pseudo_xg_feature_table(_keys(target), canonical, fitted)
    after = build_pseudo_xg_feature_table(_keys(target), changed, fitted)

    pd.testing.assert_frame_equal(before, after)


def test_future_rows_cannot_change_past_features() -> None:
    canonical = _canonical()
    fitted = fit_pseudo_xg_estimator(canonical.loc[canonical["SeasonStartYear"].le(2020)])
    target = canonical.loc[canonical["SeasonStartYear"].eq(2021)].iloc[[5]]
    changed = canonical.copy(deep=True)
    future = changed["MatchDate"].gt(target["MatchDate"].iloc[0])
    changed.loc[future, ["HomeShots", "AwayShots"]] += 100

    before = build_pseudo_xg_feature_table(_keys(target), canonical, fitted)
    after = build_pseudo_xg_feature_table(_keys(target), changed, fitted)

    pd.testing.assert_frame_equal(before, after)


def test_same_date_matches_are_one_information_batch() -> None:
    canonical = _canonical()
    fitted = fit_pseudo_xg_estimator(canonical.loc[canonical["SeasonStartYear"].le(2020)])
    date = canonical.groupby("MatchDate").size().loc[lambda value: value.gt(1)].index[-2]
    targets = canonical.loc[canonical["MatchDate"].eq(date)]
    changed = canonical.copy(deep=True)
    same_date = changed["MatchDate"].eq(date)
    changed.loc[same_date, ["HomeShots", "AwayShots"]] += 100

    before = build_pseudo_xg_feature_table(_keys(targets), canonical, fitted)
    after = build_pseudo_xg_feature_table(_keys(targets), changed, fitted)

    pd.testing.assert_frame_equal(before, after)


def test_history_is_isolated_by_division_and_season() -> None:
    canonical = _canonical()
    fitted = fit_pseudo_xg_estimator(canonical.loc[canonical["SeasonStartYear"].le(2020)])
    target = canonical.loc[
        canonical["Division"].eq("SYN1") & canonical["SeasonStartYear"].eq(2021)
    ].iloc[[5]]
    changed = canonical.copy(deep=True)
    other_division_past = changed["Division"].eq("SYN2") & changed["MatchDate"].lt(
        target["MatchDate"].iloc[0]
    )
    changed.loc[other_division_past, ["HomeShots", "AwayShots"]] += 100

    before = build_pseudo_xg_feature_table(_keys(target), canonical, fitted)
    after = build_pseudo_xg_feature_table(_keys(target), changed, fitted)
    pd.testing.assert_frame_equal(before, after)

    first_by_season = (
        canonical.loc[canonical["SeasonStartYear"].eq(2022)]
        .sort_values(list(NATURAL_KEY))
        .iloc[[0]]
    )
    first_features = build_pseudo_xg_feature_table(_keys(first_by_season), canonical, fitted)
    assert first_features.loc[:, list(PSEUDO_XG_FEATURE_COLUMNS)].isna().all().all()


def test_walk_forward_training_features_are_current_and_future_invariant() -> None:
    canonical = _canonical()
    before = build_walk_forward_training_feature_table(canonical)
    changed = canonical.copy(deep=True)
    changed.loc[changed["SeasonStartYear"].eq(2022), ["HomeShots", "AwayShots"]] += 100
    after = build_walk_forward_training_feature_table(changed)
    earlier = before["MatchDate"].lt(
        canonical.loc[canonical["SeasonStartYear"].eq(2022), "MatchDate"].min()
    )

    pd.testing.assert_frame_equal(
        before.loc[earlier].reset_index(drop=True),
        after.loc[earlier].reset_index(drop=True),
    )

    current = canonical.loc[canonical["SeasonStartYear"].eq(2021)].iloc[[5]]
    current_keys = pd.MultiIndex.from_frame(current.loc[:, list(NATURAL_KEY)])
    current_mask = pd.MultiIndex.from_frame(changed.loc[:, list(NATURAL_KEY)]).isin(current_keys)
    changed_current = canonical.copy(deep=True)
    changed_current.loc[current_mask, ["HomeShots", "AwayShots"]] += 100
    after_current = build_walk_forward_training_feature_table(changed_current)
    output_mask = pd.MultiIndex.from_frame(before.loc[:, list(NATURAL_KEY)]).isin(current_keys)
    pd.testing.assert_frame_equal(
        before.loc[output_mask].reset_index(drop=True),
        after_current.loc[output_mask].reset_index(drop=True),
    )
