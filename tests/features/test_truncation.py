from __future__ import annotations

import pandas as pd
import pytest

from fbai.data.schema import NATURAL_KEY, CanonicalSchemaError
from fbai.features.build import build_feature_table
from fbai.features.schema import FEATURE_COLUMNS
from fbai.testing.synthetic import make_synthetic_canonical_matches


def canonical() -> pd.DataFrame:
    return make_synthetic_canonical_matches(
        seed=303,
        season_start_years=(2022, 2023),
        divisions=("SYN1", "SYN2"),
        teams_per_division=6,
    )


def feature_view(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values(list(NATURAL_KEY), kind="mergesort")
        .reset_index(drop=True)
        .loc[:, [*NATURAL_KEY, *FEATURE_COLUMNS]]
    )


def assert_same_features(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        feature_view(left),
        feature_view(right),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_full_and_truncated_builds_match_for_every_shared_past_row() -> None:
    source = canonical()
    cutoff = source["MatchDate"].sort_values().unique()[8]

    full = build_feature_table(source)
    truncated = build_feature_table(source.loc[source["MatchDate"] <= cutoff])

    assert_same_features(full.loc[full["MatchDate"] <= cutoff], truncated)


def test_appended_or_modified_future_rows_cannot_change_past_features() -> None:
    source = canonical()
    cutoff = source["MatchDate"].sort_values().unique()[8]
    past = source.loc[source["MatchDate"] <= cutoff].copy()
    full = build_feature_table(source)

    modified = source.copy()
    future_mask = modified["MatchDate"] > cutoff
    modified.loc[future_mask, "FTR"] = modified.loc[future_mask, "FTR"].map(
        {"H": "A", "A": "H", "D": "H"}
    )
    modified.loc[future_mask, "FTHome"] += 20
    modified.loc[future_mask, "HomeShots"] += 20
    modified_features = build_feature_table(modified)
    past_only = build_feature_table(past)

    assert_same_features(full.loc[full["MatchDate"] <= cutoff], past_only)
    assert_same_features(
        full.loc[full["MatchDate"] <= cutoff],
        modified_features.loc[modified_features["MatchDate"] <= cutoff],
    )


def test_current_match_goals_and_statistics_do_not_change_own_features() -> None:
    source = canonical()
    key = source.loc[len(source) // 2, list(NATURAL_KEY)]
    baseline = build_feature_table(source)
    changed = source.copy()
    mask = pd.Series(True, index=changed.index)
    for column in NATURAL_KEY:
        mask &= changed[column].eq(key[column])
    for column in (
        "FTHome",
        "FTAway",
        "HomeShots",
        "AwayShots",
        "HomeTarget",
        "AwayTarget",
        "HomeCorners",
        "AwayCorners",
        "HomeFouls",
        "AwayFouls",
        "HomeYellow",
        "AwayYellow",
        "HomeRed",
        "AwayRed",
    ):
        changed.loc[mask, column] += 50
    rebuilt = build_feature_table(changed)

    baseline_row = baseline.loc[mask.to_numpy(), list(FEATURE_COLUMNS)].reset_index(drop=True)
    rebuilt_key_mask = pd.Series(True, index=rebuilt.index)
    for column in NATURAL_KEY:
        rebuilt_key_mask &= rebuilt[column].eq(key[column])
    rebuilt_row = rebuilt.loc[rebuilt_key_mask, list(FEATURE_COLUMNS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        baseline_row,
        rebuilt_row,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_same_date_changes_do_not_change_other_rows_in_date_batch() -> None:
    source = canonical()
    batch_date = source.groupby("MatchDate").size().loc[lambda values: values > 1].index[3]
    same_date_indexes = source.index[source["MatchDate"].eq(batch_date)].tolist()
    changed_index = same_date_indexes[0]
    comparison_indexes = same_date_indexes[1:]
    baseline = build_feature_table(source)
    changed = source.copy()
    changed.loc[changed_index, "FTR"] = "A" if changed.loc[changed_index, "FTR"] != "A" else "H"
    changed.loc[changed_index, "FTHome"] += 100
    changed.loc[changed_index, "HomeShots"] += 100
    rebuilt = build_feature_table(changed)
    comparison_keys = source.loc[comparison_indexes, list(NATURAL_KEY)]

    baseline_rows = baseline.merge(comparison_keys, on=list(NATURAL_KEY), how="inner")
    rebuilt_rows = rebuilt.merge(comparison_keys, on=list(NATURAL_KEY), how="inner")
    assert_same_features(baseline_rows, rebuilt_rows)


def test_full_and_in_date_shuffles_are_deterministic() -> None:
    source = canonical()
    fully_shuffled = source.sample(frac=1.0, random_state=55)
    date_shuffled = (
        source.groupby("MatchDate", group_keys=False, sort=True)
        .sample(frac=1.0, random_state=56)
        .reset_index(drop=True)
    )

    expected = build_feature_table(source)

    pd.testing.assert_frame_equal(build_feature_table(fully_shuffled), expected)
    pd.testing.assert_frame_equal(build_feature_table(date_shuffled), expected)


def test_duplicate_natural_key_is_rejected() -> None:
    source = canonical()
    duplicate = pd.concat([source, source.iloc[[0]]], ignore_index=True)

    with pytest.raises(CanonicalSchemaError, match="not unique"):
        build_feature_table(duplicate)
