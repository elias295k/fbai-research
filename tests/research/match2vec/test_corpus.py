from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbai.research.match2vec.corpus import (
    DESCRIPTOR_NAMES,
    build_match_sequence_corpus,
    build_train_vocabulary,
)
from fbai.testing.synthetic import make_synthetic_canonical_matches


@pytest.fixture
def canonical() -> pd.DataFrame:
    return make_synthetic_canonical_matches(
        seed=510,
        season_start_years=(2021, 2022, 2023),
        divisions=("SYN1", "SYN2"),
        teams_per_division=4,
    )


def _keys(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        :,
        ["MatchDate", "SeasonStartYear", "Division", "HomeTeam", "AwayTeam"],
    ]


def test_vocabulary_is_fold_local_stable_and_future_invariant(canonical: pd.DataFrame) -> None:
    train = canonical.loc[canonical["SeasonStartYear"].le(2022)]
    vocabulary = build_train_vocabulary(train)
    with_future = build_train_vocabulary(
        pd.concat(
            [
                train,
                canonical.loc[canonical["SeasonStartYear"].eq(2023)].assign(
                    HomeTeam="SYN1 Test Only Club"
                ),
            ],
            ignore_index=True,
        )
    )
    shuffled = build_train_vocabulary(train.sample(frac=1.0, random_state=7))

    assert dict(vocabulary.team_to_id) == dict(shuffled.team_to_id)
    assert ("SYN1", "SYN1 Test Only Club") not in vocabulary.team_to_id
    assert with_future.team_token_count >= vocabulary.team_token_count


def test_test_only_team_maps_to_per_league_unknown(canonical: pd.DataFrame) -> None:
    train = canonical.loc[canonical["SeasonStartYear"].le(2022)]
    vocabulary = build_train_vocabulary(train)
    test = train.iloc[[0]].copy()
    test["HomeTeam"] = "SYN1 Test Only Club"

    identifier = vocabulary.team_ids(test, side="home")[0]

    assert identifier == vocabulary.league_unknown_id[str(test.iloc[0]["Division"])]
    assert vocabulary.oov_counts(test) == (1, 0)


def test_sequence_output_is_keyed_and_shuffle_invariant(canonical: pd.DataFrame) -> None:
    targets = _keys(canonical)
    first = build_match_sequence_corpus(targets, canonical)
    shuffled = build_match_sequence_corpus(
        targets.sample(frac=1.0, random_state=9),
        canonical.sample(frac=1.0, random_state=10),
    )
    first_batch = first.batch_for(targets)
    shuffled_batch = shuffled.batch_for(targets)

    np.testing.assert_array_equal(first_batch.home_sequences, shuffled_batch.home_sequences)
    np.testing.assert_array_equal(first_batch.away_sequences, shuffled_batch.away_sequences)
    np.testing.assert_array_equal(first_batch.home_mask, shuffled_batch.home_mask)


def test_current_match_outcome_and_stats_are_excluded(canonical: pd.DataFrame) -> None:
    target = canonical.iloc[[10]]
    changed = canonical.copy(deep=True)
    changed.loc[target.index, ["FTR", "FTHome", "FTAway", "HomeShots"]] = [
        "A",
        9,
        8,
        40,
    ]
    before = build_match_sequence_corpus(_keys(target), canonical).batch_for(_keys(target))
    after = build_match_sequence_corpus(_keys(target), changed).batch_for(_keys(target))

    np.testing.assert_array_equal(before.home_sequences, after.home_sequences)
    np.testing.assert_array_equal(before.away_sequences, after.away_sequences)


def test_future_results_cannot_change_past_sequences(canonical: pd.DataFrame) -> None:
    target = canonical.iloc[[5]]
    changed = canonical.copy(deep=True)
    future = changed["MatchDate"].gt(target.iloc[0]["MatchDate"])
    changed.loc[future, "FTR"] = "H"
    changed.loc[future, "FTHome"] = 8
    before = build_match_sequence_corpus(_keys(target), canonical).batch_for(_keys(target))
    after = build_match_sequence_corpus(_keys(target), changed).batch_for(_keys(target))

    np.testing.assert_array_equal(before.home_sequences, after.home_sequences)
    np.testing.assert_array_equal(before.away_sequences, after.away_sequences)


def test_same_date_results_are_excluded_from_the_batch(canonical: pd.DataFrame) -> None:
    date = canonical.groupby("MatchDate").size().loc[lambda values: values.gt(1)].index[0]
    targets = canonical.loc[canonical["MatchDate"].eq(date)]
    changed = canonical.copy(deep=True)
    changed.loc[changed["MatchDate"].eq(date), "FTR"] = "A"
    changed.loc[changed["MatchDate"].eq(date), "FTAway"] = 7
    before = build_match_sequence_corpus(_keys(targets), canonical).batch_for(_keys(targets))
    after = build_match_sequence_corpus(_keys(targets), changed).batch_for(_keys(targets))

    np.testing.assert_array_equal(before.home_sequences, after.home_sequences)
    np.testing.assert_array_equal(before.away_sequences, after.away_sequences)


def test_descriptor_shape_padding_and_direction(canonical: pd.DataFrame) -> None:
    targets = _keys(canonical.iloc[[0, -1]])
    batch = build_match_sequence_corpus(targets, canonical).batch_for(targets)

    assert batch.home_sequences.shape == (2, 10, len(DESCRIPTOR_NAMES))
    assert not batch.home_mask[0].any()
    assert batch.home_mask[1].any()


def test_duplicate_natural_keys_fail(canonical: pd.DataFrame) -> None:
    duplicate = pd.concat([canonical, canonical.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="unique|duplicate"):
        build_match_sequence_corpus(_keys(canonical), duplicate)


def test_corpus_construction_does_not_mutate_inputs(canonical: pd.DataFrame) -> None:
    targets = _keys(canonical)
    before_targets = targets.copy(deep=True)
    before_canonical = canonical.copy(deep=True)

    build_match_sequence_corpus(targets, canonical)

    pd.testing.assert_frame_equal(targets, before_targets)
    pd.testing.assert_frame_equal(canonical, before_canonical)
