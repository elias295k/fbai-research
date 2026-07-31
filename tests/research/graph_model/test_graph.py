from __future__ import annotations

import pandas as pd
import pytest

from fbai.research.graph_model.config import GraphEmbeddingConfig, graph_feature_columns
from fbai.research.graph_model.graph import (
    GraphInputError,
    build_fold_graph_features,
    build_fold_graph_index,
)


def _tiny_graph_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixtures = pd.DataFrame(
        {
            "Division": ["SYN"] * 5,
            "SeasonStartYear": [2021, 2022, 2022, 2022, 2022],
            "MatchDate": pd.to_datetime(
                [
                    "2021-08-01",
                    "2022-08-01",
                    "2022-08-08",
                    "2022-08-08",
                    "2022-08-15",
                ]
            ),
            "HomeTeam": ["Alpha", "Alpha", "Alpha", "Alpha", "Alpha"],
            "AwayTeam": ["Beta", "Beta", "Beta", "Gamma", "Beta"],
            "game_id": [1, 2, 3, 4, 5],
            "home_club_id": [10] * 5,
            "away_club_id": [20, 20, 20, 30, 20],
            "FTR": ["H", "D", "A", "H", "D"],
        }
    )
    appearances: list[dict[str, int]] = []
    lineups: list[dict[str, int | str]] = []
    away_clubs = {1: 20, 2: 20, 3: 20, 4: 30, 5: 20}
    for game_id in range(1, 6):
        away_first_player = away_clubs[game_id] + 1
        for club_id, players in (
            (10, (1, 2)),
            (away_clubs[game_id], (away_first_player, away_first_player + 1)),
        ):
            for player_id in players:
                appearances.append(
                    {
                        "game_id": game_id,
                        "player_id": player_id,
                        "player_club_id": club_id,
                        "minutes_played": 90,
                    }
                )
                lineups.append(
                    {
                        "game_id": game_id,
                        "player_id": player_id,
                        "club_id": club_id,
                        "type": "starting_lineup",
                    }
                )
    return fixtures, pd.DataFrame(appearances), pd.DataFrame(lineups)


def _target_features(frame: pd.DataFrame, game_ids: tuple[int, ...]) -> pd.DataFrame:
    config = GraphEmbeddingConfig(name="svd2_test", dimension=2)
    columns = graph_feature_columns(config)
    return (
        frame.loc[frame["game_id"].isin(game_ids), ["game_id", *columns]]
        .sort_values("game_id")
        .reset_index(drop=True)
    )


def test_node_ids_and_features_are_independent_of_input_row_order() -> None:
    fixtures, appearances, lineups = _tiny_graph_frames()
    config = GraphEmbeddingConfig(name="svd2_test", dimension=2)

    expected_index = build_fold_graph_index(
        fixtures,
        appearances,
        lineups,
        test_year=2022,
        config=config,
    )
    expected = build_fold_graph_features(
        fixtures,
        appearances,
        lineups,
        test_year=2022,
        config=config,
    )
    shuffled_index = build_fold_graph_index(
        fixtures.sample(frac=1.0, random_state=1),
        appearances.sample(frac=1.0, random_state=2),
        lineups.sample(frac=1.0, random_state=3),
        test_year=2022,
        config=config,
    )
    shuffled = build_fold_graph_features(
        fixtures.sample(frac=1.0, random_state=4),
        appearances.sample(frac=1.0, random_state=5),
        lineups.sample(frac=1.0, random_state=6),
        test_year=2022,
        config=config,
    )

    assert shuffled_index == expected_index
    pd.testing.assert_frame_equal(shuffled.features, expected.features)
    assert shuffled.metadata == expected.metadata


def test_duplicate_fixture_natural_keys_fail() -> None:
    fixtures, appearances, lineups = _tiny_graph_frames()
    duplicated = pd.concat([fixtures, fixtures.iloc[[0]]], ignore_index=True)

    with pytest.raises(GraphInputError, match="Natural key is not unique"):
        build_fold_graph_features(
            duplicated,
            appearances,
            lineups,
            test_year=2022,
            config=GraphEmbeddingConfig(name="svd2_test", dimension=2),
        )


def test_current_date_appearances_and_lineups_are_isolated_as_one_batch() -> None:
    fixtures, appearances, lineups = _tiny_graph_frames()
    config = GraphEmbeddingConfig(name="svd2_test", dimension=2)
    base = build_fold_graph_features(
        fixtures,
        appearances,
        lineups,
        test_year=2022,
        config=config,
    )
    changed_appearances = appearances.copy(deep=True)
    changed_appearances.loc[changed_appearances["game_id"].isin([3, 4]), "minutes_played"] = 1
    changed_lineups = lineups.loc[~lineups["game_id"].isin([3, 4])].copy(deep=True)
    changed = build_fold_graph_features(
        fixtures,
        changed_appearances,
        changed_lineups,
        test_year=2022,
        config=config,
    )

    pd.testing.assert_frame_equal(
        _target_features(base.features, (3, 4)),
        _target_features(changed.features, (3, 4)),
    )
    assert base.metadata.same_date_batches


def test_future_graph_relations_and_results_cannot_change_earlier_features() -> None:
    fixtures, appearances, lineups = _tiny_graph_frames()
    config = GraphEmbeddingConfig(name="svd2_test", dimension=2)
    base = build_fold_graph_features(
        fixtures,
        appearances,
        lineups,
        test_year=2022,
        config=config,
    )
    changed_fixtures = fixtures.copy(deep=True)
    changed_fixtures.loc[changed_fixtures["game_id"].isin([3, 4, 5]), "FTR"] = "A"
    changed_appearances = appearances.copy(deep=True)
    changed_appearances.loc[changed_appearances["game_id"] == 5, "minutes_played"] = 999
    changed = build_fold_graph_features(
        changed_fixtures,
        changed_appearances,
        lineups,
        test_year=2022,
        config=config,
    )

    pd.testing.assert_frame_equal(
        _target_features(base.features, (3,)),
        _target_features(changed.features, (3,)),
    )


def test_graph_fit_metadata_contains_counts_not_node_names() -> None:
    fixtures, appearances, lineups = _tiny_graph_frames()
    built = build_fold_graph_features(
        fixtures,
        appearances,
        lineups,
        test_year=2022,
        config=GraphEmbeddingConfig(name="svd2_test", dimension=2),
    )

    metadata = built.metadata.to_dict()
    assert metadata["training_fixture_count"] == 1
    assert metadata["team_node_count"] == 2
    assert metadata["player_node_count"] == 4
    assert metadata["nonzero_edge_count"] == 4
    assert all(isinstance(value, (int, bool)) for value in metadata.values())
    assert not any(isinstance(value, (str, list, dict)) for value in metadata.values())
