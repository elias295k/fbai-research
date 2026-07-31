"""Leakage-safe temporal team-player graph representation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from fbai.core.leakage import NATURAL_KEY, TableValidationError, assert_natural_key_valid
from fbai.core.splits import STABLE_ORDER_KEY
from fbai.research.graph_model.config import (
    GraphEmbeddingConfig,
    graph_feature_columns,
)

FIXTURE_REQUIRED_COLUMNS: tuple[str, ...] = (
    *NATURAL_KEY,
    "SeasonStartYear",
    "game_id",
    "home_club_id",
    "away_club_id",
)
APPEARANCE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "player_club_id",
    "minutes_played",
)
LINEUP_REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "club_id",
    "type",
)


class GraphInputError(ValueError):
    """Raised when graph source frames violate the experimental contract."""


@dataclass(frozen=True, slots=True)
class GraphIndex:
    """Deterministic fold-local IDs for team and player nodes."""

    team_to_id: dict[tuple[str, str], int]
    player_to_id: dict[int, int]


@dataclass(frozen=True, slots=True)
class GraphFitMetadata:
    """Aggregate-only graph size and fold-fit metadata."""

    training_fixture_count: int
    team_node_count: int
    player_node_count: int
    nonzero_edge_count: int
    embedding_dimension: int
    graph_feature_count: int
    same_date_batches: bool

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "training_fixture_count": self.training_fixture_count,
            "team_node_count": self.team_node_count,
            "player_node_count": self.player_node_count,
            "nonzero_edge_count": self.nonzero_edge_count,
            "embedding_dimension": self.embedding_dimension,
            "graph_feature_count": self.graph_feature_count,
            "same_date_batches": self.same_date_batches,
        }


@dataclass(frozen=True, slots=True)
class GraphFeatureBuild:
    """Fold-specific graph features and non-identifying fit metadata."""

    features: pd.DataFrame
    metadata: GraphFitMetadata


@dataclass(frozen=True, slots=True)
class GraphSourceFrames:
    """Validated, deterministically ordered local graph inputs."""

    fixtures: pd.DataFrame
    appearances: pd.DataFrame
    lineups: pd.DataFrame
    appearance_maps: dict[tuple[int, int], dict[int, float]]
    starter_maps: dict[tuple[int, int], set[int]]
    bench_maps: dict[tuple[int, int], set[int]]


@dataclass(frozen=True, slots=True)
class _SideRecord:
    fixture_position: int
    side: str
    division: str
    season_start_year: int
    match_date: pd.Timestamp
    team: str
    game_id: int
    weights: dict[int, float]

    @property
    def team_key(self) -> tuple[str, str]:
        return (self.division, self.team)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise GraphInputError(f"{name} is missing columns: {', '.join(missing)}")


def _validated_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(fixtures, pd.DataFrame) or fixtures.empty:
        raise GraphInputError("graph fixtures must be a non-empty pandas DataFrame")
    _require_columns(fixtures, FIXTURE_REQUIRED_COLUMNS, "graph fixtures")
    prepared = fixtures.loc[:, list(FIXTURE_REQUIRED_COLUMNS)].copy(deep=True)
    prepared["MatchDate"] = pd.to_datetime(prepared["MatchDate"], errors="raise")
    try:
        assert_natural_key_valid(prepared, key_columns=NATURAL_KEY)
    except TableValidationError as exc:
        raise GraphInputError(str(exc)) from exc
    if prepared["game_id"].isna().any() or prepared["game_id"].duplicated().any():
        raise GraphInputError("graph fixture game_id values must be complete and unique")
    for column in ("SeasonStartYear", "game_id", "home_club_id", "away_club_id"):
        numeric = pd.to_numeric(prepared[column], errors="coerce")
        if numeric.isna().any():
            raise GraphInputError(f"graph fixture {column} values must be numeric")
        prepared[column] = numeric.astype("int64")
    return prepared.sort_values(list(STABLE_ORDER_KEY), kind="mergesort").reset_index(drop=True)


def _validated_relations(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    known_game_ids: frozenset[int],
    name: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise GraphInputError(f"{name} must be a pandas DataFrame")
    _require_columns(frame, columns, name)
    prepared = frame.loc[:, list(columns)].copy(deep=True)
    key_columns = [column for column in columns if column != "minutes_played" and column != "type"]
    prepared = prepared.dropna(subset=key_columns)
    for column in key_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise").astype("int64")
    unknown = set(prepared["game_id"].astype(int)).difference(known_game_ids)
    if unknown:
        raise GraphInputError(f"{name} contains game IDs outside the fixture contract")
    return prepared.sort_values(key_columns, kind="mergesort").reset_index(drop=True)


def _appearance_maps(appearances: pd.DataFrame) -> dict[tuple[int, int], dict[int, float]]:
    output: dict[tuple[int, int], dict[int, float]] = {}
    if appearances.empty:
        return output
    prepared = appearances.copy(deep=True)
    prepared["minutes_played"] = pd.to_numeric(prepared["minutes_played"], errors="coerce")
    for key, group in prepared.groupby(["game_id", "player_club_id"], sort=True):
        minutes = group.groupby("player_id", sort=True)["minutes_played"].sum(min_count=1)
        output[(int(key[0]), int(key[1]))] = {
            int(player_id): float(value) for player_id, value in minutes.dropna().items()
        }
    return output


def _lineup_maps(
    lineups: pd.DataFrame,
) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]]]:
    starters: dict[tuple[int, int], set[int]] = defaultdict(set)
    bench: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in lineups.itertuples(index=False):
        key = (int(row.game_id), int(row.club_id))
        if row.type == "starting_lineup":
            starters[key].add(int(row.player_id))
        elif row.type == "substitutes":
            bench[key].add(int(row.player_id))
    return starters, bench


def _side_records(
    fixtures: pd.DataFrame,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
    config: GraphEmbeddingConfig,
) -> tuple[_SideRecord, ...]:
    appearance_maps = _appearance_maps(appearances)
    starters, bench = _lineup_maps(lineups)
    return _side_records_from_maps(
        fixtures,
        appearance_maps,
        starters,
        bench,
        config,
    )


def _side_records_from_maps(
    fixtures: pd.DataFrame,
    appearance_maps: dict[tuple[int, int], dict[int, float]],
    starters: dict[tuple[int, int], set[int]],
    bench: dict[tuple[int, int], set[int]],
    config: GraphEmbeddingConfig,
) -> tuple[_SideRecord, ...]:
    records: list[_SideRecord] = []
    for position, row in fixtures.iterrows():
        for side, team_column, club_column in (
            ("H", "HomeTeam", "home_club_id"),
            ("A", "AwayTeam", "away_club_id"),
        ):
            game_id = int(row["game_id"])
            club_id = int(row[club_column])
            relation_key = (game_id, club_id)
            weights: dict[int, float] = defaultdict(float)
            for player_id, minutes in appearance_maps.get(relation_key, {}).items():
                weights[int(player_id)] += float(minutes)
            for player_id in sorted(starters.get(relation_key, set())):
                weights[int(player_id)] += config.starter_weight
            for player_id in sorted(bench.get(relation_key, set())):
                weights[int(player_id)] += config.bench_weight
            records.append(
                _SideRecord(
                    fixture_position=int(position),
                    side=side,
                    division=str(row["Division"]),
                    season_start_year=int(row["SeasonStartYear"]),
                    match_date=pd.Timestamp(row["MatchDate"]),
                    team=str(row[team_column]),
                    game_id=game_id,
                    weights=dict(weights),
                )
            )
    return tuple(records)


def prepare_graph_source_frames(
    fixtures: pd.DataFrame,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
) -> GraphSourceFrames:
    """Validate graph inputs once for repeated fold-local construction."""

    prepared_fixtures = _validated_fixtures(fixtures)
    game_ids = frozenset(prepared_fixtures["game_id"].astype(int))
    prepared_appearances = _validated_relations(
        appearances,
        columns=APPEARANCE_REQUIRED_COLUMNS,
        known_game_ids=game_ids,
        name="graph appearances",
    )
    prepared_lineups = _validated_relations(
        lineups,
        columns=LINEUP_REQUIRED_COLUMNS,
        known_game_ids=game_ids,
        name="graph lineups",
    )
    appearance_maps = _appearance_maps(prepared_appearances)
    starter_maps, bench_maps = _lineup_maps(prepared_lineups)
    return GraphSourceFrames(
        fixtures=prepared_fixtures,
        appearances=prepared_appearances,
        lineups=prepared_lineups,
        appearance_maps=appearance_maps,
        starter_maps=starter_maps,
        bench_maps=bench_maps,
    )


def _index_from_records(
    records: tuple[_SideRecord, ...],
    training_game_ids: frozenset[int],
) -> GraphIndex:
    team_to_id: dict[tuple[str, str], int] = {}
    player_to_id: dict[int, int] = {}
    for record in records:
        if record.game_id not in training_game_ids:
            continue
        team_to_id.setdefault(record.team_key, len(team_to_id))
        for player_id, weight in record.weights.items():
            if weight > 0.0:
                player_to_id.setdefault(int(player_id), len(player_to_id))
    return GraphIndex(team_to_id=team_to_id, player_to_id=player_to_id)


def build_fold_graph_index(
    fixtures: pd.DataFrame,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
    *,
    test_year: int,
    config: GraphEmbeddingConfig,
) -> GraphIndex:
    """Return deterministic fold-local node IDs, primarily for audit tests."""

    source = prepare_graph_source_frames(fixtures, appearances, lineups)
    records = _side_records_from_maps(
        source.fixtures,
        source.appearance_maps,
        source.starter_maps,
        source.bench_maps,
        config,
    )
    training_ids = frozenset(
        source.fixtures.loc[
            source.fixtures["SeasonStartYear"] <= test_year - 1,
            "game_id",
        ].astype(int)
    )
    return _index_from_records(records, training_ids)


def _fit_player_embeddings(
    records: tuple[_SideRecord, ...],
    training_game_ids: frozenset[int],
    config: GraphEmbeddingConfig,
) -> tuple[dict[int, np.ndarray], GraphFitMetadata]:
    index = _index_from_records(records, training_game_ids)
    totals: dict[tuple[int, int], float] = defaultdict(float)
    for record in records:
        if record.game_id not in training_game_ids:
            continue
        team_id = index.team_to_id[record.team_key]
        for player_id, weight in record.weights.items():
            if weight > 0.0:
                totals[(team_id, index.player_to_id[int(player_id)])] += float(weight)
    metadata = GraphFitMetadata(
        training_fixture_count=len(training_game_ids),
        team_node_count=len(index.team_to_id),
        player_node_count=len(index.player_to_id),
        nonzero_edge_count=len(totals),
        embedding_dimension=config.dimension,
        graph_feature_count=config.feature_count,
        same_date_batches=True,
    )
    if not index.team_to_id or not index.player_to_id:
        return {}, metadata
    matrix = np.zeros(
        (len(index.team_to_id), len(index.player_to_id)),
        dtype=np.float64,
    )
    for (team_id, player_id), weight in totals.items():
        matrix[team_id, player_id] = np.log1p(weight)
    component_count = min(config.dimension, max(1, min(matrix.shape) - 1))
    svd = TruncatedSVD(
        n_components=component_count,
        random_state=config.random_seed,
    )
    svd.fit(matrix)
    embeddings = svd.components_.T * np.sqrt(svd.singular_values_)
    if component_count < config.dimension:
        embeddings = np.pad(
            embeddings,
            ((0, 0), (0, config.dimension - component_count)),
        )
    player_by_id = {node_id: player_id for player_id, node_id in index.player_to_id.items()}
    return (
        {
            player_by_id[node_id]: embeddings[node_id].astype(np.float64)
            for node_id in range(len(player_by_id))
        },
        metadata,
    )


def _history_pool(
    history: list[_SideRecord],
    config: GraphEmbeddingConfig,
) -> dict[int, float]:
    pool: dict[int, float] = defaultdict(float)
    for record in history[-config.history_window_matches :]:
        for player_id, weight in record.weights.items():
            pool[player_id] += weight
    return pool


def _aggregate_pool(
    player_weights: dict[int, float],
    player_embeddings: dict[int, np.ndarray],
    dimension: int,
) -> tuple[np.ndarray, float, float]:
    vector = np.zeros(dimension, dtype=np.float64)
    total_weight = float(sum(weight for weight in player_weights.values() if weight > 0.0))
    known_weight = 0.0
    if total_weight <= 0.0:
        return vector, np.nan, 0.0
    for player_id, weight in player_weights.items():
        if weight <= 0.0:
            continue
        embedding = player_embeddings.get(int(player_id))
        if embedding is None:
            continue
        known_weight += float(weight)
        vector += float(weight) * embedding
    if known_weight > 0.0:
        vector /= known_weight
    return vector, known_weight / total_weight, float(len(player_weights))


def _build_side_features(
    records: tuple[_SideRecord, ...],
    embeddings: dict[int, np.ndarray],
    config: GraphEmbeddingConfig,
) -> dict[tuple[int, str], tuple[np.ndarray, float, float]]:
    side_features: dict[tuple[int, str], tuple[np.ndarray, float, float]] = {}
    histories: dict[tuple[str, str], list[_SideRecord]] = defaultdict(list)
    ordered = sorted(
        records,
        key=lambda record: (
            record.match_date,
            record.division,
            record.team,
            record.game_id,
            record.side,
        ),
    )
    by_date: dict[pd.Timestamp, list[_SideRecord]] = defaultdict(list)
    for record in ordered:
        by_date[record.match_date].append(record)
    for match_date in sorted(by_date):
        date_batch = by_date[match_date]
        for record in date_batch:
            same_season_history = [
                item
                for item in histories[record.team_key]
                if item.season_start_year == record.season_start_year
            ]
            pool = _history_pool(same_season_history, config)
            side_features[(record.fixture_position, record.side)] = _aggregate_pool(
                pool,
                embeddings,
                config.dimension,
            )
        for record in date_batch:
            histories[record.team_key].append(record)
    return side_features


def build_fold_graph_features(
    fixtures: pd.DataFrame,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
    *,
    test_year: int,
    config: GraphEmbeddingConfig,
) -> GraphFeatureBuild:
    """Build fold-local SVD features with strict-prior, complete-date histories."""

    source = prepare_graph_source_frames(fixtures, appearances, lineups)
    return build_fold_graph_features_from_source(
        source,
        test_year=test_year,
        config=config,
    )


def build_fold_graph_features_from_source(
    source: GraphSourceFrames,
    *,
    test_year: int,
    config: GraphEmbeddingConfig,
) -> GraphFeatureBuild:
    """Build a fold from already validated graph frames."""

    records = _side_records_from_maps(
        source.fixtures,
        source.appearance_maps,
        source.starter_maps,
        source.bench_maps,
        config,
    )
    training_ids = frozenset(
        source.fixtures.loc[
            source.fixtures["SeasonStartYear"] <= test_year - 1,
            "game_id",
        ].astype(int)
    )
    embeddings, metadata = _fit_player_embeddings(records, training_ids, config)
    side_features = _build_side_features(records, embeddings, config)
    prefix = f"graph_{config.name}"
    output: list[dict[str, Any]] = []
    for position, row in source.fixtures.iterrows():
        home_vector, home_known, home_players = side_features[(int(position), "H")]
        away_vector, away_known, away_players = side_features[(int(position), "A")]
        difference = home_vector - away_vector
        denominator = float(np.linalg.norm(home_vector) * np.linalg.norm(away_vector))
        cosine = (
            float(np.dot(home_vector, away_vector) / denominator) if denominator > 0.0 else np.nan
        )
        values: dict[str, Any] = {
            column: row[column] for column in (*NATURAL_KEY, "SeasonStartYear", "game_id")
        }
        for dimension in range(config.dimension):
            values[f"{prefix}_H_e{dimension:02d}_pre"] = float(home_vector[dimension])
            values[f"{prefix}_A_e{dimension:02d}_pre"] = float(away_vector[dimension])
            values[f"{prefix}_diff_e{dimension:02d}_pre"] = float(difference[dimension])
            values[f"{prefix}_absdiff_e{dimension:02d}_pre"] = float(abs(difference[dimension]))
        values[f"{prefix}_cosine_pre"] = cosine
        values[f"{prefix}_l2_pre"] = float(np.linalg.norm(difference))
        values[f"{prefix}_H_known_weight_share_pre"] = home_known
        values[f"{prefix}_A_known_weight_share_pre"] = away_known
        values[f"{prefix}_diff_known_weight_share_pre"] = (
            float(home_known - away_known)
            if pd.notna(home_known) and pd.notna(away_known)
            else np.nan
        )
        values[f"{prefix}_H_pool_players_pre"] = home_players
        values[f"{prefix}_A_pool_players_pre"] = away_players
        values[f"{prefix}_diff_pool_players_pre"] = float(home_players - away_players)
        output.append(values)
    feature_frame = pd.DataFrame(output)
    expected_columns = (*NATURAL_KEY, "SeasonStartYear", "game_id", *graph_feature_columns(config))
    feature_frame = feature_frame.loc[:, list(expected_columns)]
    return GraphFeatureBuild(features=feature_frame, metadata=metadata)
