from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbai.core.leakage import NATURAL_KEY
from fbai.core.metrics import weighted_fold_summary
from fbai.features import FEATURE_COLUMNS, build_feature_table
from fbai.research.graph_model.config import (
    GRAPH_ONLY_CONTEXT,
    LR52_GRAPH_CONTEXT,
)
from fbai.research.graph_model.evaluation import evaluate_graph_model
from fbai.testing.synthetic import make_synthetic_canonical_matches

ROOT = Path(__file__).resolve().parents[3]
PROTECTED_RESULTS = (
    ROOT / "research" / "lr52_baseline" / "result.json",
    ROOT / "research" / "closing_market_benchmark" / "result.json",
    ROOT / "research" / "match2vec" / "result.json",
    ROOT / "research" / "pseudo_xg" / "result.json",
    ROOT / "research" / "understat_xg" / "result.json",
    ROOT / "research" / "deep_capacity" / "result.json",
)


def _graph_frames(
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixtures = feature_table.loc[:, [*NATURAL_KEY, "SeasonStartYear"]].copy(deep=True)
    fixtures["game_id"] = np.arange(1, len(fixtures) + 1, dtype=np.int64)
    team_keys = sorted(
        {
            (str(row.Division), str(team))
            for row in fixtures.itertuples(index=False)
            for team in (row.HomeTeam, row.AwayTeam)
        }
    )
    club_ids = {key: index + 100 for index, key in enumerate(team_keys)}
    fixtures["home_club_id"] = [
        club_ids[(str(row.Division), str(row.HomeTeam))] for row in fixtures.itertuples(index=False)
    ]
    fixtures["away_club_id"] = [
        club_ids[(str(row.Division), str(row.AwayTeam))] for row in fixtures.itertuples(index=False)
    ]
    appearances: list[dict[str, int]] = []
    lineups: list[dict[str, int | str]] = []
    for row in fixtures.itertuples(index=False):
        for club_id in (int(row.home_club_id), int(row.away_club_id)):
            for player_offset in range(1, 5):
                player_id = club_id * 10 + player_offset
                appearances.append(
                    {
                        "game_id": int(row.game_id),
                        "player_id": player_id,
                        "player_club_id": club_id,
                        "minutes_played": 90 - player_offset,
                    }
                )
                lineups.append(
                    {
                        "game_id": int(row.game_id),
                        "player_id": player_id,
                        "club_id": club_id,
                        "type": ("starting_lineup" if player_offset <= 3 else "substitutes"),
                    }
                )
    return fixtures, pd.DataFrame(appearances), pd.DataFrame(lineups)


@pytest.fixture(scope="module")
def synthetic_inputs():
    canonical = make_synthetic_canonical_matches(
        seed=1204,
        season_start_years=(2020, 2021, 2022, 2023, 2024, 2025),
        divisions=("SYN1", "SYN2"),
        teams_per_division=4,
    )
    feature_table = build_feature_table(canonical)
    return feature_table, *_graph_frames(feature_table)


@pytest.fixture(scope="module")
def report(synthetic_inputs):
    return evaluate_graph_model(*synthetic_inputs)


def test_all_internal_candidates_use_development_and_selected_method_uses_final(
    report,
) -> None:
    assert [
        (candidate.configuration.name, candidate.context) for candidate in report.candidates
    ] == [
        ("svd8_recent10", GRAPH_ONLY_CONTEXT),
        ("svd8_recent10", LR52_GRAPH_CONTEXT),
        ("svd16_recent10", GRAPH_ONLY_CONTEXT),
        ("svd16_recent10", LR52_GRAPH_CONTEXT),
    ]
    assert all(
        [fold.test_year for fold in candidate.development_folds] == [2022, 2023, 2024]
        for candidate in report.candidates
    )
    assert report.selection_context == LR52_GRAPH_CONTEXT
    assert report.selection_used_development_only
    assert {fold.context for fold in report.selected_final_contexts} == {
        GRAPH_ONLY_CONTEXT,
        LR52_GRAPH_CONTEXT,
    }
    assert all(fold.test_year == 2025 for fold in report.selected_final_contexts)


def test_graph_and_classifier_fitting_are_fold_local_and_date_safe(report) -> None:
    for candidate in report.candidates:
        previous_training_fixtures = 0
        for fold in candidate.development_folds:
            assert fold.preprocessing_fit_rows == fold.train_rows
            assert fold.train_end_date < fold.test_start_date
            assert fold.graph_metadata.training_fixture_count > previous_training_fixtures
            assert fold.graph_metadata.team_node_count > 0
            assert fold.graph_metadata.player_node_count > 0
            assert fold.graph_metadata.nonzero_edge_count > 0
            assert fold.graph_metadata.same_date_batches
            assert fold.same_date_isolation
            previous_training_fixtures = fold.graph_metadata.training_fixture_count


def test_candidate_and_lr52_use_identical_hda_test_rows(report) -> None:
    assert report.class_order == ("H", "D", "A")
    for candidate in report.candidates:
        for fold in candidate.development_folds:
            assert fold.identical_test_rows
            assert fold.lr52.n_samples == fold.candidate.n_samples == fold.test_rows
            assert np.isfinite(fold.candidate.log_loss)
            assert np.isfinite(fold.candidate.brier_score)
            assert np.isfinite(fold.candidate.ece)


def test_weighted_aggregation_is_exact(report) -> None:
    selected = report.selected_candidate
    direct = weighted_fold_summary(
        [fold.candidate.to_dict() for fold in selected.development_folds]
    )

    assert selected.development.candidate.n_samples == direct["n_samples"]
    assert selected.development.candidate.log_loss == pytest.approx(direct["log_loss"])


def test_success_gate_and_disposition_are_exact(report) -> None:
    selected = report.selected_candidate
    expected = selected.development.lr52.log_loss - selected.development.candidate.log_loss

    assert report.success_gate.candidate_improvement_log_loss == pytest.approx(expected)
    assert report.success_gate.definition.minimum_improvement_log_loss == 0.005
    assert report.success_gate.definition.minimum_improving_folds == 2
    assert report.success_gate.improving_fold_count == sum(
        fold.candidate_improvement_log_loss > 0.0 for fold in selected.development_folds
    )
    assert report.disposition in {
        "GRAPH_EMBEDDING_REJECTED_FOR_NOW",
        "DATA_CEILING_UPHELD",
    }


def test_shuffled_inputs_are_keyed_equivalent(synthetic_inputs, report) -> None:
    features, fixtures, appearances, lineups = synthetic_inputs
    rebuilt = evaluate_graph_model(
        features.sample(frac=1.0, random_state=10).reset_index(drop=True),
        fixtures.sample(frac=1.0, random_state=11).reset_index(drop=True),
        appearances.sample(frac=1.0, random_state=12).reset_index(drop=True),
        lineups.sample(frac=1.0, random_state=13).reset_index(drop=True),
    )

    assert rebuilt.to_dict() == report.to_dict()


def test_final_fold_cannot_influence_development_or_selection(
    synthetic_inputs,
    report,
) -> None:
    features, fixtures, appearances, lineups = synthetic_inputs
    changed = features.copy(deep=True)
    final = changed["SeasonStartYear"].eq(2025)
    changed.loc[final, list(FEATURE_COLUMNS)] = changed.loc[final, list(FEATURE_COLUMNS)] + 100.0

    rebuilt = evaluate_graph_model(changed, fixtures, appearances, lineups)

    assert [candidate.development.to_dict() for candidate in rebuilt.candidates] == [
        candidate.development.to_dict() for candidate in report.candidates
    ]
    assert rebuilt.selected_configuration == report.selected_configuration


def test_inputs_and_previous_research_records_are_unchanged(synthetic_inputs) -> None:
    features, fixtures, appearances, lineups = synthetic_inputs
    originals = tuple(frame.copy(deep=True) for frame in (features, fixtures, appearances, lineups))
    before = {
        path.parent.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PROTECTED_RESULTS
    }

    evaluate_graph_model(features, fixtures, appearances, lineups)

    for frame, original in zip(
        (features, fixtures, appearances, lineups),
        originals,
        strict=True,
    ):
        pd.testing.assert_frame_equal(frame, original)
    after = {
        path.parent.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PROTECTED_RESULTS
    }
    assert after == before
