from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbai.core.leakage import NATURAL_KEY
from fbai.core.metrics import weighted_fold_summary
from fbai.features import FEATURE_COLUMNS, build_feature_table
from fbai.research.player_availability.evaluation import evaluate_player_availability
from fbai.testing.synthetic import make_synthetic_canonical_matches

ROOT = Path(__file__).resolve().parents[3]
PROTECTED_RESULTS = tuple(
    ROOT / "research" / name / "result.json"
    for name in (
        "lr52_baseline",
        "closing_market_benchmark",
        "match2vec",
        "pseudo_xg",
        "understat_xg",
        "deep_capacity",
        "graph_model",
    )
)


def _availability_frames(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixtures = features.loc[:, [*NATURAL_KEY, "SeasonStartYear"]].copy(deep=True)
    fixtures["game_id"] = np.arange(1, len(fixtures) + 1, dtype=np.int64)
    team_keys = sorted(
        {
            (str(row.Division), str(team))
            for row in fixtures.itertuples(index=False)
            for team in (row.HomeTeam, row.AwayTeam)
        }
    )
    club_ids = {key: index + 10 for index, key in enumerate(team_keys)}
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
            for offset in range(1, 13):
                player_id = club_id * 100 + offset
                appearances.append(
                    {
                        "game_id": int(row.game_id),
                        "player_id": player_id,
                        "player_club_id": club_id,
                        "minutes_played": 90 if offset <= 11 else 15,
                    }
                )
                lineups.append(
                    {
                        "game_id": int(row.game_id),
                        "player_id": player_id,
                        "club_id": club_id,
                        "type": "starting_lineup" if offset <= 11 else "substitutes",
                    }
                )
    valuations = pd.DataFrame(
        [
            {
                "player_id": club_id * 100 + offset,
                "date": pd.Timestamp("2019-01-01"),
                "market_value_in_eur": club_id * 100_000 + offset * 10_000,
            }
            for club_id in club_ids.values()
            for offset in range(1, 13)
        ]
    )
    return fixtures, pd.DataFrame(appearances), pd.DataFrame(lineups), valuations


@pytest.fixture(scope="module")
def synthetic_inputs():
    canonical = make_synthetic_canonical_matches(
        seed=5303,
        season_start_years=(2020, 2021, 2022, 2023, 2024, 2025),
        divisions=("SYN1", "SYN2"),
        teams_per_division=4,
    )
    features = build_feature_table(canonical)
    return features, *_availability_frames(features)


@pytest.fixture(scope="module")
def report(synthetic_inputs):
    return evaluate_player_availability(*synthetic_inputs)


def test_fold_local_preprocessing_and_final_isolation(report) -> None:
    assert [fold.test_year for fold in report.folds] == [2022, 2023, 2024, 2025]
    assert [fold.role.value for fold in report.folds] == [
        "development",
        "development",
        "development",
        "historical_final",
    ]
    assert report.final_evaluated_once
    for fold in report.folds:
        assert fold.preprocessing_fit_rows == fold.train_rows
        assert fold.train_end_date < fold.test_start_date
        assert fold.same_date_isolation


def test_candidate_and_lr52_use_identical_hda_rows(report) -> None:
    assert report.class_order == ("H", "D", "A")
    assert report.candidate_input_count == 115
    for fold in report.folds:
        assert fold.identical_test_rows
        assert fold.lr52.n_samples == fold.candidate.n_samples == fold.test_rows
        assert np.isfinite(fold.candidate.log_loss)
        assert np.isfinite(fold.candidate.brier_score)
        assert np.isfinite(fold.candidate.ece)


def test_weighted_aggregation_and_gate_are_exact(report) -> None:
    development_folds = report.folds[:3]
    direct = weighted_fold_summary([fold.candidate.to_dict() for fold in development_folds])
    expected_improvement = report.development.lr52.log_loss - report.development.candidate.log_loss

    assert report.development.candidate.n_samples == direct["n_samples"]
    assert report.development.candidate.log_loss == pytest.approx(direct["log_loss"])
    assert report.success_gate.candidate_improvement_log_loss == pytest.approx(expected_improvement)
    assert report.success_gate.definition.minimum_improvement_log_loss == 0.005
    assert report.success_gate.definition.minimum_improving_folds == 2
    assert report.success_gate.improving_fold_count == sum(
        fold.candidate_improvement_log_loss > 0.0 for fold in development_folds
    )


def test_shuffled_inputs_are_deterministically_equivalent(synthetic_inputs, report) -> None:
    features, fixtures, appearances, lineups, valuations = synthetic_inputs
    rebuilt = evaluate_player_availability(
        features.sample(frac=1.0, random_state=1).reset_index(drop=True),
        fixtures.sample(frac=1.0, random_state=2).reset_index(drop=True),
        appearances.sample(frac=1.0, random_state=3).reset_index(drop=True),
        lineups.sample(frac=1.0, random_state=4).reset_index(drop=True),
        valuations.sample(frac=1.0, random_state=5).reset_index(drop=True),
    )

    assert rebuilt.to_dict() == report.to_dict()


def test_final_rows_cannot_change_development(synthetic_inputs, report) -> None:
    features, fixtures, appearances, lineups, valuations = synthetic_inputs
    changed = features.copy(deep=True)
    final = changed["SeasonStartYear"].eq(2025)
    changed.loc[final, list(FEATURE_COLUMNS)] = changed.loc[final, list(FEATURE_COLUMNS)] + 100.0
    rebuilt = evaluate_player_availability(changed, fixtures, appearances, lineups, valuations)

    assert rebuilt.development.to_dict() == report.development.to_dict()
    assert [fold.to_dict() for fold in rebuilt.folds[:3]] == [
        fold.to_dict() for fold in report.folds[:3]
    ]


def test_inputs_and_previous_results_remain_unchanged(synthetic_inputs) -> None:
    originals = tuple(frame.copy(deep=True) for frame in synthetic_inputs)
    before = {
        path.parent.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PROTECTED_RESULTS
    }

    evaluate_player_availability(*synthetic_inputs)

    for frame, original in zip(synthetic_inputs, originals, strict=True):
        pd.testing.assert_frame_equal(frame, original)
    after = {
        path.parent.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PROTECTED_RESULTS
    }
    assert after == before
