from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbai.features.build import build_feature_table
from fbai.research.understat_xg.evaluation import evaluate_understat_xg_candidate
from fbai.testing.synthetic import make_synthetic_canonical_matches


@pytest.fixture(scope="module")
def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical = make_synthetic_canonical_matches(
        seed=805,
        season_start_years=(2020, 2021, 2022, 2023, 2024, 2025),
        divisions=("E0", "D1"),
        teams_per_division=6,
    )
    external = canonical.loc[
        canonical["SeasonStartYear"].le(2023),
        ["Division", "MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway"],
    ].copy()
    index = np.arange(len(external))
    external["home_xg"] = np.clip(
        1.05 + 0.38 * external["FTHome"] + ((index * 7) % 7 - 3) * 0.18,
        0.05,
        None,
    )
    external["away_xg"] = np.clip(
        1.05 + 0.38 * external["FTAway"] + ((index * 11) % 7 - 3) * 0.18,
        0.05,
        None,
    )
    external = external.drop(columns=["FTHome", "FTAway"]).reset_index(drop=True)
    return build_feature_table(canonical), canonical, external


@pytest.fixture(scope="module")
def report(synthetic_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]):
    return evaluate_understat_xg_candidate(*synthetic_inputs)


def test_candidate_and_lr52_use_identical_covered_rows(report) -> None:
    assert report.class_order == ("H", "D", "A")
    assert [fold.test_year for fold in report.folds] == [2022, 2023, 2024, 2025]
    for fold in report.folds:
        assert fold.identical_test_rows
        assert fold.lr52.n_samples == fold.candidate.n_samples == fold.test_rows
        assert fold.candidate_preprocessing_fit_rows == fold.train_rows
        assert np.isfinite(fold.candidate.log_loss)


def test_source_feature_coverage_observes_season_reset(report) -> None:
    coverage = {fold.test_year: fold.xg_feature_coverage for fold in report.folds}

    assert coverage[2022] > 0.8
    assert coverage[2023] > 0.8
    assert coverage[2024] == 0.0
    assert coverage[2025] == 0.0


def test_source_gate_and_improvement_sign_are_exact(report) -> None:
    expected = report.development.lr52.log_loss - report.development.candidate.log_loss
    gate = report.success_gate

    assert gate.candidate_improvement_log_loss == pytest.approx(expected)
    assert gate.definition.minimum_improvement_log_loss == 0.005
    assert gate.definition.minimum_improving_folds == 2
    assert gate.improving_fold_count == sum(
        fold.candidate_improvement_log_loss > 0 for fold in report.folds[:3]
    )
    expected_disposition = "XG_SIGNAL_CANDIDATE" if gate.passed else "XG_SIGNAL_REJECTED_FOR_NOW"
    assert report.disposition == expected_disposition


def test_final_fold_match_statistics_cannot_change_development_results(
    synthetic_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    features, canonical, external = synthetic_inputs
    changed = canonical.copy(deep=True)
    final_rows = changed["SeasonStartYear"].eq(2025)
    changed.loc[final_rows, ["HomeShots", "AwayShots"]] += 100

    first = evaluate_understat_xg_candidate(features, canonical, external)
    second = evaluate_understat_xg_candidate(features, changed, external)

    assert first.development.to_dict() == second.development.to_dict()
