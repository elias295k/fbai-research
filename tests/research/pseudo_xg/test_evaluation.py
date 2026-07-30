from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbai.core.metrics import weighted_fold_summary
from fbai.features.build import build_feature_table
from fbai.research.pseudo_xg.evaluation import evaluate_pseudo_xg_candidate
from fbai.testing.synthetic import make_synthetic_canonical_matches


@pytest.fixture(scope="module")
def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = make_synthetic_canonical_matches(
        seed=533,
        season_start_years=(2020, 2021, 2022, 2023, 2024, 2025),
        divisions=("SYN1", "SYN2"),
        teams_per_division=4,
    )
    return build_feature_table(canonical), canonical


@pytest.fixture(scope="module")
def report(synthetic_inputs: tuple[pd.DataFrame, pd.DataFrame]):
    features, canonical = synthetic_inputs
    return evaluate_pseudo_xg_candidate(features, canonical)


def test_estimator_is_refitted_in_every_outer_fold(report) -> None:
    assert [fold.test_year for fold in report.folds] == [2022, 2023, 2024, 2025]
    assert [fold.estimator_training_match_rows for fold in report.folds] == [
        fold.train_rows for fold in report.folds
    ]
    assert [fold.train_rows for fold in report.folds] == sorted(
        fold.train_rows for fold in report.folds
    )
    assert len({fold.train_rows for fold in report.folds}) == 4
    assert all(fold.estimator_training_side_rows == 2 * fold.train_rows for fold in report.folds)


def test_candidate_and_lr52_use_identical_rows_and_hda_order(report) -> None:
    assert report.class_order == ("H", "D", "A")
    for fold in report.folds:
        assert fold.identical_test_rows
        assert fold.lr52.n_samples == fold.candidate.n_samples == fold.test_rows
        assert np.isfinite(fold.candidate.log_loss)
        assert np.isfinite(fold.candidate.brier_score)
        assert np.isfinite(fold.candidate.ece)


def test_weighted_aggregation_uses_sample_counts(report) -> None:
    direct = weighted_fold_summary([fold.candidate.to_dict() for fold in report.folds[:3]])

    assert report.development.candidate.n_samples == direct["n_samples"]
    assert report.development.candidate.log_loss == pytest.approx(direct["log_loss"])


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


def test_final_fold_statistics_cannot_change_development_results(
    synthetic_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, canonical = synthetic_inputs
    changed = canonical.copy(deep=True)
    final_rows = changed["SeasonStartYear"].eq(2025)
    changed.loc[final_rows, ["HomeShots", "AwayShots"]] += 100

    first = evaluate_pseudo_xg_candidate(features, canonical)
    second = evaluate_pseudo_xg_candidate(features, changed)

    assert first.development.to_dict() == second.development.to_dict()
