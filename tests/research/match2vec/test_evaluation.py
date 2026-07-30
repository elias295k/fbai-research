from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbai.core.metrics import weighted_fold_summary
from fbai.features.build import build_feature_table
from fbai.research.match2vec.config import Match2VecConfig
from fbai.research.match2vec.evaluation import evaluate_match2vec_candidate
from fbai.testing.synthetic import make_synthetic_canonical_matches

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[3]
PHASE3_RESULT = ROOT / "research" / "lr52_baseline" / "result.json"


@pytest.fixture(scope="module")
def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = make_synthetic_canonical_matches(
        seed=520,
        season_start_years=(2020, 2021, 2022, 2023, 2024, 2025),
        divisions=("SYN1", "SYN2"),
        teams_per_division=4,
    )
    return build_feature_table(canonical), canonical


@pytest.fixture(scope="module")
def candidate_report(synthetic_inputs: tuple[pd.DataFrame, pd.DataFrame]):
    features, canonical = synthetic_inputs
    config = replace(
        Match2VecConfig(),
        max_epochs=3,
        early_stopping_patience=1,
        batch_size=32,
    )
    return evaluate_match2vec_candidate(features, canonical, config=config)


def test_synthetic_end_to_end_uses_existing_four_folds(candidate_report) -> None:
    assert [fold.test_year for fold in candidate_report.folds] == [2022, 2023, 2024, 2025]
    assert [fold.role.value for fold in candidate_report.folds] == [
        "development",
        "development",
        "development",
        "historical_final",
    ]
    assert all(fold.identical_test_rows for fold in candidate_report.folds)
    assert all(fold.train_end_date < fold.test_start_date for fold in candidate_report.folds)


def test_probabilistic_metrics_are_finite_and_hda_ordered(candidate_report) -> None:
    assert candidate_report.class_order == ("H", "D", "A")
    for fold in candidate_report.folds:
        assert fold.lr52.n_samples == fold.test_rows
        assert fold.candidate.n_samples == fold.test_rows
        assert np.isfinite(fold.candidate.log_loss)
        assert np.isfinite(fold.candidate.brier_score)
        assert np.isfinite(fold.candidate.ece)


def test_weighted_aggregation_uses_sample_counts(candidate_report) -> None:
    development = candidate_report.folds[:3]
    direct = weighted_fold_summary([fold.candidate.to_dict() for fold in development])

    assert candidate_report.development.candidate.log_loss == pytest.approx(direct["log_loss"])
    assert candidate_report.development.candidate.n_samples == direct["n_samples"]


def test_improvement_sign_and_gate_are_exact(candidate_report) -> None:
    expected = (
        candidate_report.development.lr52.log_loss - candidate_report.development.candidate.log_loss
    )

    assert candidate_report.development.candidate_improvement_log_loss == pytest.approx(expected)
    gate = candidate_report.success_gate
    assert gate.candidate_improvement_log_loss == pytest.approx(expected)
    assert gate.definition.minimum_improvement_log_loss == 0.005
    assert gate.definition.minimum_improving_folds == 2
    assert gate.improving_fold_count == sum(
        fold.candidate_improvement_log_loss > 0.0 for fold in candidate_report.folds[:3]
    )


def test_vocabulary_and_model_are_rebuilt_per_fold(candidate_report) -> None:
    assert all(fold.vocabulary_unknown_tokens == 2 for fold in candidate_report.folds)
    assert all(fold.representation_fit_rows < fold.train_rows for fold in candidate_report.folds)
    assert all(fold.representation_validation_rows > 0 for fold in candidate_report.folds)
    assert all(fold.training_epochs >= 1 for fold in candidate_report.folds)


def test_input_frames_and_phase3_record_are_not_modified(
    synthetic_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, canonical = synthetic_inputs
    before_features = features.copy(deep=True)
    before_canonical = canonical.copy(deep=True)
    before_phase3 = hashlib.sha256(PHASE3_RESULT.read_bytes()).hexdigest()
    config = replace(
        Match2VecConfig(),
        max_epochs=1,
        early_stopping_patience=1,
        batch_size=64,
    )

    evaluate_match2vec_candidate(features, canonical, config=config)

    pd.testing.assert_frame_equal(features, before_features)
    pd.testing.assert_frame_equal(canonical, before_canonical)
    assert hashlib.sha256(PHASE3_RESULT.read_bytes()).hexdigest() == before_phase3


def test_development_results_cannot_read_final_fold_rows(
    synthetic_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, canonical = synthetic_inputs
    changed = canonical.copy(deep=True)
    final_rows = changed["SeasonStartYear"].eq(2025)
    changed.loc[final_rows, "HomeShots"] = changed.loc[final_rows, "HomeShots"] + 50
    config = replace(
        Match2VecConfig(),
        max_epochs=1,
        early_stopping_patience=1,
        batch_size=64,
    )
    first = evaluate_match2vec_candidate(features, canonical, config=config)
    second = evaluate_match2vec_candidate(features, changed, config=config)

    assert first.development.to_dict() == second.development.to_dict()
