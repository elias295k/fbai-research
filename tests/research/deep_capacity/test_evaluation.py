from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbai.core.metrics import weighted_fold_summary
from fbai.features import FEATURE_COLUMNS, build_feature_table
from fbai.research.deep_capacity.config import AUTHORITATIVE_CONFIGS
from fbai.research.deep_capacity.evaluation import evaluate_deep_capacity
from fbai.testing.synthetic import make_synthetic_canonical_matches

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[3]
PROTECTED_RESULTS = (
    ROOT / "research" / "lr52_baseline" / "result.json",
    ROOT / "research" / "closing_market_benchmark" / "result.json",
    ROOT / "research" / "match2vec" / "result.json",
    ROOT / "research" / "pseudo_xg" / "result.json",
    ROOT / "research" / "understat_xg" / "result.json",
)


@pytest.fixture(scope="module")
def synthetic_features() -> pd.DataFrame:
    canonical = make_synthetic_canonical_matches(
        seed=901,
        season_start_years=(2020, 2021, 2022, 2023, 2024, 2025),
        divisions=("SYN1", "SYN2"),
        teams_per_division=4,
    )
    return build_feature_table(canonical)


@pytest.fixture(scope="module")
def test_configs():
    return tuple(
        replace(
            configuration,
            max_epochs=2,
            early_stopping_patience=1,
            batch_size=64,
        )
        for configuration in AUTHORITATIVE_CONFIGS
    )


@pytest.fixture(scope="module")
def report(synthetic_features: pd.DataFrame, test_configs):
    return evaluate_deep_capacity(
        synthetic_features,
        configurations=test_configs,
    )


def test_both_architectures_use_development_and_only_selected_uses_final(report) -> None:
    assert [candidate.configuration.name for candidate in report.candidates] == [
        "h64_do10_wd1e3",
        "h128_64_do20_wd1e3",
    ]
    assert all(
        [fold.test_year for fold in candidate.development_folds] == [2022, 2023, 2024]
        for candidate in report.candidates
    )
    assert report.selected_final_fold.test_year == 2025
    assert (
        report.selected_final_fold.architecture_name == report.selected_candidate.configuration.name
    )
    assert report.selection_used_development_only


def test_fold_local_preprocessing_and_same_date_isolation(report) -> None:
    for candidate in report.candidates:
        for fold in candidate.development_folds:
            assert fold.preprocessing_fit_rows == fold.fit_rows
            assert fold.fit_rows + fold.validation_rows == fold.train_rows
            assert fold.fit_end_date < fold.validation_start_date
            assert fold.validation_end_date <= fold.train_end_date
            assert fold.train_end_date < fold.test_start_date
            assert fold.same_date_isolation


def test_candidates_and_lr52_use_identical_hda_rows(report) -> None:
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


def test_success_gate_and_improvement_sign_are_exact(report) -> None:
    selected = report.selected_candidate
    expected = selected.development.lr52.log_loss - selected.development.candidate.log_loss

    assert report.success_gate.candidate_improvement_log_loss == pytest.approx(expected)
    assert report.success_gate.definition.minimum_improvement_log_loss == 0.005
    assert report.success_gate.definition.minimum_improving_folds == 2
    assert report.success_gate.improving_fold_count == sum(
        fold.candidate_improvement_log_loss > 0 for fold in selected.development_folds
    )
    assert report.disposition in {"DEEP_MODEL_CANDIDATE", "DATA_CEILING_UPHELD"}


def test_shuffled_input_is_keyed_equivalent(
    synthetic_features: pd.DataFrame,
    test_configs,
    report,
) -> None:
    shuffled = synthetic_features.sample(frac=1.0, random_state=12).reset_index(drop=True)

    rebuilt = evaluate_deep_capacity(shuffled, configurations=test_configs)

    assert rebuilt.to_dict() == report.to_dict()


def test_final_fold_cannot_influence_development_or_selection(
    synthetic_features: pd.DataFrame,
    test_configs,
    report,
) -> None:
    changed = synthetic_features.copy(deep=True)
    final = changed["SeasonStartYear"].eq(2025)
    changed.loc[final, list(FEATURE_COLUMNS)] = changed.loc[final, list(FEATURE_COLUMNS)] + 100.0

    rebuilt = evaluate_deep_capacity(changed, configurations=test_configs)

    assert [candidate.development.to_dict() for candidate in rebuilt.candidates] == [
        candidate.development.to_dict() for candidate in report.candidates
    ]
    assert rebuilt.selected_architecture == report.selected_architecture


def test_inputs_and_prior_research_records_are_not_modified(
    synthetic_features: pd.DataFrame,
) -> None:
    original = synthetic_features.copy(deep=True)
    before = {
        path.name + path.parent.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PROTECTED_RESULTS
    }
    config = replace(
        AUTHORITATIVE_CONFIGS[0],
        max_epochs=1,
        early_stopping_patience=1,
        batch_size=64,
    )

    evaluate_deep_capacity(synthetic_features, configurations=(config,))

    pd.testing.assert_frame_equal(synthetic_features, original)
    after = {
        path.name + path.parent.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PROTECTED_RESULTS
    }
    assert after == before
