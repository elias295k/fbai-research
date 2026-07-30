from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pytest

import fbai.evaluation.runner as runner_module
from fbai.core.metrics import CLASS_ORDER, evaluate_predictions
from fbai.core.splits import ALL_TEST_YEARS, chronological_date_batches, expanding_folds
from fbai.data.loader import canonicalize_source_frame
from fbai.data.schema import validate_canonical_frame
from fbai.evaluation.runner import evaluate_lr52
from fbai.features.build import build_feature_table
from fbai.features.schema import FEATURE_COLUMNS
from fbai.models.logistic import PROBABILITY_COLUMNS, fit_lr52, predict_lr52_proba
from fbai.testing.synthetic import make_synthetic_canonical_matches, make_synthetic_raw_matches


@lru_cache(maxsize=1)
def _cached_feature_table() -> pd.DataFrame:
    canonical = make_synthetic_canonical_matches(
        seed=503,
        season_start_years=(2021, 2022, 2023, 2024, 2025),
        divisions=("E0", "D1"),
        teams_per_division=6,
    )
    return build_feature_table(canonical)


def feature_table() -> pd.DataFrame:
    return _cached_feature_table().copy(deep=True)


def test_runner_preserves_fold_roles_counts_chronology_and_class_order() -> None:
    frame = feature_table()

    report = evaluate_lr52(frame)

    assert report.class_order == CLASS_ORDER
    assert [fold.test_year for fold in report.folds] == list(ALL_TEST_YEARS)
    assert [fold.role.value for fold in report.folds] == [
        "development",
        "development",
        "development",
        "historical_final",
    ]
    for fold in report.folds:
        assert fold.train_end_date < fold.test_start_date
        assert fold.preprocessing_fit_rows == fold.train_rows
        assert fold.model.n_samples == fold.test_rows
    assert report.development.fold_count == 3
    assert report.historical_final.fold_count == 1
    assert report.all_historical_diagnostic.fold_count == 4


def test_runner_metrics_match_direct_calculation() -> None:
    frame = feature_table()
    fold = next(expanding_folds(frame, test_years=(2022,)))
    train = frame.loc[list(fold.train_idx)]
    test = frame.loc[list(fold.test_idx)]
    fitted = fit_lr52(train)
    probabilities = predict_lr52_proba(fitted, test)
    direct = evaluate_predictions(
        test["target_1x2"].astype(str).tolist(),
        probabilities.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(),
    )

    report = evaluate_lr52(frame)
    recorded = report.folds[0].model

    assert recorded.n_samples == direct["n_samples"]
    assert recorded.log_loss == pytest.approx(direct["log_loss"], abs=1e-15)
    assert recorded.brier_score == pytest.approx(direct["brier_score"], abs=1e-15)
    assert recorded.ece == pytest.approx(direct["ece"], abs=1e-15)


def test_shuffled_input_is_equivalent_and_source_is_not_mutated() -> None:
    frame = feature_table()
    before = frame.copy(deep=True)

    expected = evaluate_lr52(frame)
    shuffled = evaluate_lr52(frame.sample(frac=1.0, random_state=73))

    assert shuffled.to_dict() == expected.to_dict()
    pd.testing.assert_frame_equal(frame, before)


def test_existing_split_and_independent_model_fit_are_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = feature_table()
    split_calls: list[tuple[int, ...]] = []
    fit_pipelines: list[int] = []
    fit_max_seasons: list[int] = []
    original_split = runner_module.expanding_folds
    original_fit = runner_module.fit_lr52

    def recording_split(
        matches: pd.DataFrame,
        *,
        test_years: tuple[int, ...],
    ) -> object:
        split_calls.append(tuple(test_years))
        return original_split(matches, test_years=test_years)

    def recording_fit(train: pd.DataFrame, *, config: object) -> object:
        fitted = original_fit(train, config=config)  # type: ignore[arg-type]
        fit_pipelines.append(id(fitted.pipeline))
        fit_max_seasons.append(int(train["SeasonStartYear"].max()))
        return fitted

    monkeypatch.setattr(runner_module, "expanding_folds", recording_split)
    monkeypatch.setattr(runner_module, "fit_lr52", recording_fit)

    evaluate_lr52(frame)

    assert split_calls == [ALL_TEST_YEARS]
    assert len(set(fit_pipelines)) == 4
    assert fit_max_seasons == [2021, 2022, 2023, 2024]


def test_final_fold_values_and_labels_cannot_change_development_results() -> None:
    frame = feature_table()
    expected = evaluate_lr52(frame)
    changed = frame.copy()
    final_mask = changed["SeasonStartYear"].eq(2025)
    changed.loc[final_mask, list(FEATURE_COLUMNS)] = 12345.0
    swapped = changed.loc[final_mask, "FTR"].map({"H": "A", "D": "H", "A": "D"})
    changed.loc[final_mask, "FTR"] = swapped
    changed.loc[final_mask, "target_1x2"] = swapped
    changed.loc[final_mask, "target_home_win"] = swapped.eq("H").astype("Int8")
    changed.loc[final_mask, "target_draw"] = swapped.eq("D").astype("Int8")
    changed.loc[final_mask, "target_away_win"] = swapped.eq("A").astype("Int8")

    rebuilt = evaluate_lr52(changed)

    assert rebuilt.development.to_dict() == expected.development.to_dict()
    assert [fold.to_dict() for fold in rebuilt.folds[:3]] == [
        fold.to_dict() for fold in expected.folds[:3]
    ]


def test_same_date_batches_are_never_split_between_train_and_test() -> None:
    frame = feature_table()
    folds = tuple(expanding_folds(frame))
    batches = chronological_date_batches(frame)

    for fold in folds:
        train = set(fold.train_idx)
        test = set(fold.test_idx)
        for batch in batches:
            batch_set = set(batch)
            assert not (batch_set & train and batch_set & test)


def test_synthetic_raw_to_weighted_report_end_to_end() -> None:
    raw = make_synthetic_raw_matches(
        seed=504,
        season_start_years=(2021, 2022, 2023, 2024, 2025),
        divisions=("E0", "D1"),
        teams_per_division=6,
    )
    canonical = canonicalize_source_frame(raw)
    validate_canonical_frame(canonical)
    features = build_feature_table(canonical)

    report = evaluate_lr52(features)

    assert len(raw) == len(canonical) == len(features) == 300
    assert report.feature_count == 52
    assert report.class_order == ("H", "D", "A")
    assert report.development.model.n_samples == 180
    assert report.historical_final.model.n_samples == 60
