from __future__ import annotations

import hashlib
from functools import lru_cache

import pandas as pd
import pytest

import fbai.evaluation.comparison as comparison_module
from fbai.core.metrics import weighted_fold_summary
from fbai.core.splits import ALL_TEST_YEARS
from fbai.data.loader import canonicalize_source_frame
from fbai.evaluation.comparison import (
    MarketAlignmentError,
    align_closing_market,
    evaluate_lr52_vs_closing_market,
)
from fbai.evaluation.market import MARKET_NATURAL_KEY
from fbai.features.build import build_feature_table
from fbai.features.schema import FEATURE_COLUMNS, feature_columns
from fbai.models.preprocessing import LR52InputError, select_lr52_features
from fbai.testing.market import make_synthetic_closing_market
from fbai.testing.synthetic import make_synthetic_canonical_matches, make_synthetic_raw_matches


@lru_cache(maxsize=1)
def _cached_feature_table() -> pd.DataFrame:
    canonical = make_synthetic_canonical_matches(
        seed=611,
        season_start_years=(2021, 2022, 2023, 2024, 2025),
        divisions=("SYN1", "SYN2"),
        teams_per_division=6,
    )
    return build_feature_table(canonical)


def feature_table() -> pd.DataFrame:
    return _cached_feature_table().copy(deep=True)


def test_exact_one_to_one_alignment_reports_both_unmatched_directions() -> None:
    features = feature_table()
    market = make_synthetic_closing_market(features, seed=612)
    extra = market.iloc[[0]].copy()
    extra["MatchDate"] = pd.Timestamp("2030-01-01")
    extra["HomeTeam"] = "SYN1 Synthetic Club 97"
    extra["AwayTeam"] = "SYN1 Synthetic Club 98"
    changed = pd.concat([market.iloc[1:], extra], ignore_index=True)

    aligned, coverage = align_closing_market(features, changed)

    assert len(aligned) == len(features) - 1
    assert coverage.unmatched_match_keys == 1
    assert coverage.unmatched_market_keys == 1
    assert coverage.aligned_rows == len(features) - 1
    assert coverage.coverage_percentage == pytest.approx(
        100.0 * (len(features) - 1) / len(features)
    )
    assert sum(item.candidate_match_rows for item in coverage.by_division) == len(features)
    assert sum(item.candidate_match_rows for item in coverage.by_test_year) == len(features)


def test_incomplete_and_invalid_odds_are_excluded_and_counted() -> None:
    features = feature_table()
    market = make_synthetic_closing_market(features, seed=613)
    market.loc[0, "ClosingDrawOdds"] = None
    market.loc[1, "ClosingAwayOdds"] = 1.0

    aligned, coverage = align_closing_market(features, market)

    assert len(aligned) == len(features) - 2
    assert coverage.incomplete_odds_rows == 1
    assert coverage.invalid_odds_rows == 1
    assert coverage.valid_market_rows == len(features) - 2
    assert coverage.unmatched_match_keys == 2


def test_duplicate_market_records_fail_instead_of_being_selected() -> None:
    features = feature_table()
    market = make_synthetic_closing_market(features, seed=614, mode="duplicate")

    with pytest.raises(ValueError, match="not unique"):
        align_closing_market(features, market)


def test_alignment_is_independent_of_both_input_orders_and_does_not_mutate() -> None:
    features = feature_table()
    market = make_synthetic_closing_market(features, seed=615)
    feature_before = features.copy(deep=True)
    market_before = market.copy(deep=True)

    expected, expected_coverage = align_closing_market(features, market)
    actual, actual_coverage = align_closing_market(
        features.sample(frac=1.0, random_state=4),
        market.sample(frac=1.0, random_state=5),
    )

    pd.testing.assert_frame_equal(actual, expected)
    assert actual_coverage.to_dict() == expected_coverage.to_dict()
    pd.testing.assert_frame_equal(features, feature_before)
    pd.testing.assert_frame_equal(market, market_before)


def test_comparison_preserves_roles_chronology_and_identical_aligned_counts() -> None:
    features = feature_table()
    market = make_synthetic_closing_market(features, seed=616, mode="missing_rows")

    report = evaluate_lr52_vs_closing_market(features, market)

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
        assert fold.aligned_lr52.n_samples == fold.market.n_samples
        assert fold.aligned_lr52.n_samples == fold.uniform.n_samples
        assert fold.aligned_lr52.n_samples == fold.training_prior.n_samples
        assert fold.aligned_rows == fold.aligned_lr52.n_samples
        assert fold.unaligned_rows == fold.candidate_test_rows - fold.aligned_rows
        assert fold.full_test_lr52.n_samples == fold.candidate_test_rows
    assert report.development.fold_count == 3
    assert report.historical_final.fold_count == 1


def test_market_advantage_sign_and_weighted_aggregation_are_unambiguous() -> None:
    features = feature_table()
    market = make_synthetic_closing_market(features, seed=617, mode="missing_rows")

    report = evaluate_lr52_vs_closing_market(features, market)
    direct = weighted_fold_summary([fold.market.to_dict() for fold in report.folds[:3]])

    assert report.development.market.log_loss == pytest.approx(direct["log_loss"], abs=1e-15)
    assert report.development.market.n_samples == direct["n_samples"]
    assert report.development.market_advantage_log_loss == pytest.approx(
        report.development.aligned_lr52.log_loss - report.development.market.log_loss,
        abs=1e-15,
    )
    for fold in report.folds:
        assert fold.market_advantage_log_loss == pytest.approx(
            fold.aligned_lr52.log_loss - fold.market.log_loss,
            abs=1e-15,
        )


def test_full_test_and_aligned_subset_lr52_remain_distinct_contexts() -> None:
    features = feature_table()
    market = make_synthetic_closing_market(features, seed=618, mode="missing_rows")

    report = evaluate_lr52_vs_closing_market(features, market)

    assert report.all_historical_diagnostic.full_test_lr52.n_samples > (
        report.all_historical_diagnostic.aligned_lr52.n_samples
    )
    assert any(fold.full_test_lr52.n_samples > fold.aligned_lr52.n_samples for fold in report.folds)


def test_changing_odds_changes_only_market_metrics_when_coverage_is_fixed() -> None:
    features = feature_table()
    market = make_synthetic_closing_market(features, seed=619)
    changed = market.copy(deep=True)
    changed["ClosingHomeOdds"] *= 1.4

    expected = evaluate_lr52_vs_closing_market(features, market)
    actual = evaluate_lr52_vs_closing_market(features, changed)

    assert [fold.aligned_lr52.to_dict() for fold in actual.folds] == [
        fold.aligned_lr52.to_dict() for fold in expected.folds
    ]
    assert [fold.full_test_lr52.to_dict() for fold in actual.folds] == [
        fold.full_test_lr52.to_dict() for fold in expected.folds
    ]
    assert [fold.market.log_loss for fold in actual.folds] != [
        fold.market.log_loss for fold in expected.folds
    ]


def test_final_fold_odds_cannot_change_development_results() -> None:
    features = feature_table()
    market = make_synthetic_closing_market(features, seed=620)
    final_keys = {
        tuple(row)
        for row in features.loc[
            features["SeasonStartYear"].eq(2025),
            list(MARKET_NATURAL_KEY),
        ].itertuples(index=False, name=None)
    }
    changed = market.copy(deep=True)
    mask = pd.Series(
        [
            tuple(row) in final_keys
            for row in changed.loc[:, list(MARKET_NATURAL_KEY)].itertuples(
                index=False,
                name=None,
            )
        ],
        index=changed.index,
    )
    changed.loc[mask, "ClosingAwayOdds"] *= 1.5

    expected = evaluate_lr52_vs_closing_market(features, market)
    actual = evaluate_lr52_vs_closing_market(features, changed)

    assert actual.development.to_dict() == expected.development.to_dict()
    assert [fold.to_dict() for fold in actual.folds[:3]] == [
        fold.to_dict() for fold in expected.folds[:3]
    ]


def test_each_fold_fits_an_independent_lr52_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = feature_table()
    market = make_synthetic_closing_market(features, seed=621)
    original_fit = comparison_module.fit_lr52
    pipeline_ids: list[int] = []
    train_max_seasons: list[int] = []

    def recording_fit(train: pd.DataFrame, *, config: object) -> object:
        fitted = original_fit(train, config=config)  # type: ignore[arg-type]
        pipeline_ids.append(id(fitted.pipeline))
        train_max_seasons.append(int(train["SeasonStartYear"].max()))
        return fitted

    monkeypatch.setattr(comparison_module, "fit_lr52", recording_fit)
    evaluate_lr52_vs_closing_market(features, market)

    assert len(set(pipeline_ids)) == 4
    assert train_max_seasons == [2021, 2022, 2023, 2024]


def test_changing_test_labels_changes_metrics_not_fold_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = feature_table()
    market = make_synthetic_closing_market(features, seed=622)
    changed = features.copy(deep=True)
    final = changed["SeasonStartYear"].eq(2025)
    labels = changed.loc[final, "FTR"].map({"H": "A", "D": "H", "A": "D"})
    changed.loc[final, "FTR"] = labels
    changed.loc[final, "target_1x2"] = labels
    changed.loc[final, "target_home_win"] = labels.eq("H").astype("Int8")
    changed.loc[final, "target_draw"] = labels.eq("D").astype("Int8")
    changed.loc[final, "target_away_win"] = labels.eq("A").astype("Int8")
    original_predict = comparison_module.predict_lr52_proba
    captured: list[pd.DataFrame] = []

    def recording_predict(model: object, frame: pd.DataFrame) -> pd.DataFrame:
        probabilities = original_predict(model, frame)  # type: ignore[arg-type]
        captured.append(probabilities.copy(deep=True))
        return probabilities

    monkeypatch.setattr(comparison_module, "predict_lr52_proba", recording_predict)
    expected = evaluate_lr52_vs_closing_market(features, market)
    first_predictions = [item.copy(deep=True) for item in captured]
    captured.clear()
    actual = evaluate_lr52_vs_closing_market(changed, market)

    for first, second in zip(first_predictions, captured, strict=True):
        pd.testing.assert_frame_equal(first, second)
    assert actual.development.to_dict() == expected.development.to_dict()
    assert actual.historical_final.aligned_lr52.log_loss != (
        expected.historical_final.aligned_lr52.log_loss
    )


def test_shuffled_comparison_is_equivalent_and_inputs_are_unchanged() -> None:
    features = feature_table()
    market = make_synthetic_closing_market(features, seed=623)
    feature_before = features.copy(deep=True)
    market_before = market.copy(deep=True)

    expected = evaluate_lr52_vs_closing_market(features, market)
    actual = evaluate_lr52_vs_closing_market(
        features.sample(frac=1.0, random_state=2),
        market.sample(frac=1.0, random_state=3),
    )

    assert actual.to_dict() == expected.to_dict()
    pd.testing.assert_frame_equal(features, feature_before)
    pd.testing.assert_frame_equal(market, market_before)


def test_market_fields_cannot_enter_lr52_and_feature_contract_stays_fixed() -> None:
    features = feature_table()
    market = make_synthetic_closing_market(features, seed=624)
    with_odds = features.assign(ClosingHomeOdds=2.0)
    with_probabilities = features.assign(probability_home=0.4)

    with pytest.raises(LR52InputError, match="Unexpected"):
        select_lr52_features(with_odds)
    with pytest.raises(LR52InputError, match="Unexpected"):
        select_lr52_features(with_probabilities)
    assert feature_columns() == FEATURE_COLUMNS
    assert len(feature_columns()) == 52
    fingerprint = hashlib.sha256("\0".join(feature_columns()).encode()).hexdigest()
    assert fingerprint == "09b9204c283e8adf7e91e98cb1a547183b3f832cc95f815b9845208be822de78"
    report = evaluate_lr52_vs_closing_market(features, market)
    assert report.feature_count == 52


def test_synthetic_raw_to_aligned_market_report_end_to_end() -> None:
    raw = make_synthetic_raw_matches(
        seed=625,
        season_start_years=(2021, 2022, 2023, 2024, 2025),
        divisions=("SYN1", "SYN2"),
        teams_per_division=6,
    )
    canonical = canonicalize_source_frame(raw)
    features = build_feature_table(canonical)
    market = make_synthetic_closing_market(features, seed=626, mode="missing_rows")

    report = evaluate_lr52_vs_closing_market(features, market)

    assert report.market_coverage.candidate_match_rows == len(features)
    assert report.market_coverage.aligned_rows == len(market)
    assert report.all_historical_diagnostic.aligned_lr52.n_samples == (
        report.all_historical_diagnostic.market.n_samples
    )
    assert report.class_order == ("H", "D", "A")
    assert report.feature_count == 52


def test_no_valid_market_rows_raise_an_explicit_alignment_error() -> None:
    features = feature_table()
    market = make_synthetic_closing_market(features.iloc[[0]], seed=627, mode="invalid")

    with pytest.raises(MarketAlignmentError, match="no valid rows"):
        align_closing_market(features, market)
