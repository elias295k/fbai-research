"""Expanding chronological LR52 evaluation using the Phase 1 split contract."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from fbai.core.metrics import evaluate_predictions
from fbai.core.splits import ALL_TEST_YEARS, STABLE_ORDER_KEY, expanding_folds
from fbai.evaluation.baselines import fit_training_prior, uniform_probabilities
from fbai.evaluation.report import (
    EvaluationReport,
    FoldEvaluation,
    MetricRecord,
    build_evaluation_report,
)
from fbai.features.checks import validate_feature_table
from fbai.features.schema import FEATURE_COLUMNS
from fbai.models.logistic import (
    PROBABILITY_COLUMNS,
    LR52Config,
    fit_lr52,
    predict_lr52_proba,
)


def _metric_record(labels: pd.Series, probabilities: pd.DataFrame, *, n_bins: int) -> MetricRecord:
    metrics = evaluate_predictions(
        labels.astype(str).tolist(),
        probabilities.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(dtype="float64"),
        n_bins=n_bins,
    )
    return MetricRecord.from_mapping(metrics)


def _date_text(value: object) -> str:
    return str(pd.Timestamp(value).date().isoformat())


def evaluate_lr52(
    feature_table: pd.DataFrame,
    *,
    config: LR52Config | None = None,
    test_years: Sequence[int] = ALL_TEST_YEARS,
    n_bins: int = 10,
) -> EvaluationReport:
    """Evaluate independently fitted LR52 pipelines on expanding folds."""

    ordered = (
        feature_table.sort_values(list(STABLE_ORDER_KEY), kind="mergesort")
        .reset_index(drop=True)
        .copy(deep=True)
    )
    validate_feature_table(ordered, expected_row_count=len(feature_table))
    resolved = config or LR52Config()
    fold_records: list[FoldEvaluation] = []

    for fold in expanding_folds(ordered, test_years=test_years):
        train = ordered.loc[list(fold.train_idx)].copy(deep=True)
        test = ordered.loc[list(fold.test_idx)].copy(deep=True)
        latest_train = train["MatchDate"].max()
        earliest_test = test["MatchDate"].min()
        if latest_train >= earliest_test:
            raise ValueError(f"Fold {fold.name} violates strict train-before-test chronology")

        fitted = fit_lr52(train, config=resolved)
        model_probabilities = predict_lr52_proba(fitted, test)
        uniform = uniform_probabilities(test)
        prior = fit_training_prior(train).predict(test)
        labels = test["target_1x2"]
        fold_records.append(
            FoldEvaluation(
                fold_name=fold.name,
                test_year=fold.test_year,
                role=fold.role,
                train_start_date=_date_text(train["MatchDate"].min()),
                train_end_date=_date_text(latest_train),
                test_start_date=_date_text(earliest_test),
                test_end_date=_date_text(test["MatchDate"].max()),
                train_rows=len(train),
                test_rows=len(test),
                model_configuration_id=resolved.identifier,
                preprocessing_fit_rows=len(train),
                model_iterations=fitted.iterations,
                model=_metric_record(labels, model_probabilities, n_bins=n_bins),
                uniform=_metric_record(labels, uniform, n_bins=n_bins),
                training_prior=_metric_record(labels, prior, n_bins=n_bins),
            )
        )

    return build_evaluation_report(
        tuple(fold_records),
        model_configuration_id=resolved.identifier,
        feature_count=len(FEATURE_COLUMNS),
    )
