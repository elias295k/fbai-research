"""Chronological pseudo-xG candidate evaluation against LR52 on identical rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fbai.core.metrics import CLASS_ORDER, evaluate_predictions, weighted_fold_summary
from fbai.core.splits import (
    ALL_TEST_YEARS,
    DEVELOPMENT_TEST_YEARS,
    STABLE_ORDER_KEY,
    FoldRole,
    expanding_folds,
)
from fbai.data.schema import NATURAL_KEY, validate_canonical_frame
from fbai.evaluation.report import MetricRecord
from fbai.features.checks import validate_feature_table
from fbai.features.schema import FEATURE_COLUMNS
from fbai.models.logistic import PROBABILITY_COLUMNS, fit_lr52, predict_lr52_proba
from fbai.research.common import ResearchGate, ResearchGateResult
from fbai.research.pseudo_xg.config import PseudoXGConfig
from fbai.research.pseudo_xg.features import (
    PSEUDO_XG_FEATURE_COLUMNS,
    build_pseudo_xg_feature_table,
    build_walk_forward_training_feature_table,
)
from fbai.research.pseudo_xg.model import (
    fit_pseudo_xg_candidate,
    fit_pseudo_xg_estimator,
    predict_pseudo_xg_candidate_proba,
)

EXPERIMENT_ID = "pseudo_xg_lr64_historical_reproduction"
CANDIDATE_NAME = "LR52 + 12 historical pseudo-xG features"


def _date_text(value: object) -> str:
    return str(pd.Timestamp(value).date().isoformat())


def _metrics(
    labels: pd.Series,
    probabilities: np.ndarray | pd.DataFrame,
    *,
    n_bins: int,
) -> MetricRecord:
    values = (
        probabilities.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        if isinstance(probabilities, pd.DataFrame)
        else np.asarray(probabilities, dtype=np.float64)
    )
    return MetricRecord.from_mapping(
        evaluate_predictions(labels.astype(str).tolist(), values, n_bins=n_bins)
    )


def _aligned_inputs(
    feature_table: pd.DataFrame,
    canonical_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_feature_table(feature_table, expected_row_count=len(feature_table))
    validate_canonical_frame(canonical_matches)
    ordered_features = (
        feature_table.sort_values(list(STABLE_ORDER_KEY), kind="mergesort")
        .reset_index(drop=True)
        .copy(deep=True)
    )
    ordered_canonical = (
        canonical_matches.sort_values(list(STABLE_ORDER_KEY), kind="mergesort")
        .reset_index(drop=True)
        .copy(deep=True)
    )
    try:
        pd.testing.assert_frame_equal(
            ordered_features.loc[:, list(NATURAL_KEY)],
            ordered_canonical.loc[:, list(NATURAL_KEY)],
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        raise ValueError(
            "Feature and canonical tables must contain identical natural keys"
        ) from exc
    return ordered_features, ordered_canonical


@dataclass(frozen=True, slots=True)
class PseudoXGFoldEvaluation:
    """Aggregate-only comparison for one independently fitted outer fold."""

    fold_name: str
    test_year: int
    role: FoldRole
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    train_rows: int
    test_rows: int
    estimator_training_match_rows: int
    estimator_training_side_rows: int
    estimator_iterations: int
    candidate_preprocessing_fit_rows: int
    candidate_iterations: tuple[int, ...]
    identical_test_rows: bool
    lr52: MetricRecord
    candidate: MetricRecord
    candidate_improvement_log_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_name": self.fold_name,
            "test_year": self.test_year,
            "role": self.role.value,
            "train_date_range": [self.train_start_date, self.train_end_date],
            "test_date_range": [self.test_start_date, self.test_end_date],
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "estimator_training_match_rows": self.estimator_training_match_rows,
            "estimator_training_side_rows": self.estimator_training_side_rows,
            "estimator_iterations": self.estimator_iterations,
            "candidate_preprocessing_fit_rows": self.candidate_preprocessing_fit_rows,
            "candidate_iterations": list(self.candidate_iterations),
            "identical_test_rows": self.identical_test_rows,
            "lr52": self.lr52.to_dict(),
            "candidate": self.candidate.to_dict(),
            "candidate_improvement_log_loss": self.candidate_improvement_log_loss,
        }


@dataclass(frozen=True, slots=True)
class PseudoXGAggregate:
    """Sample-count-weighted metrics for a named fold view."""

    name: str
    fold_count: int
    lr52: MetricRecord
    candidate: MetricRecord
    candidate_improvement_log_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fold_count": self.fold_count,
            "lr52": self.lr52.to_dict(),
            "candidate": self.candidate.to_dict(),
            "candidate_improvement_log_loss": self.candidate_improvement_log_loss,
        }


def aggregate_pseudo_xg_folds(
    name: str,
    folds: tuple[PseudoXGFoldEvaluation, ...],
) -> PseudoXGAggregate:
    """Aggregate folds using test sample counts, never an unweighted fold mean."""

    if not folds:
        raise ValueError(f"Aggregate {name} requires at least one fold")
    lr52 = MetricRecord.from_mapping(weighted_fold_summary([fold.lr52.to_dict() for fold in folds]))
    candidate = MetricRecord.from_mapping(
        weighted_fold_summary([fold.candidate.to_dict() for fold in folds])
    )
    return PseudoXGAggregate(
        name=name,
        fold_count=len(folds),
        lr52=lr52,
        candidate=candidate,
        candidate_improvement_log_loss=lr52.log_loss - candidate.log_loss,
    )


@dataclass(frozen=True, slots=True)
class PseudoXGComparisonReport:
    """Immutable aggregate-only report for the isolated research candidate."""

    schema_version: str
    experiment_id: str
    candidate_name: str
    configuration: PseudoXGConfig
    class_order: tuple[str, str, str]
    base_feature_count: int
    pseudo_xg_feature_names: tuple[str, ...]
    folds: tuple[PseudoXGFoldEvaluation, ...]
    development: PseudoXGAggregate
    historical_final: PseudoXGAggregate
    all_historical_diagnostic: PseudoXGAggregate
    success_gate: ResearchGateResult

    def __post_init__(self) -> None:
        if self.class_order != CLASS_ORDER:
            raise ValueError(f"class_order must be exactly {CLASS_ORDER}")
        if self.base_feature_count != len(FEATURE_COLUMNS):
            raise ValueError("base feature count must preserve the exact LR52 tuple")
        if self.pseudo_xg_feature_names != PSEUDO_XG_FEATURE_COLUMNS:
            raise ValueError("pseudo-xG feature names must match the explicit contract")

    @property
    def disposition(self) -> str:
        return self.success_gate.disposition

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "candidate_name": self.candidate_name,
            "candidate_configuration": self.configuration.to_dict(),
            "class_order": list(self.class_order),
            "base_feature_count": self.base_feature_count,
            "pseudo_xg_feature_count": len(self.pseudo_xg_feature_names),
            "pseudo_xg_feature_names": list(self.pseudo_xg_feature_names),
            "folds": [fold.to_dict() for fold in self.folds],
            "development": self.development.to_dict(),
            "historical_final": self.historical_final.to_dict(),
            "all_historical_diagnostic": self.all_historical_diagnostic.to_dict(),
            "success_gate": self.success_gate.to_dict(),
            "verdict": "passed" if self.success_gate.passed else "failed",
            "disposition": self.disposition,
        }


def evaluate_pseudo_xg_candidate(
    feature_table: pd.DataFrame,
    canonical_matches: pd.DataFrame,
    *,
    config: PseudoXGConfig | None = None,
    test_years: tuple[int, ...] = ALL_TEST_YEARS,
    n_bins: int = 10,
) -> PseudoXGComparisonReport:
    """Re-fit and compare the source primary pseudo-xG candidate in every fold."""

    if tuple(test_years) != ALL_TEST_YEARS:
        raise ValueError("The predefined pseudo-xG gate requires the complete 2022-2025 protocol")
    original_features = feature_table.copy(deep=True)
    original_canonical = canonical_matches.copy(deep=True)
    ordered, canonical = _aligned_inputs(feature_table, canonical_matches)
    resolved = config or PseudoXGConfig()
    records: list[PseudoXGFoldEvaluation] = []

    for fold in expanding_folds(ordered, test_years=test_years):
        train = ordered.loc[list(fold.train_idx)].copy(deep=True)
        test = ordered.loc[list(fold.test_idx)].copy(deep=True)
        canonical_train = canonical.loc[list(fold.train_idx)].copy(deep=True)
        if train["MatchDate"].max() >= test["MatchDate"].min():
            raise ValueError(f"Fold {fold.name} violates strict train-before-test chronology")

        estimator = fit_pseudo_xg_estimator(canonical_train, config=resolved)
        training_pseudo_features = build_walk_forward_training_feature_table(
            canonical_train,
            config=resolved,
        )
        test_pseudo_features = build_pseudo_xg_feature_table(test, canonical, estimator)
        augmented_train = train.merge(
            training_pseudo_features,
            on=list(NATURAL_KEY),
            how="left",
            sort=False,
            validate="one_to_one",
        )
        augmented_test = test.merge(
            test_pseudo_features,
            on=list(NATURAL_KEY),
            how="left",
            sort=False,
            validate="one_to_one",
        )
        lr52_probabilities = predict_lr52_proba(fit_lr52(train), test)
        candidate_model = fit_pseudo_xg_candidate(augmented_train, config=resolved)
        candidate_probabilities = predict_pseudo_xg_candidate_proba(
            candidate_model,
            augmented_test,
        )
        identical_test_rows = len(lr52_probabilities) == len(candidate_probabilities) == len(
            test
        ) and test.loc[:, list(NATURAL_KEY)].reset_index(drop=True).equals(
            augmented_test.loc[:, list(NATURAL_KEY)].reset_index(drop=True)
        )
        if not identical_test_rows:
            raise ValueError("pseudo-xG candidate and LR52 test keys are not identical")

        labels = test["target_1x2"]
        lr52_metrics = _metrics(labels, lr52_probabilities, n_bins=n_bins)
        candidate_metrics = _metrics(labels, candidate_probabilities, n_bins=n_bins)
        records.append(
            PseudoXGFoldEvaluation(
                fold_name=fold.name,
                test_year=fold.test_year,
                role=fold.role,
                train_start_date=_date_text(train["MatchDate"].min()),
                train_end_date=_date_text(train["MatchDate"].max()),
                test_start_date=_date_text(test["MatchDate"].min()),
                test_end_date=_date_text(test["MatchDate"].max()),
                train_rows=len(train),
                test_rows=len(test),
                estimator_training_match_rows=estimator.training_match_rows,
                estimator_training_side_rows=estimator.training_side_rows,
                estimator_iterations=estimator.iterations,
                candidate_preprocessing_fit_rows=len(augmented_train),
                candidate_iterations=candidate_model.iterations,
                identical_test_rows=True,
                lr52=lr52_metrics,
                candidate=candidate_metrics,
                candidate_improvement_log_loss=(lr52_metrics.log_loss - candidate_metrics.log_loss),
            )
        )

    folds = tuple(records)
    development_folds = tuple(fold for fold in folds if fold.test_year in DEVELOPMENT_TEST_YEARS)
    final_folds = tuple(fold for fold in folds if fold.role is FoldRole.HISTORICAL_FINAL)
    if len(development_folds) != 3 or len(final_folds) != 1:
        raise ValueError("pseudo-xG evaluation requires three development and one final fold")
    development = aggregate_pseudo_xg_folds("development", development_folds)
    historical_final = aggregate_pseudo_xg_folds("historical_final", final_folds)
    all_historical = aggregate_pseudo_xg_folds("all_historical_diagnostic", folds)
    gate = ResearchGate().evaluate(
        lr52_log_loss=development.lr52.log_loss,
        candidate_log_loss=development.candidate.log_loss,
        fold_improvements=tuple(fold.candidate_improvement_log_loss for fold in development_folds),
        historical_final_improvement_log_loss=(historical_final.candidate_improvement_log_loss),
        passing_disposition="XG_SIGNAL_CANDIDATE",
        failing_disposition="XG_SIGNAL_REJECTED_FOR_NOW",
    )
    report = PseudoXGComparisonReport(
        schema_version="1.0",
        experiment_id=EXPERIMENT_ID,
        candidate_name=CANDIDATE_NAME,
        configuration=resolved,
        class_order=CLASS_ORDER,
        base_feature_count=len(FEATURE_COLUMNS),
        pseudo_xg_feature_names=PSEUDO_XG_FEATURE_COLUMNS,
        folds=folds,
        development=development,
        historical_final=historical_final,
        all_historical_diagnostic=all_historical,
        success_gate=gate,
    )
    pd.testing.assert_frame_equal(feature_table, original_features)
    pd.testing.assert_frame_equal(canonical_matches, original_canonical)
    return report
