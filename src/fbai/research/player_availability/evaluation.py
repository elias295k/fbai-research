"""Source-verified historical LR52 plus prior availability audit."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fbai.core.leakage import NATURAL_KEY, assert_model_inputs_safe
from fbai.core.metrics import CLASS_ORDER, evaluate_predictions, validate_probabilities
from fbai.core.splits import (
    ALL_TEST_YEARS,
    DEVELOPMENT_TEST_YEARS,
    STABLE_ORDER_KEY,
    FoldRole,
    expanding_folds,
)
from fbai.evaluation.report import MetricRecord
from fbai.features.schema import FEATURE_COLUMNS, FEATURE_TABLE_COLUMNS
from fbai.models.logistic import PROBABILITY_COLUMNS, fit_lr52, predict_lr52_proba
from fbai.research.common import CandidateAggregate, ResearchGate, ResearchGateResult
from fbai.research.player_availability.alignment import (
    AvailabilityAlignmentReport,
    align_availability_scope,
    prepare_availability_sources,
)
from fbai.research.player_availability.features import build_prior_availability_features
from fbai.research.player_availability.schema import (
    AVAILABILITY_CLASSIFIER_CONFIG,
    PRIOR_AVAILABILITY_FEATURES,
    TIMING_AUDIT,
    AvailabilityClassifierConfig,
)

EXPERIMENT_ID = "prior_player_availability_lr115_historical_reproduction"
CANDIDATE_NAME = "LR52 + strictly-prior player availability"


class AvailabilityEvaluationError(ValueError):
    """Raised when the frozen availability evaluation protocol is violated."""


@dataclass(frozen=True, slots=True)
class FittedAvailabilityClassifier:
    """Fitted source-equivalent pipeline and auditable class ordering."""

    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    sklearn_classes: tuple[str, ...]
    iterations: tuple[int, ...]
    parameter_count: int


def _candidate_columns() -> tuple[str, ...]:
    return (*FEATURE_COLUMNS, *PRIOR_AVAILABILITY_FEATURES)


def _candidate_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    columns = _candidate_columns()
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AvailabilityEvaluationError(
            f"availability candidate is missing: {', '.join(missing)}"
        )
    assert_model_inputs_safe(columns, approved_pre_features=columns)
    for column in columns:
        if not is_numeric_dtype(frame[column].dtype):
            raise AvailabilityEvaluationError(
                f"availability candidate input {column} must be numeric"
            )
    selected = frame.loc[:, list(columns)].copy(deep=True)
    if np.isinf(selected.to_numpy(dtype=float, na_value=np.nan)).any():
        raise AvailabilityEvaluationError("availability candidate inputs contain infinity")
    return selected


def _fit_candidate(
    train: pd.DataFrame,
    config: AvailabilityClassifierConfig,
) -> FittedAvailabilityClassifier:
    target = train["target_1x2"].astype(str)
    if frozenset(target) != frozenset(CLASS_ORDER):
        raise AvailabilityEvaluationError("candidate training requires all H/D/A classes")
    classifier = LogisticRegression(
        C=config.regularization_c,
        solver=config.solver,
        max_iter=config.max_iter,
        fit_intercept=config.fit_intercept,
        class_weight=config.class_weight,
        random_state=config.random_seed,
        tol=config.tolerance,
    )
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logistic", classifier),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(_candidate_inputs(train), target)
    iterations = tuple(int(value) for value in classifier.n_iter_)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught) or any(
        iteration >= config.max_iter for iteration in iterations
    ):
        raise AvailabilityEvaluationError("availability candidate failed to converge")
    classes = tuple(str(value) for value in classifier.classes_)
    if frozenset(classes) != frozenset(CLASS_ORDER):
        raise AvailabilityEvaluationError("availability candidate class set is invalid")
    return FittedAvailabilityClassifier(
        pipeline=pipeline,
        feature_columns=_candidate_columns(),
        sklearn_classes=classes,
        iterations=iterations,
        parameter_count=config.parameter_count,
    )


def _predict_candidate(
    fitted: FittedAvailabilityClassifier,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if fitted.feature_columns != _candidate_columns():
        raise AvailabilityEvaluationError("candidate feature contract changed")
    raw = np.asarray(fitted.pipeline.predict_proba(_candidate_inputs(frame)), dtype=float)
    order = [fitted.sklearn_classes.index(label) for label in CLASS_ORDER]
    probabilities = validate_probabilities(raw[:, order])
    return pd.DataFrame(
        probabilities,
        index=frame.index.copy(),
        columns=PROBABILITY_COLUMNS,
    )


def _metrics(
    labels: pd.Series,
    probabilities: np.ndarray | pd.DataFrame,
    *,
    n_bins: int,
) -> MetricRecord:
    values = (
        probabilities.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        if isinstance(probabilities, pd.DataFrame)
        else np.asarray(probabilities, dtype=float)
    )
    return MetricRecord.from_mapping(
        evaluate_predictions(labels.astype(str).tolist(), values, n_bins=n_bins)
    )


def _date_text(value: object) -> str:
    return str(pd.Timestamp(value).date().isoformat())


@dataclass(frozen=True, slots=True)
class FeatureCoverageRecord:
    """Aggregate availability-cell completeness without player-level output."""

    match_rows: int
    observed_cells: int
    total_cells: int

    @property
    def observed_share(self) -> float:
        return self.observed_cells / self.total_cells if self.total_cells else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "match_rows": self.match_rows,
            "observed_cells": self.observed_cells,
            "total_cells": self.total_cells,
            "observed_share": self.observed_share,
        }


def _feature_coverage(frame: pd.DataFrame) -> FeatureCoverageRecord:
    values = frame.loc[:, list(PRIOR_AVAILABILITY_FEATURES)].to_numpy(dtype=float)
    return FeatureCoverageRecord(
        match_rows=len(frame),
        observed_cells=int(np.isfinite(values).sum()),
        total_cells=int(values.size),
    )


@dataclass(frozen=True, slots=True)
class AvailabilityFoldEvaluation:
    """One identical-row chronological LR52/candidate comparison."""

    fold_name: str
    test_year: int
    role: FoldRole
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    train_rows: int
    test_rows: int
    preprocessing_fit_rows: int
    classifier_iterations: tuple[int, ...]
    identical_test_rows: bool
    same_date_isolation: bool
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
            "preprocessing_fit_rows": self.preprocessing_fit_rows,
            "classifier_iterations": list(self.classifier_iterations),
            "identical_test_rows": self.identical_test_rows,
            "same_date_isolation": self.same_date_isolation,
            "lr52": self.lr52.to_dict(),
            "candidate": self.candidate.to_dict(),
            "candidate_improvement_log_loss": self.candidate_improvement_log_loss,
        }


def _aggregate(
    name: str,
    folds: tuple[AvailabilityFoldEvaluation, ...],
) -> CandidateAggregate:
    from fbai.core.metrics import weighted_fold_summary

    if not folds:
        raise ValueError(f"Aggregate {name} requires at least one fold")
    lr52 = MetricRecord.from_mapping(weighted_fold_summary([fold.lr52.to_dict() for fold in folds]))
    candidate = MetricRecord.from_mapping(
        weighted_fold_summary([fold.candidate.to_dict() for fold in folds])
    )
    return CandidateAggregate(
        name=name,
        fold_count=len(folds),
        lr52=lr52,
        candidate=candidate,
        candidate_improvement_log_loss=lr52.log_loss - candidate.log_loss,
    )


@dataclass(frozen=True, slots=True)
class AvailabilityComparisonReport:
    """Immutable aggregate-only result of the frozen availability audit."""

    schema_version: str
    experiment_id: str
    candidate_name: str
    class_order: tuple[str, str, str]
    stable_feature_count: int
    availability_feature_count: int
    candidate_input_count: int
    classifier_configuration: AvailabilityClassifierConfig
    timing_audit: tuple[dict[str, str | list[str]], ...]
    alignment: AvailabilityAlignmentReport
    feature_coverage_by_division: dict[str, FeatureCoverageRecord]
    feature_coverage_by_season: dict[str, FeatureCoverageRecord]
    feature_coverage_by_fold: dict[str, FeatureCoverageRecord]
    folds: tuple[AvailabilityFoldEvaluation, ...]
    development: CandidateAggregate
    historical_final: CandidateAggregate
    all_historical_diagnostic: CandidateAggregate
    success_gate: ResearchGateResult
    final_evaluated_once: bool
    individual_player_rows_exported: bool
    disposition: str

    def __post_init__(self) -> None:
        if self.class_order != CLASS_ORDER:
            raise ValueError(f"class_order must be exactly {CLASS_ORDER}")
        if self.stable_feature_count != 52 or self.availability_feature_count != 63:
            raise ValueError("availability audit must preserve LR52 and add exactly 63")
        if self.candidate_input_count != 115:
            raise ValueError("availability candidate must have 115 inputs")
        if self.individual_player_rows_exported:
            raise ValueError("availability report cannot export player-level rows")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "candidate_name": self.candidate_name,
            "class_order": list(self.class_order),
            "stable_feature_count": self.stable_feature_count,
            "stable_feature_fingerprint_unchanged": True,
            "availability_feature_count": self.availability_feature_count,
            "candidate_input_count": self.candidate_input_count,
            "classifier_configuration": self.classifier_configuration.to_dict(),
            "timing_audit": list(self.timing_audit),
            "excluded_target_information": [
                "target official lineup and bench (timing unknown)",
                "target participation, minutes, substitutions, injuries, and suspensions",
            ],
            "alignment": self.alignment.to_dict(),
            "feature_coverage": {
                "by_division": {
                    key: value.to_dict() for key, value in self.feature_coverage_by_division.items()
                },
                "by_season": {
                    key: value.to_dict() for key, value in self.feature_coverage_by_season.items()
                },
                "by_fold": {
                    key: value.to_dict() for key, value in self.feature_coverage_by_fold.items()
                },
            },
            "folds": [fold.to_dict() for fold in self.folds],
            "development": self.development.to_dict(),
            "historical_final": self.historical_final.to_dict(),
            "all_historical_diagnostic": self.all_historical_diagnostic.to_dict(),
            "success_gate": self.success_gate.to_dict(),
            "final_evaluated_once": self.final_evaluated_once,
            "individual_player_rows_exported": self.individual_player_rows_exported,
            "verdict": "passed" if self.success_gate.passed else "failed",
            "disposition": self.disposition,
        }


def _evaluate_fold(
    fold_name: str,
    test_year: int,
    role: FoldRole,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: AvailabilityClassifierConfig,
    *,
    n_bins: int,
) -> AvailabilityFoldEvaluation:
    if train["MatchDate"].max() >= test["MatchDate"].min():
        raise AvailabilityEvaluationError(
            f"Fold {fold_name} violates strict train-before-test chronology"
        )
    train_keys = list(train.loc[:, list(NATURAL_KEY)].itertuples(index=False, name=None))
    test_keys = list(test.loc[:, list(NATURAL_KEY)].itertuples(index=False, name=None))
    lr52_train = train.loc[:, list(FEATURE_TABLE_COLUMNS)]
    lr52_test = test.loc[:, list(FEATURE_TABLE_COLUMNS)]
    lr52 = _metrics(
        test["target_1x2"],
        predict_lr52_proba(fit_lr52(lr52_train), lr52_test),
        n_bins=n_bins,
    )
    fitted = _fit_candidate(train, config)
    candidate_probabilities = _predict_candidate(fitted, test)
    identical = candidate_probabilities.index.equals(test.index) and test_keys == list(
        test.loc[:, list(NATURAL_KEY)].itertuples(index=False, name=None)
    )
    if not identical or train_keys != list(
        train.loc[:, list(NATURAL_KEY)].itertuples(index=False, name=None)
    ):
        raise AvailabilityEvaluationError("candidate and LR52 fold keys differ")
    candidate = _metrics(test["target_1x2"], candidate_probabilities, n_bins=n_bins)
    return AvailabilityFoldEvaluation(
        fold_name=fold_name,
        test_year=test_year,
        role=role,
        train_start_date=_date_text(train["MatchDate"].min()),
        train_end_date=_date_text(train["MatchDate"].max()),
        test_start_date=_date_text(test["MatchDate"].min()),
        test_end_date=_date_text(test["MatchDate"].max()),
        train_rows=len(train),
        test_rows=len(test),
        preprocessing_fit_rows=len(train),
        classifier_iterations=fitted.iterations,
        identical_test_rows=True,
        same_date_isolation=True,
        lr52=lr52,
        candidate=candidate,
        candidate_improvement_log_loss=lr52.log_loss - candidate.log_loss,
    )


def _coverage_maps(
    frame: pd.DataFrame,
) -> tuple[
    dict[str, FeatureCoverageRecord],
    dict[str, FeatureCoverageRecord],
    dict[str, FeatureCoverageRecord],
]:
    def grouped(column: str) -> dict[str, FeatureCoverageRecord]:
        return {
            str(key): _feature_coverage(group)
            for key, group in frame.groupby(column, sort=True, observed=True)
        }

    by_fold = {
        str(year): _feature_coverage(frame.loc[frame["SeasonStartYear"].eq(year)])
        for year in ALL_TEST_YEARS
    }
    return grouped("Division"), grouped("SeasonStartYear"), by_fold


def evaluate_player_availability(
    feature_table: pd.DataFrame,
    fixtures: pd.DataFrame,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
    valuations: pd.DataFrame,
    *,
    classifier_config: AvailabilityClassifierConfig = AVAILABILITY_CLASSIFIER_CONFIG,
    test_years: tuple[int, ...] = ALL_TEST_YEARS,
    n_bins: int = 10,
) -> AvailabilityComparisonReport:
    """Run the frozen prior-only experiment, reserving 2025 until the end."""

    if tuple(test_years) != ALL_TEST_YEARS:
        raise ValueError("availability audit requires the complete 2022-2025 protocol")
    sources = prepare_availability_sources(fixtures, appearances, lineups, valuations)
    alignment = align_availability_scope(feature_table, sources)
    built = build_prior_availability_features(sources)
    augmented = alignment.feature_table.merge(
        built.features.loc[:, [*NATURAL_KEY, *PRIOR_AVAILABILITY_FEATURES]],
        on=list(NATURAL_KEY),
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    if not augmented["_merge"].eq("both").all():
        raise AvailabilityEvaluationError("availability merge omitted a feature row")
    augmented = augmented.drop(columns="_merge")
    augmented = augmented.sort_values(list(STABLE_ORDER_KEY), kind="mergesort").reset_index(
        drop=True
    )

    development_folds: list[AvailabilityFoldEvaluation] = []
    final_inputs: tuple[str, int, FoldRole, pd.DataFrame, pd.DataFrame] | None = None
    for fold in expanding_folds(augmented, test_years=test_years):
        train = augmented.loc[list(fold.train_idx)].copy(deep=True)
        test = augmented.loc[list(fold.test_idx)].copy(deep=True)
        inputs = (fold.name, fold.test_year, fold.role, train, test)
        if fold.test_year in DEVELOPMENT_TEST_YEARS:
            development_folds.append(_evaluate_fold(*inputs, classifier_config, n_bins=n_bins))
        elif fold.role is FoldRole.HISTORICAL_FINAL:
            if final_inputs is not None:
                raise AvailabilityEvaluationError("historical final was supplied twice")
            final_inputs = inputs
    if len(development_folds) != 3 or final_inputs is None:
        raise AvailabilityEvaluationError(
            "availability audit requires three development and one final fold"
        )

    development_tuple = tuple(development_folds)
    development = _aggregate("development", development_tuple)
    final_fold = _evaluate_fold(*final_inputs, classifier_config, n_bins=n_bins)
    historical_final = _aggregate("historical_final", (final_fold,))
    folds = (*development_tuple, final_fold)
    all_historical = _aggregate("all_historical_diagnostic", folds)
    gate = ResearchGate().evaluate(
        lr52_log_loss=development.lr52.log_loss,
        candidate_log_loss=development.candidate.log_loss,
        fold_improvements=tuple(fold.candidate_improvement_log_loss for fold in development_tuple),
        historical_final_improvement_log_loss=(historical_final.candidate_improvement_log_loss),
        passing_disposition="PLAYER_AVAILABILITY_CANDIDATE",
        failing_disposition="PLAYER_AVAILABILITY_REJECTED_FOR_NOW",
    )
    by_division, by_season, by_fold = _coverage_maps(augmented)
    return AvailabilityComparisonReport(
        schema_version="1.0",
        experiment_id=EXPERIMENT_ID,
        candidate_name=CANDIDATE_NAME,
        class_order=CLASS_ORDER,
        stable_feature_count=len(FEATURE_COLUMNS),
        availability_feature_count=len(PRIOR_AVAILABILITY_FEATURES),
        candidate_input_count=len(_candidate_columns()),
        classifier_configuration=classifier_config,
        timing_audit=tuple(item.to_dict() for item in TIMING_AUDIT),
        alignment=alignment.report,
        feature_coverage_by_division=by_division,
        feature_coverage_by_season=by_season,
        feature_coverage_by_fold=by_fold,
        folds=folds,
        development=development,
        historical_final=historical_final,
        all_historical_diagnostic=all_historical,
        success_gate=gate,
        final_evaluated_once=True,
        individual_player_rows_exported=False,
        disposition=gate.disposition,
    )
