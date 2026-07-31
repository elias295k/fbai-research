"""Chronological internal graph-model audit against identical-row LR52."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from fbai.core.leakage import NATURAL_KEY
from fbai.core.metrics import CLASS_ORDER, evaluate_predictions, weighted_fold_summary
from fbai.core.splits import (
    ALL_TEST_YEARS,
    DEVELOPMENT_TEST_YEARS,
    STABLE_ORDER_KEY,
    FoldRole,
    expanding_folds,
)
from fbai.evaluation.report import MetricRecord
from fbai.features.checks import validate_feature_table
from fbai.features.schema import FEATURE_COLUMNS
from fbai.models.logistic import PROBABILITY_COLUMNS, fit_lr52, predict_lr52_proba
from fbai.research.common import ResearchGate, ResearchGateResult
from fbai.research.graph_model.config import (
    AUTHORITATIVE_CONTEXTS,
    AUTHORITATIVE_GRAPH_CONFIGS,
    GRAPH_CLASSIFIER_CONFIG,
    LR52_GRAPH_CONTEXT,
    GraphClassifierConfig,
    GraphEmbeddingConfig,
    graph_feature_columns,
)
from fbai.research.graph_model.graph import (
    GraphFitMetadata,
    GraphSourceFrames,
    build_fold_graph_features_from_source,
    prepare_graph_source_frames,
)
from fbai.research.graph_model.model import (
    fit_graph_classifier,
    predict_graph_classifier_proba,
)

EXPERIMENT_ID = "temporal_graph_embedding_internal_historical_reproduction"
CANDIDATE_FAMILY = "team_player_bipartite_svd_logistic"


class GraphEvaluationError(ValueError):
    """Raised when the frozen graph evaluation protocol is violated."""


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


def _ordered_feature_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(feature_table, pd.DataFrame):
        raise GraphEvaluationError("graph-model input must be a pandas DataFrame")
    ordered = (
        feature_table.sort_values(list(STABLE_ORDER_KEY), kind="mergesort")
        .reset_index(drop=True)
        .copy(deep=True)
    )
    validate_feature_table(ordered, expected_row_count=len(feature_table))
    return ordered


def _merge_graph_features(
    feature_table: pd.DataFrame,
    graph_features: pd.DataFrame,
    graph_columns: tuple[str, ...],
) -> pd.DataFrame:
    selected = graph_features.loc[:, [*NATURAL_KEY, "game_id", *graph_columns]]
    merged = feature_table.merge(
        selected,
        on=list(NATURAL_KEY),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != len(feature_table) or merged["game_id"].isna().any():
        raise GraphEvaluationError("every evaluation match must have exactly one graph fixture")
    merged = merged.sort_values(list(STABLE_ORDER_KEY), kind="mergesort").reset_index(drop=True)
    merged_keys = list(merged.loc[:, list(NATURAL_KEY)].itertuples(index=False, name=None))
    expected_keys = list(feature_table.loc[:, list(NATURAL_KEY)].itertuples(index=False, name=None))
    if merged_keys != expected_keys:
        raise GraphEvaluationError("graph merge changed evaluation natural-key order")
    return merged


@dataclass(frozen=True, slots=True)
class GraphFoldEvaluation:
    """One aggregate-only classifier/configuration/fold comparison."""

    configuration_name: str
    configuration_identifier: str
    context: str
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
    graph_metadata: GraphFitMetadata
    graph_feature_count: int
    classifier_feature_count: int
    classifier_parameter_count: int
    classifier_iterations: tuple[int, ...]
    same_date_isolation: bool
    identical_test_rows: bool
    lr52: MetricRecord
    candidate: MetricRecord
    candidate_improvement_log_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_name": self.configuration_name,
            "configuration_identifier": self.configuration_identifier,
            "context": self.context,
            "fold_name": self.fold_name,
            "test_year": self.test_year,
            "role": self.role.value,
            "train_date_range": [self.train_start_date, self.train_end_date],
            "test_date_range": [self.test_start_date, self.test_end_date],
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "preprocessing_fit_rows": self.preprocessing_fit_rows,
            "graph": self.graph_metadata.to_dict(),
            "graph_feature_count": self.graph_feature_count,
            "classifier_feature_count": self.classifier_feature_count,
            "classifier_parameter_count": self.classifier_parameter_count,
            "classifier_iterations": list(self.classifier_iterations),
            "same_date_isolation": self.same_date_isolation,
            "identical_test_rows": self.identical_test_rows,
            "lr52": self.lr52.to_dict(),
            "candidate": self.candidate.to_dict(),
            "candidate_improvement_log_loss": self.candidate_improvement_log_loss,
        }


@dataclass(frozen=True, slots=True)
class GraphAggregate:
    """Sample-count-weighted LR52 and graph-candidate metrics."""

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


def aggregate_graph_folds(
    name: str,
    folds: tuple[GraphFoldEvaluation, ...],
) -> GraphAggregate:
    """Aggregate graph folds using their identical test sample counts."""

    if not folds:
        raise ValueError(f"Aggregate {name} requires at least one fold")
    lr52 = MetricRecord.from_mapping(weighted_fold_summary([fold.lr52.to_dict() for fold in folds]))
    candidate = MetricRecord.from_mapping(
        weighted_fold_summary([fold.candidate.to_dict() for fold in folds])
    )
    return GraphAggregate(
        name=name,
        fold_count=len(folds),
        lr52=lr52,
        candidate=candidate,
        candidate_improvement_log_loss=lr52.log_loss - candidate.log_loss,
    )


@dataclass(frozen=True, slots=True)
class GraphCandidateEvaluation:
    """Development-only evidence for one representation/classifier context."""

    configuration: GraphEmbeddingConfig
    context: str
    development_folds: tuple[GraphFoldEvaluation, ...]
    development: GraphAggregate

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration.to_dict(),
            "context": self.context,
            "development_folds": [fold.to_dict() for fold in self.development_folds],
            "development": self.development.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GraphComparisonReport:
    """Immutable development selection, final result, and gate record."""

    schema_version: str
    experiment_id: str
    candidate_family: str
    class_order: tuple[str, str, str]
    stable_feature_count: int
    classifier_configuration: GraphClassifierConfig
    configurations: tuple[GraphEmbeddingConfig, ...]
    candidates: tuple[GraphCandidateEvaluation, ...]
    selected_configuration: str
    selection_context: str
    selection_metric: str
    selection_used_development_only: bool
    selected_final_contexts: tuple[GraphFoldEvaluation, ...]
    historical_final: GraphAggregate
    all_historical_diagnostic: GraphAggregate
    success_gate: ResearchGateResult
    final_confirmed: bool
    disposition: str

    def __post_init__(self) -> None:
        if self.class_order != CLASS_ORDER:
            raise ValueError(f"class_order must be exactly {CLASS_ORDER}")
        if self.stable_feature_count != len(FEATURE_COLUMNS):
            raise ValueError("graph audit must preserve the exact stable 52 inputs")
        if self.selection_context != LR52_GRAPH_CONTEXT:
            raise ValueError("graph method selection must use LR52 + graph development")
        if not self.selection_used_development_only:
            raise ValueError("graph selection must use development folds only")
        names = tuple(config.name for config in self.configurations)
        if self.selected_configuration not in names:
            raise ValueError("selected graph configuration is not authoritative")
        final_contexts = {fold.context for fold in self.selected_final_contexts}
        if final_contexts != set(AUTHORITATIVE_CONTEXTS):
            raise ValueError("selected graph method final requires both internal contexts")

    @property
    def selected_candidate(self) -> GraphCandidateEvaluation:
        return next(
            candidate
            for candidate in self.candidates
            if candidate.configuration.name == self.selected_configuration
            and candidate.context == self.selection_context
        )

    @property
    def selected_final_fold(self) -> GraphFoldEvaluation:
        return next(
            fold for fold in self.selected_final_contexts if fold.context == self.selection_context
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "candidate_family": self.candidate_family,
            "class_order": list(self.class_order),
            "stable_feature_count": self.stable_feature_count,
            "classifier_configuration": self.classifier_configuration.to_dict(),
            "configurations": [config.to_dict() for config in self.configurations],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selection": {
                "configuration": self.selected_configuration,
                "context": self.selection_context,
                "metric": self.selection_metric,
                "development_only": self.selection_used_development_only,
            },
            "selected_final_contexts": [fold.to_dict() for fold in self.selected_final_contexts],
            "historical_final": self.historical_final.to_dict(),
            "all_historical_diagnostic": self.all_historical_diagnostic.to_dict(),
            "success_gate": self.success_gate.to_dict(),
            "final_confirmed": self.final_confirmed,
            "verdict": "passed" if self.success_gate.passed else "failed",
            "disposition": self.disposition,
        }


@dataclass(frozen=True, slots=True)
class _FoldInputs:
    fold_name: str
    test_year: int
    role: FoldRole
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train: pd.DataFrame
    test: pd.DataFrame
    lr52: MetricRecord


def _evaluate_context_fold(
    inputs: _FoldInputs,
    augmented: pd.DataFrame,
    graph_metadata: GraphFitMetadata,
    config: GraphEmbeddingConfig,
    context: str,
    classifier_config: GraphClassifierConfig,
    *,
    n_bins: int,
) -> GraphFoldEvaluation:
    train = augmented.loc[list(inputs.train_indices)].copy(deep=True)
    test = augmented.loc[list(inputs.test_indices)].copy(deep=True)
    train_keys = list(train.loc[:, list(NATURAL_KEY)].itertuples(index=False, name=None))
    expected_train_keys = list(
        inputs.train.loc[:, list(NATURAL_KEY)].itertuples(index=False, name=None)
    )
    test_keys = list(test.loc[:, list(NATURAL_KEY)].itertuples(index=False, name=None))
    expected_test_keys = list(
        inputs.test.loc[:, list(NATURAL_KEY)].itertuples(index=False, name=None)
    )
    if train_keys != expected_train_keys or test_keys != expected_test_keys:
        raise GraphEvaluationError("candidate and LR52 fold keys differ")
    graph_columns = graph_feature_columns(config)
    fitted = fit_graph_classifier(
        train,
        graph_columns,
        context=context,
        config=classifier_config,
    )
    probabilities = predict_graph_classifier_proba(fitted, test)
    identical_test_rows = probabilities.index.equals(test.index)
    if not identical_test_rows:
        raise GraphEvaluationError("candidate probabilities changed test row identity")
    candidate = _metrics(test["target_1x2"], probabilities, n_bins=n_bins)
    return GraphFoldEvaluation(
        configuration_name=config.name,
        configuration_identifier=config.identifier,
        context=context,
        fold_name=inputs.fold_name,
        test_year=inputs.test_year,
        role=inputs.role,
        train_start_date=_date_text(train["MatchDate"].min()),
        train_end_date=_date_text(train["MatchDate"].max()),
        test_start_date=_date_text(test["MatchDate"].min()),
        test_end_date=_date_text(test["MatchDate"].max()),
        train_rows=len(train),
        test_rows=len(test),
        preprocessing_fit_rows=len(train),
        graph_metadata=graph_metadata,
        graph_feature_count=len(graph_columns),
        classifier_feature_count=len(fitted.feature_columns),
        classifier_parameter_count=fitted.parameter_count,
        classifier_iterations=fitted.iterations,
        same_date_isolation=graph_metadata.same_date_batches,
        identical_test_rows=True,
        lr52=inputs.lr52,
        candidate=candidate,
        candidate_improvement_log_loss=inputs.lr52.log_loss - candidate.log_loss,
    )


def evaluate_graph_model(
    feature_table: pd.DataFrame,
    fixtures: pd.DataFrame,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
    *,
    configurations: tuple[GraphEmbeddingConfig, ...] = AUTHORITATIVE_GRAPH_CONFIGS,
    classifier_config: GraphClassifierConfig = GRAPH_CLASSIFIER_CONFIG,
    test_years: tuple[int, ...] = ALL_TEST_YEARS,
    n_bins: int = 10,
) -> GraphComparisonReport:
    """Reproduce the internal source graph audit with final-fold isolation."""

    if tuple(test_years) != ALL_TEST_YEARS:
        raise ValueError("graph audit requires the complete 2022-2025 protocol")
    if not configurations:
        raise ValueError("graph audit requires at least one configuration")
    names = tuple(config.name for config in configurations)
    if len(names) != len(set(names)):
        raise ValueError("graph configuration names must be unique")
    ordered = _ordered_feature_table(feature_table)
    source: GraphSourceFrames = prepare_graph_source_frames(
        fixtures,
        appearances,
        lineups,
    )

    fold_inputs: list[_FoldInputs] = []
    for fold in expanding_folds(ordered, test_years=test_years):
        train_indices = tuple(cast(int, index) for index in fold.train_idx)
        test_indices = tuple(cast(int, index) for index in fold.test_idx)
        train = ordered.loc[list(train_indices)].copy(deep=True)
        test = ordered.loc[list(test_indices)].copy(deep=True)
        if train["MatchDate"].max() >= test["MatchDate"].min():
            raise GraphEvaluationError(
                f"Fold {fold.name} violates strict train-before-test chronology"
            )
        lr52_probabilities = predict_lr52_proba(fit_lr52(train), test)
        fold_inputs.append(
            _FoldInputs(
                fold_name=fold.name,
                test_year=fold.test_year,
                role=fold.role,
                train_indices=train_indices,
                test_indices=test_indices,
                train=train,
                test=test,
                lr52=_metrics(
                    test["target_1x2"],
                    lr52_probabilities,
                    n_bins=n_bins,
                ),
            )
        )

    development_inputs = tuple(
        inputs for inputs in fold_inputs if inputs.test_year in DEVELOPMENT_TEST_YEARS
    )
    final_inputs = tuple(
        inputs for inputs in fold_inputs if inputs.role is FoldRole.HISTORICAL_FINAL
    )
    if len(development_inputs) != 3 or len(final_inputs) != 1:
        raise GraphEvaluationError(
            "graph audit requires three development folds and one final fold"
        )

    candidates: list[GraphCandidateEvaluation] = []
    for config in configurations:
        context_folds: dict[str, list[GraphFoldEvaluation]] = {
            context: [] for context in AUTHORITATIVE_CONTEXTS
        }
        for inputs in development_inputs:
            built = build_fold_graph_features_from_source(
                source,
                test_year=inputs.test_year,
                config=config,
            )
            graph_columns = graph_feature_columns(config)
            augmented = _merge_graph_features(
                ordered,
                built.features,
                graph_columns,
            )
            for context in AUTHORITATIVE_CONTEXTS:
                context_folds[context].append(
                    _evaluate_context_fold(
                        inputs,
                        augmented,
                        built.metadata,
                        config,
                        context,
                        classifier_config,
                        n_bins=n_bins,
                    )
                )
        for context in AUTHORITATIVE_CONTEXTS:
            folds = tuple(context_folds[context])
            candidates.append(
                GraphCandidateEvaluation(
                    configuration=config,
                    context=context,
                    development_folds=folds,
                    development=aggregate_graph_folds("development", folds),
                )
            )

    selection_candidates = tuple(
        candidate for candidate in candidates if candidate.context == LR52_GRAPH_CONTEXT
    )
    selected = min(
        selection_candidates,
        key=lambda candidate: (
            candidate.development.candidate.log_loss,
            candidate.configuration.name,
        ),
    )
    final_input = final_inputs[0]
    final_build = build_fold_graph_features_from_source(
        source,
        test_year=final_input.test_year,
        config=selected.configuration,
    )
    final_augmented = _merge_graph_features(
        ordered,
        final_build.features,
        graph_feature_columns(selected.configuration),
    )
    final_contexts = tuple(
        _evaluate_context_fold(
            final_input,
            final_augmented,
            final_build.metadata,
            selected.configuration,
            context,
            classifier_config,
            n_bins=n_bins,
        )
        for context in AUTHORITATIVE_CONTEXTS
    )
    selected_final = next(fold for fold in final_contexts if fold.context == LR52_GRAPH_CONTEXT)
    historical_final = aggregate_graph_folds("historical_final", (selected_final,))
    all_historical = aggregate_graph_folds(
        "all_historical_diagnostic",
        (*selected.development_folds, selected_final),
    )
    gate = ResearchGate().evaluate(
        lr52_log_loss=selected.development.lr52.log_loss,
        candidate_log_loss=selected.development.candidate.log_loss,
        fold_improvements=tuple(
            fold.candidate_improvement_log_loss for fold in selected.development_folds
        ),
        historical_final_improvement_log_loss=(historical_final.candidate_improvement_log_loss),
        passing_disposition="GRAPH_EMBEDDING_REJECTED_FOR_NOW",
        failing_disposition="DATA_CEILING_UPHELD",
    )
    final_confirmed = historical_final.candidate_improvement_log_loss > 0.0
    disposition = (
        "GRAPH_EMBEDDING_REJECTED_FOR_NOW"
        if gate.passed and final_confirmed
        else "DATA_CEILING_UPHELD"
    )
    return GraphComparisonReport(
        schema_version="1.0",
        experiment_id=EXPERIMENT_ID,
        candidate_family=CANDIDATE_FAMILY,
        class_order=CLASS_ORDER,
        stable_feature_count=len(FEATURE_COLUMNS),
        classifier_configuration=classifier_config,
        configurations=configurations,
        candidates=tuple(candidates),
        selected_configuration=selected.configuration.name,
        selection_context=LR52_GRAPH_CONTEXT,
        selection_metric="development_sample_count_weighted_log_loss",
        selection_used_development_only=True,
        selected_final_contexts=final_contexts,
        historical_final=historical_final,
        all_historical_diagnostic=all_historical,
        success_gate=gate,
        final_confirmed=final_confirmed,
        disposition=disposition,
    )
