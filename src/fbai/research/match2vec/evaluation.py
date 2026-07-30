"""Fold-local Match2Vec evaluation against LR52 on identical historical rows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from fbai.core.metrics import CLASS_ORDER, evaluate_predictions
from fbai.core.splits import (
    ALL_TEST_YEARS,
    DEVELOPMENT_TEST_YEARS,
    STABLE_ORDER_KEY,
    FoldRole,
    expanding_folds,
    inner_time_split,
)
from fbai.data.schema import NATURAL_KEY, validate_canonical_frame
from fbai.evaluation.report import MetricRecord
from fbai.features.checks import validate_feature_table
from fbai.features.schema import FEATURE_COLUMNS
from fbai.models.logistic import PROBABILITY_COLUMNS, fit_lr52, predict_lr52_proba
from fbai.models.preprocessing import select_lr52_features
from fbai.research.common import (
    CandidateComparisonReport,
    CandidateFoldEvaluation,
    ResearchGate,
    aggregate_candidate_folds,
)
from fbai.research.match2vec.config import Match2VecConfig
from fbai.research.match2vec.corpus import (
    build_match_sequence_corpus,
    build_train_vocabulary,
)
from fbai.research.match2vec.features import (
    REPRESENTATION_FEATURE_COLUMNS,
    build_representation_feature_table,
)
from fbai.research.match2vec.model import Match2VecSequenceModel

EXPERIMENT_ID = "match2vec_sequence_hybrid_historical_reproduction"
CANDIDATE_NAME = "Match2Vec M2V-SEQ + LR52"


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
    feature_keys = ordered_features.loc[:, list(NATURAL_KEY)]
    canonical_keys = ordered_canonical.loc[:, list(NATURAL_KEY)]
    try:
        pd.testing.assert_frame_equal(
            feature_keys,
            canonical_keys,
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        raise ValueError(
            "Feature and canonical tables must contain identical natural keys"
        ) from exc
    return ordered_features, ordered_canonical


def _numeric_inputs(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit source median imputation and scaling on inner-fit rows only."""

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    fit_values = scaler.fit_transform(imputer.fit_transform(select_lr52_features(fit)))
    validation_values = scaler.transform(imputer.transform(select_lr52_features(validation)))
    test_values = scaler.transform(imputer.transform(select_lr52_features(test)))
    arrays = (
        np.asarray(fit_values, dtype=np.float32),
        np.asarray(validation_values, dtype=np.float32),
        np.asarray(test_values, dtype=np.float32),
    )
    if any(values.shape[1] != len(FEATURE_COLUMNS) for values in arrays):
        raise ValueError("Match2Vec preprocessing must preserve all 52 LR52 features")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("Match2Vec preprocessing produced non-finite numeric inputs")
    return arrays


def evaluate_match2vec_candidate(
    feature_table: pd.DataFrame,
    canonical_matches: pd.DataFrame,
    *,
    config: Match2VecConfig | None = None,
    test_years: Sequence[int] = ALL_TEST_YEARS,
    n_bins: int = 10,
) -> CandidateComparisonReport:
    """Re-fit and compare the source-selected Match2Vec candidate in every fold."""

    if tuple(test_years) != ALL_TEST_YEARS:
        raise ValueError("The predefined Match2Vec gate requires the complete 2022-2025 protocol")
    original_features = feature_table.copy(deep=True)
    original_canonical = canonical_matches.copy(deep=True)
    ordered, canonical = _aligned_inputs(feature_table, canonical_matches)
    resolved = config or Match2VecConfig()
    corpus = build_match_sequence_corpus(
        ordered,
        canonical,
        sequence_length=resolved.sequence_length,
    )
    records: list[CandidateFoldEvaluation] = []

    for fold in expanding_folds(ordered, test_years=test_years):
        train = ordered.loc[list(fold.train_idx)].copy(deep=True)
        test = ordered.loc[list(fold.test_idx)].copy(deep=True)
        if train["MatchDate"].max() >= test["MatchDate"].min():
            raise ValueError(f"Fold {fold.name} violates strict train-before-test chronology")
        fit_indices, validation_indices = inner_time_split(
            train,
            validation_fraction=resolved.validation_fraction,
        )
        fit = train.loc[list(fit_indices)].copy(deep=True)
        validation = train.loc[list(validation_indices)].copy(deep=True)

        vocabulary = build_train_vocabulary(train)
        fit_numeric, validation_numeric, test_numeric = _numeric_inputs(
            fit,
            validation,
            test,
        )
        fit_batch = corpus.batch_for(fit)
        validation_batch = corpus.batch_for(validation)
        test_batch = corpus.batch_for(test)
        model = Match2VecSequenceModel(
            league_count=vocabulary.league_count,
            numeric_feature_count=len(FEATURE_COLUMNS),
            config=resolved,
        )
        model.fit(
            fit_batch=fit_batch,
            fit_league_ids=vocabulary.league_ids(fit),
            fit_numeric=fit_numeric,
            fit_labels=fit["target_1x2"].astype(str).tolist(),
            validation_batch=validation_batch,
            validation_league_ids=vocabulary.league_ids(validation),
            validation_numeric=validation_numeric,
            validation_labels=validation["target_1x2"].astype(str).tolist(),
        )
        candidate_probabilities = model.predict_proba(
            test_batch,
            vocabulary.league_ids(test),
            test_numeric,
        )
        representation_table = build_representation_feature_table(
            model,
            test_batch,
            vocabulary.league_ids(test),
        )
        if len(representation_table) != len(test):
            raise ValueError("Match2Vec representation and test rows are not aligned")

        lr52_probabilities = predict_lr52_proba(fit_lr52(train), test)
        labels = test["target_1x2"]
        lr52_metrics = _metrics(labels, lr52_probabilities, n_bins=n_bins)
        candidate_metrics = _metrics(labels, candidate_probabilities, n_bins=n_bins)
        oov_home, oov_away = vocabulary.oov_counts(test)
        records.append(
            CandidateFoldEvaluation(
                fold_name=fold.name,
                test_year=fold.test_year,
                role=fold.role,
                train_start_date=_date_text(train["MatchDate"].min()),
                train_end_date=_date_text(train["MatchDate"].max()),
                test_start_date=_date_text(test["MatchDate"].min()),
                test_end_date=_date_text(test["MatchDate"].max()),
                train_rows=len(train),
                representation_fit_rows=len(fit),
                representation_validation_rows=len(validation),
                test_rows=len(test),
                vocabulary_team_tokens=vocabulary.team_token_count,
                vocabulary_unknown_tokens=vocabulary.unknown_token_count,
                vocabulary_total_tokens=vocabulary.total_team_token_count,
                oov_home_occurrences=oov_home,
                oov_away_occurrences=oov_away,
                training_epochs=model.epochs_trained,
                identical_test_rows=len(candidate_probabilities) == len(lr52_probabilities),
                lr52=lr52_metrics,
                candidate=candidate_metrics,
                candidate_improvement_log_loss=(lr52_metrics.log_loss - candidate_metrics.log_loss),
            )
        )

    folds = tuple(records)
    development_folds = tuple(fold for fold in folds if fold.test_year in DEVELOPMENT_TEST_YEARS)
    final_folds = tuple(fold for fold in folds if fold.role is FoldRole.HISTORICAL_FINAL)
    if len(development_folds) != 3 or len(final_folds) != 1:
        raise ValueError("Match2Vec evaluation requires three development and one final fold")
    development = aggregate_candidate_folds("development", development_folds)
    historical_final = aggregate_candidate_folds("historical_final", final_folds)
    all_historical = aggregate_candidate_folds("all_historical_diagnostic", folds)
    gate = ResearchGate().evaluate(
        lr52_log_loss=development.lr52.log_loss,
        candidate_log_loss=development.candidate.log_loss,
        fold_improvements=tuple(fold.candidate_improvement_log_loss for fold in development_folds),
        historical_final_improvement_log_loss=(historical_final.candidate_improvement_log_loss),
    )
    report = CandidateComparisonReport(
        schema_version="1.0",
        experiment_id=EXPERIMENT_ID,
        candidate_name=CANDIDATE_NAME,
        candidate_configuration_id=resolved.identifier,
        class_order=CLASS_ORDER,
        representation_feature_count=len(REPRESENTATION_FEATURE_COLUMNS),
        representation_feature_names=REPRESENTATION_FEATURE_COLUMNS,
        folds=folds,
        development=development,
        historical_final=historical_final,
        all_historical_diagnostic=all_historical,
        success_gate=gate,
    )
    pd.testing.assert_frame_equal(feature_table, original_features)
    pd.testing.assert_frame_equal(canonical_matches, original_canonical)
    return report
