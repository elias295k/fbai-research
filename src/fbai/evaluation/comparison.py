"""Fair same-match chronological comparison of LR52 and the closing market."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from fbai.core.metrics import (
    CLASS_ORDER,
    evaluate_predictions,
    weighted_fold_summary,
)
from fbai.core.splits import ALL_TEST_YEARS, STABLE_ORDER_KEY, FoldRole, expanding_folds
from fbai.data.schema import CanonicalSchemaError, parse_match_dates
from fbai.evaluation.baselines import fit_training_prior, uniform_probabilities
from fbai.evaluation.market import (
    AUTHORITATIVE_SOURCE_ODDS_COLUMNS,
    CANONICAL_MARKET_COLUMNS,
    MARKET_NATURAL_KEY,
    MARKET_PROBABILITY_COLUMNS,
    closing_market_probabilities,
    prepare_closing_market,
)
from fbai.evaluation.report import MetricRecord
from fbai.features.checks import validate_feature_table
from fbai.features.schema import FEATURE_COLUMNS
from fbai.models.logistic import (
    PROBABILITY_COLUMNS,
    LR52Config,
    fit_lr52,
    predict_lr52_proba,
)

MARKET_BENCHMARK_NAME = "closing market benchmark"
MARKET_TIMING = "closing_near_kickoff"
COMPARISON_POLICY = (
    "LR52 fits on the complete valid training history; LR52 and market metrics "
    "are then evaluated on identical market-covered test keys."
)


class MarketAlignmentError(ValueError):
    """Raised when market-to-match alignment cannot satisfy its key contract."""


@dataclass(frozen=True, slots=True)
class CoverageSlice:
    """Coverage for one named fold, division, or test-year slice."""

    name: str
    candidate_match_rows: int
    aligned_rows: int
    coverage_percentage: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "name": self.name,
            "candidate_match_rows": self.candidate_match_rows,
            "aligned_rows": self.aligned_rows,
            "coverage_percentage": self.coverage_percentage,
        }


@dataclass(frozen=True, slots=True)
class MarketCoverage:
    """Aggregate, division, and test-year market alignment diagnostics."""

    candidate_match_rows: int
    supplied_market_rows: int
    valid_market_rows: int
    duplicate_market_keys: int
    unmatched_match_keys: int
    unmatched_market_keys: int
    incomplete_odds_rows: int
    invalid_odds_rows: int
    aligned_rows: int
    coverage_percentage: float
    by_division: tuple[CoverageSlice, ...]
    by_test_year: tuple[CoverageSlice, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_match_rows": self.candidate_match_rows,
            "supplied_market_rows": self.supplied_market_rows,
            "valid_market_rows": self.valid_market_rows,
            "duplicate_market_keys": self.duplicate_market_keys,
            "unmatched_match_keys": self.unmatched_match_keys,
            "unmatched_market_keys": self.unmatched_market_keys,
            "incomplete_odds_rows": self.incomplete_odds_rows,
            "invalid_odds_rows": self.invalid_odds_rows,
            "aligned_rows": self.aligned_rows,
            "coverage_percentage": self.coverage_percentage,
            "by_division": [item.to_dict() for item in self.by_division],
            "by_test_year": [item.to_dict() for item in self.by_test_year],
        }


@dataclass(frozen=True, slots=True)
class AlignedFoldComparison:
    """Aggregate-only same-match results for one chronological fold."""

    fold_name: str
    test_year: int
    role: FoldRole
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    train_rows: int
    candidate_test_rows: int
    aligned_rows: int
    unaligned_rows: int
    coverage_percentage: float
    preprocessing_fit_rows: int
    model_iterations: tuple[int, ...]
    full_test_lr52: MetricRecord
    aligned_lr52: MetricRecord
    market: MetricRecord
    uniform: MetricRecord
    training_prior: MetricRecord
    market_advantage_log_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_name": self.fold_name,
            "test_year": self.test_year,
            "role": self.role.value,
            "train_date_range": [self.train_start_date, self.train_end_date],
            "test_date_range": [self.test_start_date, self.test_end_date],
            "train_rows": self.train_rows,
            "candidate_test_rows": self.candidate_test_rows,
            "aligned_rows": self.aligned_rows,
            "unaligned_rows": self.unaligned_rows,
            "coverage_percentage": self.coverage_percentage,
            "preprocessing_fit_rows": self.preprocessing_fit_rows,
            "model_iterations": list(self.model_iterations),
            "full_test_lr52": self.full_test_lr52.to_dict(),
            "aligned_lr52": self.aligned_lr52.to_dict(),
            "market": self.market.to_dict(),
            "uniform": self.uniform.to_dict(),
            "training_prior": self.training_prior.to_dict(),
            "market_advantage_log_loss": self.market_advantage_log_loss,
        }


@dataclass(frozen=True, slots=True)
class MarketBenchmarkSummary:
    """Sample-count-weighted aligned metrics for an explicitly named view."""

    name: str
    fold_count: int
    candidate_test_rows: int
    aligned_rows: int
    coverage_percentage: float
    full_test_lr52: MetricRecord
    aligned_lr52: MetricRecord
    market: MetricRecord
    uniform: MetricRecord
    training_prior: MetricRecord
    market_advantage_log_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fold_count": self.fold_count,
            "candidate_test_rows": self.candidate_test_rows,
            "aligned_rows": self.aligned_rows,
            "coverage_percentage": self.coverage_percentage,
            "full_test_lr52": self.full_test_lr52.to_dict(),
            "aligned_lr52": self.aligned_lr52.to_dict(),
            "market": self.market.to_dict(),
            "uniform": self.uniform.to_dict(),
            "training_prior": self.training_prior.to_dict(),
            "market_advantage_log_loss": self.market_advantage_log_loss,
        }


@dataclass(frozen=True, slots=True)
class MarketComparisonReport:
    """Immutable public report for LR52 versus the external closing market."""

    schema_version: str
    benchmark_name: str
    market_timing: str
    class_order: tuple[str, str, str]
    source_odds_columns: tuple[str, str, str]
    canonical_market_columns: tuple[str, ...]
    probability_transformation: str
    comparison_policy: str
    feature_count: int
    market_coverage: MarketCoverage
    folds: tuple[AlignedFoldComparison, ...]
    development: MarketBenchmarkSummary
    historical_final: MarketBenchmarkSummary
    all_historical_diagnostic: MarketBenchmarkSummary

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe aggregate and fold-level metadata only."""

        return {
            "schema_version": self.schema_version,
            "benchmark_name": self.benchmark_name,
            "market_timing": self.market_timing,
            "class_order": list(self.class_order),
            "source_odds_columns": list(self.source_odds_columns),
            "canonical_market_columns": list(self.canonical_market_columns),
            "probability_transformation": self.probability_transformation,
            "comparison_policy": self.comparison_policy,
            "feature_count": self.feature_count,
            "market_coverage": self.market_coverage.to_dict(),
            "folds": [fold.to_dict() for fold in self.folds],
            "development": self.development.to_dict(),
            "historical_final": self.historical_final.to_dict(),
            "all_historical_diagnostic": self.all_historical_diagnostic.to_dict(),
        }


def _date_text(value: object) -> str:
    return str(pd.Timestamp(value).date().isoformat())


def _percentage(aligned: int, candidate: int) -> float:
    return 0.0 if candidate == 0 else 100.0 * aligned / candidate


def _validated_match_keys(matches: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in MARKET_NATURAL_KEY if column not in matches.columns]
    if missing:
        raise MarketAlignmentError(f"Missing match key fields: {', '.join(missing)}")
    if matches.empty:
        raise MarketAlignmentError("Candidate match table must contain at least one row")

    columns = list(MARKET_NATURAL_KEY)
    if "SeasonStartYear" in matches.columns:
        columns.append("SeasonStartYear")
    keys = matches.loc[:, columns].copy(deep=True)
    try:
        keys["MatchDate"] = parse_match_dates(keys["MatchDate"])
    except CanonicalSchemaError as exc:
        raise MarketAlignmentError(str(exc)) from exc
    for column in ("Division", "HomeTeam", "AwayTeam"):
        values = keys[column].astype("string")
        if bool((values.isna() | values.str.strip().eq("")).any()):
            raise MarketAlignmentError(f"{column} must contain non-empty strings")
        keys[column] = values.str.strip()
    duplicate_rows = int(keys.duplicated(list(MARKET_NATURAL_KEY), keep=False).sum())
    if duplicate_rows:
        raise MarketAlignmentError(
            f"Match natural key is not unique; {duplicate_rows} rows are duplicated"
        )
    if "SeasonStartYear" in keys.columns:
        try:
            keys["SeasonStartYear"] = pd.to_numeric(keys["SeasonStartYear"], errors="raise").astype(
                int
            )
        except (TypeError, ValueError) as exc:
            raise MarketAlignmentError("SeasonStartYear must be numeric") from exc
    return keys.sort_values(list(MARKET_NATURAL_KEY), kind="mergesort").reset_index(drop=True)


def _coverage_slices(
    candidates: pd.DataFrame,
    aligned: pd.DataFrame,
    *,
    column: str,
) -> tuple[CoverageSlice, ...]:
    candidate_counts = candidates.groupby(column, sort=True, observed=True).size()
    aligned_counts = aligned.groupby(column, sort=True, observed=True).size()
    return tuple(
        CoverageSlice(
            name=str(name),
            candidate_match_rows=int(candidate),
            aligned_rows=int(aligned_counts.get(name, 0)),
            coverage_percentage=_percentage(int(aligned_counts.get(name, 0)), int(candidate)),
        )
        for name, candidate in candidate_counts.items()
    )


def _prepare_alignment(
    matches: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, MarketCoverage]:
    candidates = _validated_match_keys(matches)
    valid_market, audit = prepare_closing_market(market)
    if valid_market.empty:
        raise MarketAlignmentError(
            "Closing-market table contains no valid rows after explicit validation"
        )
    probabilities = closing_market_probabilities(valid_market)

    match_keys = candidates.loc[:, list(MARKET_NATURAL_KEY)]
    market_keys = probabilities.loc[:, list(MARKET_NATURAL_KEY)]
    outer = match_keys.merge(
        market_keys,
        on=list(MARKET_NATURAL_KEY),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    aligned = candidates.merge(
        probabilities,
        on=list(MARKET_NATURAL_KEY),
        how="inner",
        validate="one_to_one",
    )
    aligned = aligned.sort_values(list(MARKET_NATURAL_KEY), kind="mergesort").reset_index(drop=True)
    by_division = _coverage_slices(candidates, aligned, column="Division")
    by_test_year: tuple[CoverageSlice, ...] = ()
    if "SeasonStartYear" in candidates.columns:
        by_test_year = _coverage_slices(
            candidates,
            aligned,
            column="SeasonStartYear",
        )
    coverage = MarketCoverage(
        candidate_match_rows=len(candidates),
        supplied_market_rows=audit.supplied_market_rows,
        valid_market_rows=audit.valid_market_rows,
        duplicate_market_keys=audit.duplicate_market_keys,
        unmatched_match_keys=int(outer["_merge"].eq("left_only").sum()),
        unmatched_market_keys=int(outer["_merge"].eq("right_only").sum()),
        incomplete_odds_rows=audit.incomplete_odds_rows,
        invalid_odds_rows=audit.invalid_odds_rows,
        aligned_rows=len(aligned),
        coverage_percentage=_percentage(len(aligned), len(candidates)),
        by_division=by_division,
        by_test_year=by_test_year,
    )
    return aligned, coverage


def align_closing_market(
    matches: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, MarketCoverage]:
    """Align valid market probabilities to matches by exact natural key.

    The returned frame contains keys, optional ``SeasonStartYear``, named
    market probabilities, and overround. Neither input is mutated.
    """

    return _prepare_alignment(matches, market)


def _metric_record(
    labels: pd.Series,
    probabilities: pd.DataFrame,
    *,
    probability_columns: Sequence[str],
    n_bins: int,
) -> MetricRecord:
    values = probabilities.loc[:, list(probability_columns)].to_numpy(dtype="float64")
    return MetricRecord.from_mapping(
        evaluate_predictions(labels.astype(str).tolist(), values, n_bins=n_bins)
    )


def _aggregate(
    name: str,
    folds: tuple[AlignedFoldComparison, ...],
) -> MarketBenchmarkSummary:
    if not folds:
        raise ValueError(f"Aggregate {name} requires at least one fold")

    def summarize(attribute: str) -> MetricRecord:
        records = [getattr(fold, attribute).to_dict() for fold in folds]
        return MetricRecord.from_mapping(weighted_fold_summary(records))

    candidate_rows = sum(fold.candidate_test_rows for fold in folds)
    aligned_rows = sum(fold.aligned_rows for fold in folds)
    aligned_lr52 = summarize("aligned_lr52")
    market = summarize("market")
    return MarketBenchmarkSummary(
        name=name,
        fold_count=len(folds),
        candidate_test_rows=candidate_rows,
        aligned_rows=aligned_rows,
        coverage_percentage=_percentage(aligned_rows, candidate_rows),
        full_test_lr52=summarize("full_test_lr52"),
        aligned_lr52=aligned_lr52,
        market=market,
        uniform=summarize("uniform"),
        training_prior=summarize("training_prior"),
        market_advantage_log_loss=aligned_lr52.log_loss - market.log_loss,
    )


def _build_report(
    folds: tuple[AlignedFoldComparison, ...],
    coverage: MarketCoverage,
) -> MarketComparisonReport:
    development_folds = tuple(fold for fold in folds if fold.role is FoldRole.DEVELOPMENT)
    final_folds = tuple(fold for fold in folds if fold.role is FoldRole.HISTORICAL_FINAL)
    if len(final_folds) != 1:
        raise ValueError("Market comparison requires exactly one historically frozen final fold")
    return MarketComparisonReport(
        schema_version="1.0",
        benchmark_name=MARKET_BENCHMARK_NAME,
        market_timing=MARKET_TIMING,
        class_order=CLASS_ORDER,
        source_odds_columns=AUTHORITATIVE_SOURCE_ODDS_COLUMNS,
        canonical_market_columns=CANONICAL_MARKET_COLUMNS,
        probability_transformation="reciprocal_odds_then_divide_by_row_implied_sum",
        comparison_policy=COMPARISON_POLICY,
        feature_count=len(FEATURE_COLUMNS),
        market_coverage=coverage,
        folds=folds,
        development=_aggregate("development", development_folds),
        historical_final=_aggregate("historical_final", final_folds),
        all_historical_diagnostic=_aggregate("all_historical_diagnostic", folds),
    )


def evaluate_lr52_vs_closing_market(
    feature_table: pd.DataFrame,
    market: pd.DataFrame,
    *,
    config: LR52Config | None = None,
    test_years: Sequence[int] = ALL_TEST_YEARS,
    n_bins: int = 10,
) -> MarketComparisonReport:
    """Evaluate LR52 and closing-market probabilities on identical test keys."""

    ordered = (
        feature_table.sort_values(list(STABLE_ORDER_KEY), kind="mergesort")
        .reset_index(drop=True)
        .copy(deep=True)
    )
    validate_feature_table(ordered, expected_row_count=len(feature_table))
    aligned_market, coverage = _prepare_alignment(ordered, market)
    market_probabilities = aligned_market.loc[:, [*MARKET_NATURAL_KEY, *MARKET_PROBABILITY_COLUMNS]]
    resolved = config or LR52Config()
    fold_records: list[AlignedFoldComparison] = []

    for fold in expanding_folds(ordered, test_years=test_years):
        train = ordered.loc[list(fold.train_idx)].copy(deep=True)
        test = ordered.loc[list(fold.test_idx)].copy(deep=True)
        if train["MatchDate"].max() >= test["MatchDate"].min():
            raise ValueError(f"Fold {fold.name} violates strict train-before-test chronology")

        fitted = fit_lr52(train, config=resolved)
        lr52 = predict_lr52_proba(fitted, test)
        uniform = uniform_probabilities(test)
        prior = fit_training_prior(train).predict(test)
        full_lr52 = _metric_record(
            test["target_1x2"],
            lr52,
            probability_columns=PROBABILITY_COLUMNS,
            n_bins=n_bins,
        )

        evaluation = test.loc[:, [*MARKET_NATURAL_KEY, "target_1x2"]].copy(deep=True)
        for column in PROBABILITY_COLUMNS:
            evaluation[f"lr52_{column}"] = lr52[column]
            evaluation[f"uniform_{column}"] = uniform[column]
            evaluation[f"prior_{column}"] = prior[column]
        evaluation = evaluation.merge(
            market_probabilities,
            on=list(MARKET_NATURAL_KEY),
            how="inner",
            validate="one_to_one",
        ).sort_values(list(MARKET_NATURAL_KEY), kind="mergesort")
        if evaluation.empty:
            raise MarketAlignmentError(f"Fold {fold.name} has no market-covered test rows")

        labels = evaluation["target_1x2"]
        lr52_columns = tuple(f"lr52_{column}" for column in PROBABILITY_COLUMNS)
        uniform_columns = tuple(f"uniform_{column}" for column in PROBABILITY_COLUMNS)
        prior_columns = tuple(f"prior_{column}" for column in PROBABILITY_COLUMNS)
        aligned_lr52 = _metric_record(
            labels,
            evaluation,
            probability_columns=lr52_columns,
            n_bins=n_bins,
        )
        market_metrics = _metric_record(
            labels,
            evaluation,
            probability_columns=MARKET_PROBABILITY_COLUMNS,
            n_bins=n_bins,
        )
        aligned_rows = len(evaluation)
        fold_records.append(
            AlignedFoldComparison(
                fold_name=fold.name,
                test_year=fold.test_year,
                role=fold.role,
                train_start_date=_date_text(train["MatchDate"].min()),
                train_end_date=_date_text(train["MatchDate"].max()),
                test_start_date=_date_text(test["MatchDate"].min()),
                test_end_date=_date_text(test["MatchDate"].max()),
                train_rows=len(train),
                candidate_test_rows=len(test),
                aligned_rows=aligned_rows,
                unaligned_rows=len(test) - aligned_rows,
                coverage_percentage=_percentage(aligned_rows, len(test)),
                preprocessing_fit_rows=len(train),
                model_iterations=fitted.iterations,
                full_test_lr52=full_lr52,
                aligned_lr52=aligned_lr52,
                market=market_metrics,
                uniform=_metric_record(
                    labels,
                    evaluation,
                    probability_columns=uniform_columns,
                    n_bins=n_bins,
                ),
                training_prior=_metric_record(
                    labels,
                    evaluation,
                    probability_columns=prior_columns,
                    n_bins=n_bins,
                ),
                market_advantage_log_loss=aligned_lr52.log_loss - market_metrics.log_loss,
            )
        )

    return _build_report(tuple(fold_records), coverage)
