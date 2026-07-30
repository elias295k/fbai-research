"""Chronological LR52 evaluation and external closing-market comparison."""

from fbai.evaluation.baselines import (
    BaselineTargetError,
    TrainingPriorBaseline,
    fit_training_prior,
    uniform_probabilities,
)
from fbai.evaluation.comparison import (
    AlignedFoldComparison,
    CoverageSlice,
    MarketAlignmentError,
    MarketBenchmarkSummary,
    MarketComparisonReport,
    MarketCoverage,
    align_closing_market,
    evaluate_lr52_vs_closing_market,
)
from fbai.evaluation.market import (
    AUTHORITATIVE_SOURCE_ODDS_COLUMNS,
    CANONICAL_MARKET_COLUMNS,
    CANONICAL_ODDS_COLUMNS,
    MARKET_CLASS_ORDER,
    MARKET_NATURAL_KEY,
    MARKET_OVERROUND_COLUMN,
    MARKET_PROBABILITY_COLUMNS,
    MARKET_PROBABILITY_OUTPUT_COLUMNS,
    MarketInputAudit,
    MarketSchemaError,
    closing_market_probabilities,
    normalize_closing_market,
    prepare_closing_market,
)
from fbai.evaluation.report import (
    AggregateEvaluation,
    EvaluationReport,
    FoldEvaluation,
    MetricRecord,
)
from fbai.evaluation.runner import evaluate_lr52

__all__ = [
    "AUTHORITATIVE_SOURCE_ODDS_COLUMNS",
    "AggregateEvaluation",
    "AlignedFoldComparison",
    "BaselineTargetError",
    "CANONICAL_MARKET_COLUMNS",
    "CANONICAL_ODDS_COLUMNS",
    "CoverageSlice",
    "EvaluationReport",
    "FoldEvaluation",
    "MARKET_CLASS_ORDER",
    "MARKET_NATURAL_KEY",
    "MARKET_OVERROUND_COLUMN",
    "MARKET_PROBABILITY_COLUMNS",
    "MARKET_PROBABILITY_OUTPUT_COLUMNS",
    "MarketAlignmentError",
    "MarketBenchmarkSummary",
    "MarketComparisonReport",
    "MarketCoverage",
    "MarketInputAudit",
    "MarketSchemaError",
    "MetricRecord",
    "TrainingPriorBaseline",
    "align_closing_market",
    "closing_market_probabilities",
    "evaluate_lr52",
    "evaluate_lr52_vs_closing_market",
    "fit_training_prior",
    "normalize_closing_market",
    "prepare_closing_market",
    "uniform_probabilities",
]
