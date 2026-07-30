"""Optional, isolated research candidates built on the stable public pipeline."""

from fbai.research.common import (
    CandidateAggregate,
    CandidateComparisonReport,
    CandidateFoldEvaluation,
    ResearchGate,
    ResearchGateResult,
)
from fbai.research.deep_capacity import (
    AUTHORITATIVE_CONFIGS,
    DeepCapacityComparisonReport,
    DeepCapacityConfig,
    evaluate_deep_capacity,
)
from fbai.research.pseudo_xg import (
    PSEUDO_XG_FEATURE_COLUMNS,
    PseudoXGComparisonReport,
    PseudoXGConfig,
    evaluate_pseudo_xg_candidate,
)
from fbai.research.understat_xg import (
    UNDERSTAT_XG_FEATURE_COLUMNS,
    UnderstatXGComparisonReport,
    UnderstatXGConfig,
    align_understat_xg,
    evaluate_understat_xg_candidate,
)

__all__ = [
    "CandidateAggregate",
    "CandidateComparisonReport",
    "CandidateFoldEvaluation",
    "AUTHORITATIVE_CONFIGS",
    "DeepCapacityComparisonReport",
    "DeepCapacityConfig",
    "PSEUDO_XG_FEATURE_COLUMNS",
    "PseudoXGComparisonReport",
    "PseudoXGConfig",
    "ResearchGate",
    "ResearchGateResult",
    "UNDERSTAT_XG_FEATURE_COLUMNS",
    "UnderstatXGComparisonReport",
    "UnderstatXGConfig",
    "align_understat_xg",
    "evaluate_deep_capacity",
    "evaluate_understat_xg_candidate",
    "evaluate_pseudo_xg_candidate",
]
