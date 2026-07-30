"""Optional, isolated research candidates built on the stable public pipeline."""

from fbai.research.common import (
    CandidateAggregate,
    CandidateComparisonReport,
    CandidateFoldEvaluation,
    ResearchGate,
    ResearchGateResult,
)
from fbai.research.pseudo_xg import (
    PSEUDO_XG_FEATURE_COLUMNS,
    PseudoXGComparisonReport,
    PseudoXGConfig,
    evaluate_pseudo_xg_candidate,
)

__all__ = [
    "CandidateAggregate",
    "CandidateComparisonReport",
    "CandidateFoldEvaluation",
    "PSEUDO_XG_FEATURE_COLUMNS",
    "PseudoXGComparisonReport",
    "PseudoXGConfig",
    "ResearchGate",
    "ResearchGateResult",
    "evaluate_pseudo_xg_candidate",
]
