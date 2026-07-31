"""Optional, isolated research candidates built on the stable public pipeline."""

from fbai.research.catalog import (
    PROGRAMME_CONCLUSION,
    RESULT_ALLOWLIST,
    CatalogValidationError,
    CrossPopulationComparisonError,
    ResearchCatalog,
    build_research_catalog,
    load_research_catalog,
)
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
from fbai.research.graph_model import (
    AUTHORITATIVE_GRAPH_CONFIGS,
    GraphComparisonReport,
    GraphEmbeddingConfig,
    evaluate_graph_model,
)
from fbai.research.player_availability import (
    PRIOR_AVAILABILITY_FEATURES,
    AvailabilityComparisonReport,
    evaluate_player_availability,
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
    "CatalogValidationError",
    "CrossPopulationComparisonError",
    "AUTHORITATIVE_CONFIGS",
    "AUTHORITATIVE_GRAPH_CONFIGS",
    "DeepCapacityComparisonReport",
    "DeepCapacityConfig",
    "GraphComparisonReport",
    "GraphEmbeddingConfig",
    "PRIOR_AVAILABILITY_FEATURES",
    "AvailabilityComparisonReport",
    "PSEUDO_XG_FEATURE_COLUMNS",
    "PROGRAMME_CONCLUSION",
    "PseudoXGComparisonReport",
    "PseudoXGConfig",
    "ResearchGate",
    "ResearchGateResult",
    "ResearchCatalog",
    "RESULT_ALLOWLIST",
    "UNDERSTAT_XG_FEATURE_COLUMNS",
    "UnderstatXGComparisonReport",
    "UnderstatXGConfig",
    "align_understat_xg",
    "build_research_catalog",
    "evaluate_deep_capacity",
    "evaluate_graph_model",
    "evaluate_player_availability",
    "evaluate_understat_xg_candidate",
    "evaluate_pseudo_xg_candidate",
    "load_research_catalog",
]
