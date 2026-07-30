"""Optional historical Understat xG research path, isolated from LR52."""

from fbai.research.understat_xg.alignment import (
    AlignedUnderstatXG,
    UnderstatXGAlignmentReport,
    align_understat_xg,
)
from fbai.research.understat_xg.evaluation import (
    UnderstatXGComparisonReport,
    UnderstatXGConfig,
    evaluate_understat_xg_candidate,
)
from fbai.research.understat_xg.features import (
    UNDERSTAT_XG_FEATURE_COLUMNS,
    build_understat_xg_feature_table,
)
from fbai.research.understat_xg.schema import (
    SUPPORTED_DIVISIONS,
    TEAM_ALIASES,
    UNDERSTAT_XG_COLUMNS,
    UnderstatXGSchemaAudit,
    audit_understat_xg_frame,
    normalize_team_name,
    validate_understat_xg_frame,
)

__all__ = [
    "AlignedUnderstatXG",
    "SUPPORTED_DIVISIONS",
    "TEAM_ALIASES",
    "UNDERSTAT_XG_COLUMNS",
    "UNDERSTAT_XG_FEATURE_COLUMNS",
    "UnderstatXGAlignmentReport",
    "UnderstatXGComparisonReport",
    "UnderstatXGConfig",
    "UnderstatXGSchemaAudit",
    "align_understat_xg",
    "audit_understat_xg_frame",
    "build_understat_xg_feature_table",
    "evaluate_understat_xg_candidate",
    "normalize_team_name",
    "validate_understat_xg_frame",
]
