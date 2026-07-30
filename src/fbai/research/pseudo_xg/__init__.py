"""Fold-local pseudo-xG research candidate, isolated from the LR52 contract."""

from fbai.research.pseudo_xg.config import PseudoXGConfig
from fbai.research.pseudo_xg.evaluation import (
    PseudoXGComparisonReport,
    evaluate_pseudo_xg_candidate,
)
from fbai.research.pseudo_xg.features import (
    PSEUDO_XG_FEATURE_COLUMNS,
    build_pseudo_xg_feature_table,
    build_walk_forward_training_feature_table,
)
from fbai.research.pseudo_xg.model import (
    FittedPseudoXGEstimator,
    fit_pseudo_xg_estimator,
)

__all__ = [
    "FittedPseudoXGEstimator",
    "PSEUDO_XG_FEATURE_COLUMNS",
    "PseudoXGComparisonReport",
    "PseudoXGConfig",
    "build_pseudo_xg_feature_table",
    "build_walk_forward_training_feature_table",
    "evaluate_pseudo_xg_candidate",
    "fit_pseudo_xg_estimator",
]
