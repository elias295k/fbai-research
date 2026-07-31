"""Source-verified prior-only player-availability research API."""

from fbai.research.player_availability.alignment import (
    AvailabilityAlignmentError,
    AvailabilityAlignmentReport,
    AvailabilityInputQuality,
    AvailabilitySourceFrames,
    align_availability_scope,
    prepare_availability_sources,
)
from fbai.research.player_availability.evaluation import (
    AvailabilityComparisonReport,
    AvailabilityEvaluationError,
    evaluate_player_availability,
)
from fbai.research.player_availability.features import (
    AvailabilityFeatureBuild,
    AvailabilityFeatureError,
    build_prior_availability_features,
)
from fbai.research.player_availability.schema import (
    AVAILABILITY_CLASSIFIER_CONFIG,
    PRIOR_AVAILABILITY_FEATURES,
    PRIOR_BASE_FEATURES,
    TIMING_AUDIT,
    AvailabilityClassifierConfig,
    InformationTiming,
)

__all__ = [
    "AVAILABILITY_CLASSIFIER_CONFIG",
    "PRIOR_AVAILABILITY_FEATURES",
    "PRIOR_BASE_FEATURES",
    "TIMING_AUDIT",
    "AvailabilityAlignmentError",
    "AvailabilityAlignmentReport",
    "AvailabilityClassifierConfig",
    "AvailabilityComparisonReport",
    "AvailabilityEvaluationError",
    "AvailabilityFeatureBuild",
    "AvailabilityFeatureError",
    "AvailabilityInputQuality",
    "AvailabilitySourceFrames",
    "InformationTiming",
    "align_availability_scope",
    "build_prior_availability_features",
    "evaluate_player_availability",
    "prepare_availability_sources",
]
