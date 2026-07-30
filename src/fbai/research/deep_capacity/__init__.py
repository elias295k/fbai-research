"""Source-verified internal deep-capacity audit, isolated from LR52."""

from fbai.research.deep_capacity.config import (
    AUTHORITATIVE_CONFIGS,
    DEEP_MLP,
    SHALLOW_MLP,
    DeepCapacityConfig,
)
from fbai.research.deep_capacity.evaluation import (
    DeepCapacityComparisonReport,
    evaluate_deep_capacity,
)
from fbai.research.deep_capacity.model import (
    DeepCapacityModel,
    deep_capacity_available,
)

__all__ = [
    "AUTHORITATIVE_CONFIGS",
    "DEEP_MLP",
    "SHALLOW_MLP",
    "DeepCapacityComparisonReport",
    "DeepCapacityConfig",
    "DeepCapacityModel",
    "deep_capacity_available",
    "evaluate_deep_capacity",
]
