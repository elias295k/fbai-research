"""Frozen source configuration for the temporal graph-model audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from fbai.features.schema import FEATURE_COLUMNS

GRAPH_ONLY_CONTEXT = "graph_only"
LR52_GRAPH_CONTEXT = "lr52_graph"
AUTHORITATIVE_CONTEXTS: tuple[str, str] = (GRAPH_ONLY_CONTEXT, LR52_GRAPH_CONTEXT)


@dataclass(frozen=True, slots=True)
class GraphEmbeddingConfig:
    """One source-defined team-player SVD representation."""

    name: str
    dimension: int
    history_window_matches: int = 10
    starter_weight: float = 30.0
    bench_weight: float = 5.0
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("graph configuration name must be non-empty")
        if self.dimension < 1:
            raise ValueError("graph embedding dimension must be positive")
        if self.history_window_matches < 1:
            raise ValueError("graph history window must be positive")
        if self.starter_weight < 0.0 or self.bench_weight < 0.0:
            raise ValueError("graph lineup weights must be non-negative")

    @property
    def identifier(self) -> str:
        return (
            f"{self.name}_window{self.history_window_matches}"
            f"_starter{self.starter_weight:g}_bench{self.bench_weight:g}"
            f"_seed{self.random_seed}"
        )

    @property
    def feature_count(self) -> int:
        return 4 * self.dimension + 8

    def classifier_feature_count(self, context: str) -> int:
        if context == GRAPH_ONLY_CONTEXT:
            return self.feature_count
        if context == LR52_GRAPH_CONTEXT:
            return len(FEATURE_COLUMNS) + self.feature_count
        raise ValueError(f"unsupported graph-model context: {context}")

    def classifier_parameter_count(self, context: str) -> int:
        """Return nominal multinomial coefficients and intercepts for H/D/A."""

        return 3 * (self.classifier_feature_count(context) + 1)

    def to_dict(self) -> dict[str, str | int | float]:
        values: dict[str, str | int | float] = asdict(self)
        values["identifier"] = self.identifier
        values["feature_count"] = self.feature_count
        values["graph_only_parameter_count"] = self.classifier_parameter_count(GRAPH_ONLY_CONTEXT)
        values["lr52_graph_parameter_count"] = self.classifier_parameter_count(LR52_GRAPH_CONTEXT)
        return values


@dataclass(frozen=True, slots=True)
class GraphClassifierConfig:
    """Exact regularized logistic classifier used by the source audit."""

    solver: str = "lbfgs"
    penalty: str = "l2"
    regularization_c: float = 1.0
    max_iter: int = 2000
    fit_intercept: bool = True
    class_weight: str | None = None
    random_seed: int = 42
    tolerance: float = 1e-4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.solver != "lbfgs" or self.penalty != "l2":
            raise ValueError("graph classifier is fixed to lbfgs L2 logistic regression")
        if self.regularization_c <= 0.0 or self.max_iter < 1 or self.tolerance <= 0.0:
            raise ValueError("graph classifier optimization settings are invalid")
        if self.class_weight is not None:
            raise ValueError("graph classifier class_weight is fixed to None")
        if self.device != "cpu":
            raise ValueError("graph-model research supports CPU execution only")

    @property
    def identifier(self) -> str:
        return "median_standardized_lbfgs_l2_c1_iter2000_seed42"

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        values: dict[str, str | int | float | bool | None] = asdict(self)
        values["identifier"] = self.identifier
        return values


SVD8_RECENT10 = GraphEmbeddingConfig(name="svd8_recent10", dimension=8)
SVD16_RECENT10 = GraphEmbeddingConfig(name="svd16_recent10", dimension=16)

AUTHORITATIVE_GRAPH_CONFIGS: tuple[GraphEmbeddingConfig, GraphEmbeddingConfig] = (
    SVD8_RECENT10,
    SVD16_RECENT10,
)
GRAPH_CLASSIFIER_CONFIG = GraphClassifierConfig()

assert SVD8_RECENT10.feature_count == 40
assert SVD16_RECENT10.feature_count == 72
assert SVD8_RECENT10.classifier_parameter_count(GRAPH_ONLY_CONTEXT) == 123
assert SVD8_RECENT10.classifier_parameter_count(LR52_GRAPH_CONTEXT) == 279
assert SVD16_RECENT10.classifier_parameter_count(GRAPH_ONLY_CONTEXT) == 219
assert SVD16_RECENT10.classifier_parameter_count(LR52_GRAPH_CONTEXT) == 375


def graph_feature_columns(config: GraphEmbeddingConfig) -> tuple[str, ...]:
    """Return the exact ordered source feature names for one representation."""

    prefix = f"graph_{config.name}"
    columns: list[str] = []
    for side in ("H", "A"):
        columns.extend(f"{prefix}_{side}_e{index:02d}_pre" for index in range(config.dimension))
    columns.extend(f"{prefix}_diff_e{index:02d}_pre" for index in range(config.dimension))
    columns.extend(f"{prefix}_absdiff_e{index:02d}_pre" for index in range(config.dimension))
    columns.extend(
        (
            f"{prefix}_cosine_pre",
            f"{prefix}_l2_pre",
            f"{prefix}_H_known_weight_share_pre",
            f"{prefix}_A_known_weight_share_pre",
            f"{prefix}_diff_known_weight_share_pre",
            f"{prefix}_H_pool_players_pre",
            f"{prefix}_A_pool_players_pre",
            f"{prefix}_diff_pool_players_pre",
        )
    )
    return tuple(columns)
