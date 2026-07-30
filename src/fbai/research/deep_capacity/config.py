"""Frozen source configurations for the internal deep-capacity audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DeepCapacityConfig:
    """One source-defined numeric MLP architecture and training contract."""

    name: str
    hidden_dimensions: tuple[int, ...]
    dropout: float
    input_feature_count: int = 52
    output_class_count: int = 3
    activation: str = "ReLU"
    normalization_layers: tuple[str, ...] = ()
    optimizer: str = "Adam"
    learning_rate: float = 3e-3
    weight_decay: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 220
    early_stopping_patience: int = 18
    early_stopping_min_delta: float = 1e-5
    validation_fraction: float = 0.1
    random_seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("deep-capacity configuration name must be non-empty")
        if not self.hidden_dimensions or any(width < 1 for width in self.hidden_dimensions):
            raise ValueError("deep-capacity hidden dimensions must be positive")
        if self.input_feature_count != 52 or self.output_class_count != 3:
            raise ValueError("deep-capacity input/output dimensions are fixed to 52 and 3")
        if self.activation != "ReLU":
            raise ValueError("deep-capacity activation is fixed to ReLU")
        if self.normalization_layers:
            raise ValueError("the source deep-capacity MLPs use no normalization layers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("deep-capacity dropout must be in [0, 1)")
        if self.optimizer != "Adam":
            raise ValueError("deep-capacity optimizer is fixed to Adam")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning rate must be positive and weight decay non-negative")
        if self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("batch size and maximum epochs must be positive")
        if self.early_stopping_patience < 1 or self.early_stopping_min_delta < 0.0:
            raise ValueError("deep-capacity early-stopping settings are invalid")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation fraction must be between zero and one")
        if self.device != "cpu":
            raise ValueError("deep-capacity research supports deterministic CPU execution only")

    @property
    def identifier(self) -> str:
        hidden = "_".join(str(width) for width in self.hidden_dimensions)
        dropout_percent = int(round(self.dropout * 100))
        return f"deep_lr52_h{hidden}_do{dropout_percent}_adam3e3_wd1e3_seed42"

    def parameter_count(self, *, input_feature_count: int = 52) -> int:
        """Return trainable Linear weights and biases for the exact network."""

        if input_feature_count != self.input_feature_count:
            raise ValueError("parameter count requires the exact 52-feature input")
        dimensions = (input_feature_count, *self.hidden_dimensions, self.output_class_count)
        return sum(
            input_width * output_width + output_width
            for input_width, output_width in zip(dimensions[:-1], dimensions[1:], strict=True)
        )

    def to_dict(self) -> dict[str, object]:
        values: dict[str, object] = asdict(self)
        values["hidden_dimensions"] = list(self.hidden_dimensions)
        values["normalization_layers"] = list(self.normalization_layers)
        values["identifier"] = self.identifier
        values["parameter_count"] = self.parameter_count()
        return values


SHALLOW_MLP = DeepCapacityConfig(
    name="h64_do10_wd1e3",
    hidden_dimensions=(64,),
    dropout=0.10,
)

DEEP_MLP = DeepCapacityConfig(
    name="h128_64_do20_wd1e3",
    hidden_dimensions=(128, 64),
    dropout=0.20,
)

AUTHORITATIVE_CONFIGS: tuple[DeepCapacityConfig, DeepCapacityConfig] = (
    SHALLOW_MLP,
    DEEP_MLP,
)

assert SHALLOW_MLP.parameter_count() == 3587
assert DEEP_MLP.parameter_count() == 15235
