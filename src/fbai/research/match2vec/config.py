"""Frozen source-verified configuration for the selected Match2Vec candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Match2VecConfig:
    """Configuration selected by the committed development-only source experiment."""

    variant: str = "M2V-SEQ"
    sequence_length: int = 10
    descriptor_count: int = 13
    encoder_dimension: int = 32
    league_embedding_dimension: int = 4
    use_lr52_numeric: bool = True
    dropout: float = 0.1
    optimizer: str = "Adam"
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 300
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-5
    validation_fraction: float = 0.1
    random_seed: int = 42
    device: str = "cpu"
    minimum_token_frequency: int = 1
    unknown_dropout_probability: float = 0.0
    negative_sample_count: int = 0

    def __post_init__(self) -> None:
        if self.variant != "M2V-SEQ":
            raise ValueError("The public Phase 5A candidate is fixed to M2V-SEQ")
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if self.descriptor_count != 13:
            raise ValueError("descriptor_count is fixed to the 13 source descriptors")
        if self.encoder_dimension < 1 or self.league_embedding_dimension < 1:
            raise ValueError("embedding dimensions must be positive")
        if not self.use_lr52_numeric:
            raise ValueError("The selected candidate requires the exact LR52 numeric tuple")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.optimizer != "Adam":
            raise ValueError("optimizer is fixed to Adam")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("batch_size and max_epochs must be positive")
        if self.early_stopping_patience < 1 or self.early_stopping_min_delta < 0.0:
            raise ValueError("early-stopping settings are invalid")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if self.device != "cpu":
            raise ValueError("Phase 5A supports deterministic CPU execution only")
        if self.minimum_token_frequency != 1:
            raise ValueError("The source vocabulary includes every training team token")
        if self.unknown_dropout_probability != 0.0:
            raise ValueError("M2V-SEQ does not apply team-token UNK dropout")
        if self.negative_sample_count != 0:
            raise ValueError("M2V-SEQ is supervised and uses no negative sampling")

    @property
    def identifier(self) -> str:
        return "m2v_seq_k10_enc32_league4_lr52_adam3e3_wd1e4_seed42"

    @property
    def representation_feature_count(self) -> int:
        return 3 * self.encoder_dimension + self.league_embedding_dimension

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return {name: value for name, value in asdict(self).items()}
