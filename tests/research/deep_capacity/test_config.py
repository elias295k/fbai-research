from __future__ import annotations

import hashlib

import pytest

from fbai.features import feature_columns
from fbai.research.deep_capacity.config import (
    AUTHORITATIVE_CONFIGS,
    DEEP_MLP,
    SHALLOW_MLP,
    DeepCapacityConfig,
)


def test_authoritative_architecture_grid_is_exact() -> None:
    assert AUTHORITATIVE_CONFIGS == (SHALLOW_MLP, DEEP_MLP)
    assert SHALLOW_MLP.name == "h64_do10_wd1e3"
    assert SHALLOW_MLP.hidden_dimensions == (64,)
    assert SHALLOW_MLP.dropout == 0.10
    assert DEEP_MLP.name == "h128_64_do20_wd1e3"
    assert DEEP_MLP.hidden_dimensions == (128, 64)
    assert DEEP_MLP.dropout == 0.20


@pytest.mark.parametrize("configuration", AUTHORITATIVE_CONFIGS)
def test_training_configuration_is_source_exact(
    configuration: DeepCapacityConfig,
) -> None:
    assert configuration.input_feature_count == 52
    assert configuration.output_class_count == 3
    assert configuration.activation == "ReLU"
    assert configuration.normalization_layers == ()
    assert configuration.optimizer == "Adam"
    assert configuration.learning_rate == 3e-3
    assert configuration.weight_decay == 1e-3
    assert configuration.batch_size == 512
    assert configuration.max_epochs == 220
    assert configuration.early_stopping_patience == 18
    assert configuration.early_stopping_min_delta == 1e-5
    assert configuration.validation_fraction == 0.1
    assert configuration.random_seed == 42
    assert configuration.device == "cpu"


def test_parameter_counts_are_exact() -> None:
    assert SHALLOW_MLP.parameter_count() == 3587
    assert DEEP_MLP.parameter_count() == 15235


def test_stable_feature_tuple_is_unchanged() -> None:
    assert len(feature_columns()) == 52
    assert hashlib.sha256("\0".join(feature_columns()).encode()).hexdigest() == (
        "09b9204c283e8adf7e91e98cb1a547183b3f832cc95f815b9845208be822de78"
    )
