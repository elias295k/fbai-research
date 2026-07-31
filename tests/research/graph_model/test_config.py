from __future__ import annotations

import hashlib

import pytest

from fbai.features import FEATURE_COLUMNS
from fbai.research.graph_model.config import (
    AUTHORITATIVE_GRAPH_CONFIGS,
    GRAPH_CLASSIFIER_CONFIG,
    GRAPH_ONLY_CONTEXT,
    LR52_GRAPH_CONTEXT,
    graph_feature_columns,
)


def test_authoritative_graph_configurations_are_exact() -> None:
    first, second = AUTHORITATIVE_GRAPH_CONFIGS

    assert (first.name, first.dimension) == ("svd8_recent10", 8)
    assert (second.name, second.dimension) == ("svd16_recent10", 16)
    for config in AUTHORITATIVE_GRAPH_CONFIGS:
        assert config.history_window_matches == 10
        assert config.starter_weight == 30.0
        assert config.bench_weight == 5.0
        assert config.random_seed == 42


def test_exact_graph_feature_and_classifier_parameter_counts() -> None:
    first, second = AUTHORITATIVE_GRAPH_CONFIGS

    assert len(graph_feature_columns(first)) == first.feature_count == 40
    assert len(graph_feature_columns(second)) == second.feature_count == 72
    assert first.classifier_parameter_count(GRAPH_ONLY_CONTEXT) == 123
    assert first.classifier_parameter_count(LR52_GRAPH_CONTEXT) == 279
    assert second.classifier_parameter_count(GRAPH_ONLY_CONTEXT) == 219
    assert second.classifier_parameter_count(LR52_GRAPH_CONTEXT) == 375


def test_graph_feature_names_are_explicit_strict_prior_fields() -> None:
    for config in AUTHORITATIVE_GRAPH_CONFIGS:
        columns = graph_feature_columns(config)
        assert len(columns) == len(set(columns))
        assert all(column.startswith(f"graph_{config.name}_") for column in columns)
        assert all(column.endswith("_pre") for column in columns)


def test_classifier_configuration_is_source_exact_cpu_logistic() -> None:
    config = GRAPH_CLASSIFIER_CONFIG

    assert config.solver == "lbfgs"
    assert config.penalty == "l2"
    assert config.regularization_c == 1.0
    assert config.max_iter == 2000
    assert config.random_seed == 42
    assert config.device == "cpu"


def test_stable_feature_fingerprint_is_unchanged() -> None:
    assert len(FEATURE_COLUMNS) == 52
    fingerprint = hashlib.sha256("\0".join(FEATURE_COLUMNS).encode()).hexdigest()
    assert fingerprint == ("09b9204c283e8adf7e91e98cb1a547183b3f832cc95f815b9845208be822de78")


def test_unknown_classifier_context_fails() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        AUTHORITATIVE_GRAPH_CONFIGS[0].classifier_feature_count("market_graph")
