from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from fbai.models.logistic import PROBABILITY_COLUMNS
from fbai.research.graph_model.config import (
    GRAPH_ONLY_CONTEXT,
    LR52_GRAPH_CONTEXT,
    GraphEmbeddingConfig,
    graph_feature_columns,
)
from fbai.research.graph_model.model import (
    GraphModelInputError,
    fit_graph_classifier,
    graph_classifier_feature_columns,
    predict_graph_classifier_proba,
)


def _model_frame(seed: int = 8) -> tuple[pd.DataFrame, tuple[str, ...]]:
    config = GraphEmbeddingConfig(name="svd2_test", dimension=2)
    columns = graph_feature_columns(config)
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(60, len(columns))), columns=columns)
    frame["target_1x2"] = np.resize(np.asarray(["H", "D", "A"]), len(frame))
    return frame, columns


def test_same_seed_fit_is_deterministic() -> None:
    frame, columns = _model_frame()

    first = fit_graph_classifier(frame, columns, context=GRAPH_ONLY_CONTEXT)
    second = fit_graph_classifier(frame, columns, context=GRAPH_ONLY_CONTEXT)

    assert first.state_fingerprint() == second.state_fingerprint()
    np.testing.assert_allclose(
        predict_graph_classifier_proba(first, frame).to_numpy(),
        predict_graph_classifier_proba(second, frame).to_numpy(),
        rtol=0.0,
        atol=0.0,
    )


def test_probabilities_are_explicit_finite_hda_and_sum_to_one() -> None:
    frame, columns = _model_frame()
    model = fit_graph_classifier(frame, columns, context=GRAPH_ONLY_CONTEXT)

    probabilities = predict_graph_classifier_proba(model, frame)

    assert tuple(probabilities.columns) == PROBABILITY_COLUMNS
    assert np.isfinite(probabilities.to_numpy()).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


def test_preprocessing_fits_only_on_training_rows() -> None:
    frame, columns = _model_frame()
    train = frame.iloc[:45].copy(deep=True)
    future = frame.iloc[45:].copy(deep=True)
    future.loc[:, list(columns)] = 10_000.0

    model = fit_graph_classifier(train, columns, context=GRAPH_ONLY_CONTEXT)
    scaler = model.pipeline.named_steps["scaler"]

    assert isinstance(scaler, StandardScaler)
    expected = train.loc[:, list(columns)].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected)
    probabilities = predict_graph_classifier_proba(model, future)
    assert np.isfinite(probabilities.to_numpy()).all()


def test_context_contracts_have_exact_input_order_and_parameter_count() -> None:
    frame, columns = _model_frame()

    assert graph_classifier_feature_columns(columns, context=GRAPH_ONLY_CONTEXT) == columns
    lr52_columns = graph_classifier_feature_columns(columns, context=LR52_GRAPH_CONTEXT)
    assert lr52_columns[-len(columns) :] == columns
    graph_model = fit_graph_classifier(frame, columns, context=GRAPH_ONLY_CONTEXT)
    assert graph_model.parameter_count == 3 * (len(columns) + 1)


def test_unapproved_graph_names_and_contexts_fail() -> None:
    frame, columns = _model_frame()

    with pytest.raises(GraphModelInputError, match="graph_.*_pre"):
        fit_graph_classifier(frame, ("odds_pre",), context=GRAPH_ONLY_CONTEXT)
    with pytest.raises(GraphModelInputError, match="unsupported"):
        graph_classifier_feature_columns(columns, context="market_graph")
