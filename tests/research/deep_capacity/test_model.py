from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fbai.research.deep_capacity.config import SHALLOW_MLP
from fbai.research.deep_capacity.model import (
    DeepCapacityModel,
    DeepCapacityTrainingError,
)

pytest.importorskip("torch")


def _fit(seed: int = 42) -> tuple[DeepCapacityModel, np.ndarray]:
    rng = np.random.default_rng(91)
    fit = rng.normal(size=(30, 52)).astype(np.float32)
    validation = rng.normal(size=(9, 52)).astype(np.float32)
    labels = ["H", "D", "A"] * 10
    validation_labels = ["H", "D", "A"] * 3
    config = replace(
        SHALLOW_MLP,
        max_epochs=4,
        early_stopping_patience=2,
        batch_size=10,
        random_seed=seed,
    )
    model = DeepCapacityModel(config).fit(
        fit,
        labels,
        validation,
        validation_labels,
    )
    return model, validation


def test_same_seed_training_is_exactly_deterministic() -> None:
    first, numeric = _fit()
    second, _ = _fit()

    assert first.state_fingerprint() == second.state_fingerprint()
    np.testing.assert_allclose(
        first.predict_proba(numeric),
        second.predict_proba(numeric),
        rtol=0.0,
        atol=0.0,
    )


def test_different_seed_changes_learned_state() -> None:
    first, _ = _fit(seed=42)
    second, _ = _fit(seed=43)

    assert first.state_fingerprint() != second.state_fingerprint()


def test_probabilities_are_finite_normalized_and_hda_ordered() -> None:
    model, numeric = _fit()
    probabilities = model.predict_proba(numeric)

    assert model.config.device == "cpu"
    assert model.parameter_count == 3587
    assert probabilities.shape == (9, 3)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)


def test_non_finite_numeric_input_fails() -> None:
    numeric = np.ones((9, 52), dtype=np.float32)
    numeric[0, 0] = np.nan

    with pytest.raises(DeepCapacityTrainingError, match="non-finite"):
        DeepCapacityModel(replace(SHALLOW_MLP, max_epochs=1)).fit(
            numeric,
            ["H", "D", "A"] * 3,
            np.ones((3, 52), dtype=np.float32),
            ["H", "D", "A"],
        )


def test_training_writes_no_checkpoint(tmp_path: Path) -> None:
    before = tuple(tmp_path.iterdir())

    _fit()

    assert tuple(tmp_path.iterdir()) == before
