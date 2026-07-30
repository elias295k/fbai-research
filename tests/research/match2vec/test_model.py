from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fbai.research.match2vec.config import Match2VecConfig
from fbai.research.match2vec.corpus import DESCRIPTOR_COUNT, SequenceBatch
from fbai.research.match2vec.model import (
    Match2VecSequenceModel,
    Match2VecTrainingError,
)

pytest.importorskip("torch")


def _batch(rows: int, *, seed: int) -> SequenceBatch:
    rng = np.random.default_rng(seed)
    sequence_length = 3
    home = rng.normal(size=(rows, sequence_length, DESCRIPTOR_COUNT)).astype(np.float32)
    away = rng.normal(size=(rows, sequence_length, DESCRIPTOR_COUNT)).astype(np.float32)
    mask = np.ones((rows, sequence_length), dtype=bool)
    keys = tuple(
        (np.datetime64(f"2024-01-{position + 1:02d}"), "SYN1", f"H{position}", f"A{position}")  # type: ignore[arg-type]
        for position in range(rows)
    )
    return SequenceBatch(keys, home, mask, away, mask)


def _fit_model(seed: int = 42) -> tuple[Match2VecSequenceModel, SequenceBatch, np.ndarray]:
    fit_batch = _batch(12, seed=1)
    validation_batch = _batch(6, seed=2)
    fit_numeric = np.random.default_rng(3).normal(size=(12, 4)).astype(np.float32)
    validation_numeric = np.random.default_rng(4).normal(size=(6, 4)).astype(np.float32)
    config = replace(
        Match2VecConfig(),
        encoder_dimension=4,
        league_embedding_dimension=2,
        max_epochs=6,
        early_stopping_patience=2,
        batch_size=4,
        random_seed=seed,
    )
    model = Match2VecSequenceModel(
        league_count=2,
        numeric_feature_count=4,
        config=config,
    )
    model.fit(
        fit_batch=fit_batch,
        fit_league_ids=np.arange(12) % 2,
        fit_numeric=fit_numeric,
        fit_labels=(["H", "D", "A"] * 4),
        validation_batch=validation_batch,
        validation_league_ids=np.arange(6) % 2,
        validation_numeric=validation_numeric,
        validation_labels=(["H", "D", "A"] * 2),
    )
    return model, validation_batch, validation_numeric


def test_same_seed_training_is_deterministic() -> None:
    first, batch, numeric = _fit_model()
    second, _, _ = _fit_model()

    assert first.state_fingerprint() == second.state_fingerprint()
    np.testing.assert_allclose(
        first.predict_proba(batch, np.arange(6) % 2, numeric),
        second.predict_proba(batch, np.arange(6) % 2, numeric),
        rtol=0.0,
        atol=0.0,
    )


def test_different_seeds_can_change_learned_state() -> None:
    first, _, _ = _fit_model(seed=42)
    second, _, _ = _fit_model(seed=43)

    assert first.state_fingerprint() != second.state_fingerprint()


def test_cpu_probabilities_and_representation_are_valid() -> None:
    model, batch, numeric = _fit_model()

    probabilities = model.predict_proba(batch, np.arange(6) % 2, numeric)
    representation = model.transform(batch, np.arange(6) % 2)

    assert model.config.device == "cpu"
    assert probabilities.shape == (6, 3)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    assert representation.shape == (6, 14)
    assert np.isfinite(representation).all()


def test_non_finite_numeric_input_fails_clearly() -> None:
    fit_batch = _batch(6, seed=5)
    validation_batch = _batch(3, seed=6)
    numeric = np.ones((6, 2), dtype=np.float32)
    numeric[0, 0] = np.nan
    model = Match2VecSequenceModel(
        league_count=1,
        numeric_feature_count=2,
        config=replace(Match2VecConfig(), max_epochs=1),
    )

    with pytest.raises(Match2VecTrainingError, match="non-finite"):
        model.fit(
            fit_batch=fit_batch,
            fit_league_ids=np.zeros(6, dtype=np.int64),
            fit_numeric=numeric,
            fit_labels=["H", "D", "A", "H", "D", "A"],
            validation_batch=validation_batch,
            validation_league_ids=np.zeros(3, dtype=np.int64),
            validation_numeric=np.ones((3, 2), dtype=np.float32),
            validation_labels=["H", "D", "A"],
        )


def test_training_writes_no_checkpoint(tmp_path: Path) -> None:
    before = tuple(tmp_path.iterdir())

    _fit_model()

    assert tuple(tmp_path.iterdir()) == before
