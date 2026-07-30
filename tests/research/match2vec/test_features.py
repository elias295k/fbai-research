from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fbai.features.schema import FEATURE_COLUMNS, feature_columns
from fbai.research.match2vec.config import Match2VecConfig
from fbai.research.match2vec.corpus import DESCRIPTOR_COUNT, SequenceBatch
from fbai.research.match2vec.features import (
    REPRESENTATION_FEATURE_COLUMNS,
    build_representation_feature_table,
    combined_candidate_feature_columns,
    representation_feature_columns,
)
from fbai.research.match2vec.model import Match2VecSequenceModel

pytest.importorskip("torch")


def _batch(rows: int) -> SequenceBatch:
    rng = np.random.default_rng(501)
    sequences = rng.normal(size=(rows, 2, DESCRIPTOR_COUNT)).astype(np.float32)
    mask = np.ones((rows, 2), dtype=bool)
    keys = tuple(
        (
            np.datetime64(f"2024-02-{position + 1:02d}"),  # type: ignore[arg-type]
            "SYN1",
            f"Synthetic Home {position}",
            f"Synthetic Away {position}",
        )
        for position in range(rows)
    )
    return SequenceBatch(keys, sequences, mask, -sequences, mask)


def _fitted_default_shape_model() -> tuple[Match2VecSequenceModel, SequenceBatch]:
    fit = _batch(9)
    validation = _batch(3)
    fit_numeric = np.ones((9, len(FEATURE_COLUMNS)), dtype=np.float32)
    validation_numeric = np.ones((3, len(FEATURE_COLUMNS)), dtype=np.float32)
    model = Match2VecSequenceModel(
        league_count=1,
        numeric_feature_count=len(FEATURE_COLUMNS),
        config=replace(
            Match2VecConfig(),
            max_epochs=2,
            early_stopping_patience=1,
        ),
    )
    model.fit(
        fit_batch=fit,
        fit_league_ids=np.zeros(9, dtype=np.int64),
        fit_numeric=fit_numeric,
        fit_labels=["H", "D", "A"] * 3,
        validation_batch=validation,
        validation_league_ids=np.zeros(3, dtype=np.int64),
        validation_numeric=validation_numeric,
        validation_labels=["H", "D", "A"],
    )
    return model, validation


def test_representation_feature_contract_is_exact_and_isolated() -> None:
    assert representation_feature_columns() == REPRESENTATION_FEATURE_COLUMNS
    assert len(REPRESENTATION_FEATURE_COLUMNS) == 100
    assert len(set(REPRESENTATION_FEATURE_COLUMNS)) == 100
    assert all(column.endswith("_pre") for column in REPRESENTATION_FEATURE_COLUMNS)
    assert set(REPRESENTATION_FEATURE_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert combined_candidate_feature_columns()[:52] == FEATURE_COLUMNS
    assert feature_columns() == FEATURE_COLUMNS
    assert len(feature_columns()) == 52


def test_representation_table_is_key_aligned_numeric_and_finite() -> None:
    model, batch = _fitted_default_shape_model()

    table = build_representation_feature_table(
        model,
        batch,
        np.zeros(batch.row_count, dtype=np.int64),
    )

    assert tuple(table.columns[:4]) == ("MatchDate", "Division", "HomeTeam", "AwayTeam")
    assert tuple(table.columns[4:]) == REPRESENTATION_FEATURE_COLUMNS
    assert len(table) == batch.row_count
    assert np.isfinite(table.loc[:, list(REPRESENTATION_FEATURE_COLUMNS)].to_numpy()).all()


def test_representation_contract_contains_no_labels_stats_or_market_fields() -> None:
    normalized = " ".join(REPRESENTATION_FEATURE_COLUMNS).lower()

    for forbidden in ("target", "ftr", "shots", "goals", "corners", "odds", "avg", "max"):
        assert forbidden not in normalized
