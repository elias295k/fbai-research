"""Explicit feature contract for learned Match2Vec representations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fbai.data.schema import NATURAL_KEY
from fbai.features.schema import FEATURE_COLUMNS
from fbai.research.match2vec.config import Match2VecConfig
from fbai.research.match2vec.corpus import SequenceBatch
from fbai.research.match2vec.model import Match2VecSequenceModel


def _names(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{position:03d}_pre" for position in range(count))


_DEFAULT = Match2VecConfig()
REPRESENTATION_FEATURE_COLUMNS: tuple[str, ...] = (
    *_names("M2VHomeState", _DEFAULT.encoder_dimension),
    *_names("M2VAwayState", _DEFAULT.encoder_dimension),
    *_names("M2VStateDiff", _DEFAULT.encoder_dimension),
    *_names("M2VLeague", _DEFAULT.league_embedding_dimension),
)

assert len(REPRESENTATION_FEATURE_COLUMNS) == 100
assert len(REPRESENTATION_FEATURE_COLUMNS) == len(set(REPRESENTATION_FEATURE_COLUMNS))
assert all(name.endswith("_pre") for name in REPRESENTATION_FEATURE_COLUMNS)


def representation_feature_columns() -> tuple[str, ...]:
    """Return the isolated, ordered Match2Vec representation contract."""

    return REPRESENTATION_FEATURE_COLUMNS


def combined_candidate_feature_columns() -> tuple[str, ...]:
    """Return exact LR52 inputs followed by the learned representation contract."""

    return (*FEATURE_COLUMNS, *REPRESENTATION_FEATURE_COLUMNS)


def build_representation_feature_table(
    model: Match2VecSequenceModel,
    batch: SequenceBatch,
    league_ids: np.ndarray,
) -> pd.DataFrame:
    """Create a key-aligned finite table without labels, stats, odds, or vocabularies."""

    values = model.transform(batch, league_ids)
    if values.shape[1] != len(REPRESENTATION_FEATURE_COLUMNS):
        raise ValueError("Fitted representation does not match the Phase 5A feature contract")
    if not np.isfinite(values).all():
        raise ValueError("Representation features must all be finite")
    keys = pd.DataFrame(batch.keys, columns=list(NATURAL_KEY))
    features = pd.DataFrame(values, columns=list(REPRESENTATION_FEATURE_COLUMNS))
    return pd.concat([keys, features], axis=1)
