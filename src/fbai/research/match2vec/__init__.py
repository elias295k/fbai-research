"""Optional source-verified Match2Vec sequence candidate."""

from fbai.research.match2vec.config import Match2VecConfig
from fbai.research.match2vec.corpus import (
    DESCRIPTOR_NAMES,
    Match2VecVocabulary,
    MatchSequenceCorpus,
    SequenceBatch,
    build_match_sequence_corpus,
    build_train_vocabulary,
)
from fbai.research.match2vec.features import (
    REPRESENTATION_FEATURE_COLUMNS,
    combined_candidate_feature_columns,
    representation_feature_columns,
)
from fbai.research.match2vec.model import (
    Match2VecDependencyError,
    Match2VecSequenceModel,
    match2vec_available,
)

__all__ = [
    "DESCRIPTOR_NAMES",
    "REPRESENTATION_FEATURE_COLUMNS",
    "Match2VecConfig",
    "Match2VecDependencyError",
    "Match2VecSequenceModel",
    "Match2VecVocabulary",
    "MatchSequenceCorpus",
    "SequenceBatch",
    "build_match_sequence_corpus",
    "build_train_vocabulary",
    "combined_candidate_feature_columns",
    "match2vec_available",
    "representation_feature_columns",
]
