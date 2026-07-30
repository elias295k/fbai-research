"""Closed LR52 feature selection and train-fitted preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fbai.core.leakage import assert_model_inputs_safe
from fbai.features.schema import FEATURE_COLUMNS, FEATURE_TABLE_COLUMNS


class LR52InputError(ValueError):
    """Raised when a frame cannot satisfy the fixed LR52 input contract."""


def select_lr52_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the approved 52 columns in their canonical order.

    Known Phase 2B metadata and labels may accompany the features. Unknown
    columns are rejected so a same-match field, odds field, or substituted
    numeric field cannot silently enter a fit or prediction path.
    """

    if not isinstance(frame, pd.DataFrame):
        raise LR52InputError("LR52 input must be a pandas DataFrame")
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise LR52InputError(f"Missing LR52 features: {', '.join(missing)}")
    unknown = [column for column in frame.columns if column not in FEATURE_TABLE_COLUMNS]
    if unknown:
        raise LR52InputError(f"Unexpected LR52 input columns: {', '.join(unknown)}")

    assert_model_inputs_safe(FEATURE_COLUMNS, approved_pre_features=FEATURE_COLUMNS)
    for column in FEATURE_COLUMNS:
        if not is_numeric_dtype(frame[column].dtype):
            raise LR52InputError(f"LR52 feature {column} must be numeric")
    selected = frame.loc[:, list(FEATURE_COLUMNS)].copy(deep=True)
    values = selected.to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(values).any():
        raise LR52InputError("LR52 features must not contain infinite values")
    return selected


def build_lr52_preprocessor() -> Pipeline:
    """Build unfitted median-imputation and standard-scaling steps."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
