from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbai.features.build import build_feature_table
from fbai.features.schema import FEATURE_COLUMNS
from fbai.models.logistic import fit_lr52
from fbai.models.preprocessing import LR52InputError, select_lr52_features
from fbai.testing.synthetic import make_synthetic_canonical_matches


def feature_table() -> pd.DataFrame:
    canonical = make_synthetic_canonical_matches(
        seed=501,
        season_start_years=(2021, 2022, 2023),
        divisions=("E0", "D1"),
        teams_per_division=6,
    )
    return build_feature_table(canonical)


def test_exact_52_columns_are_selected_in_approved_order_without_mutation() -> None:
    frame = feature_table()
    before = frame.copy(deep=True)

    selected = select_lr52_features(frame)

    assert tuple(selected.columns) == FEATURE_COLUMNS
    assert selected.shape[1] == 52
    assert "target_1x2" not in selected
    assert "MatchDate" not in selected
    pd.testing.assert_frame_equal(frame, before)


def test_legitimate_nans_are_imputed_and_scaled_from_training_rows() -> None:
    train = feature_table().loc[lambda frame: frame["SeasonStartYear"] <= 2022].copy()

    fitted = fit_lr52(train)

    assert len(fitted.imputer_statistics) == 52
    assert len(fitted.scaler_mean) == 52
    assert len(fitted.scaler_scale) == 52
    assert np.isfinite(fitted.imputer_statistics).all()
    assert np.isfinite(fitted.scaler_mean).all()


def test_changing_future_values_cannot_change_train_preprocessing_statistics() -> None:
    frame = feature_table()
    train = frame.loc[frame["SeasonStartYear"] <= 2022].copy()
    first = fit_lr52(train)
    changed = frame.copy()
    changed.loc[changed["SeasonStartYear"] == 2023, list(FEATURE_COLUMNS)] = 1_000_000.0
    second = fit_lr52(changed.loc[changed["SeasonStartYear"] <= 2022])

    assert first.imputer_statistics == second.imputer_statistics
    assert first.scaler_mean == second.scaler_mean
    assert first.scaler_scale == second.scaler_scale


def test_infinite_and_unknown_inputs_are_rejected() -> None:
    infinite = feature_table()
    infinite.loc[0, FEATURE_COLUMNS[0]] = np.inf
    with pytest.raises(LR52InputError, match="infinite"):
        select_lr52_features(infinite)

    unknown = feature_table().assign(UnknownNumeric=1.0)
    with pytest.raises(LR52InputError, match="Unexpected"):
        select_lr52_features(unknown)
