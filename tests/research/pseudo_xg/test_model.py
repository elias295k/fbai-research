from __future__ import annotations

import numpy as np

from fbai.core.leakage import assert_model_inputs_safe
from fbai.features.schema import FEATURE_COLUMNS
from fbai.research.pseudo_xg.features import PSEUDO_XG_FEATURE_COLUMNS
from fbai.research.pseudo_xg.model import (
    candidate_feature_columns,
    fit_pseudo_xg_estimator,
    predict_match_pseudo_xg,
)
from fbai.testing.synthetic import make_synthetic_canonical_matches


def _canonical():
    return make_synthetic_canonical_matches(
        seed=531,
        season_start_years=(2020, 2021, 2022),
        divisions=("SYN1", "SYN2"),
        teams_per_division=4,
    )


def test_estimator_uses_two_team_side_rows_per_valid_training_match() -> None:
    train = _canonical().loc[lambda frame: frame["SeasonStartYear"].le(2021)]
    fitted = fit_pseudo_xg_estimator(train)

    assert fitted.training_match_rows == len(train)
    assert fitted.training_side_rows == 2 * len(train)
    assert fitted.converged
    assert fitted.iterations < fitted.config.poisson_max_iter


def test_test_rows_do_not_affect_fitted_parameters() -> None:
    canonical = _canonical()
    train_mask = canonical["SeasonStartYear"].le(2021)
    changed = canonical.copy(deep=True)
    changed.loc[~train_mask, "HomeShots"] += 500
    changed.loc[~train_mask, "AwayShots"] += 500

    first = fit_pseudo_xg_estimator(canonical.loc[train_mask])
    second = fit_pseudo_xg_estimator(changed.loc[train_mask])

    np.testing.assert_allclose(first.estimator.coef_, second.estimator.coef_, rtol=0, atol=0)
    assert first.estimator.intercept_ == second.estimator.intercept_


def test_more_shots_on_target_increases_estimated_goals_on_source_shaped_data() -> None:
    canonical = _canonical()
    fitted = fit_pseudo_xg_estimator(canonical)
    low = predict_match_pseudo_xg(
        fitted,
        shots_on_target=canonical["HomeTarget"].iloc[:1].mul(0).add(1),
        shots=canonical["HomeShots"].iloc[:1].mul(0).add(8),
    )
    high = predict_match_pseudo_xg(
        fitted,
        shots_on_target=canonical["HomeTarget"].iloc[:1].mul(0).add(6),
        shots=canonical["HomeShots"].iloc[:1].mul(0).add(8),
    )

    assert high[0] > low[0]


def test_candidate_inputs_are_only_the_explicit_52_plus_12_pre_features() -> None:
    columns = candidate_feature_columns()

    assert columns == (*FEATURE_COLUMNS, *PSEUDO_XG_FEATURE_COLUMNS)
    assert len(columns) == 64
    assert_model_inputs_safe(columns, approved_pre_features=columns)
    normalized = " ".join(columns).lower()
    assert "target_1x2" not in normalized
    assert "odds" not in normalized
    assert "homeshots" not in normalized
    assert "fthome" not in normalized
