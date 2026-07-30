from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from fbai.research.pseudo_xg.config import PseudoXGConfig


def test_exact_source_configuration_is_frozen() -> None:
    config = PseudoXGConfig()

    assert config.estimator == "PoissonRegressor"
    assert config.estimator_target == "goals"
    assert config.estimator_predictors == ("shots_on_target", "shots_off_target")
    assert config.poisson_alpha == 1e-6
    assert config.poisson_max_iter == 1000
    assert config.poisson_solver == "lbfgs"
    assert config.rolling_windows == (5, 10)
    assert config.estimator_scope == "outer_training_fold_all_divisions"
    assert config.history_division_scope == "Division"
    assert config.history_season_scope == "SeasonStartYear"
    assert config.candidate_base_feature_count == 52
    assert config.candidate_pseudo_xg_feature_count == 12
    assert config.logistic_max_iter == 2000
    assert config.logistic_random_state == 42


def test_configuration_is_immutable_and_json_safe() -> None:
    config = PseudoXGConfig()

    with pytest.raises(FrozenInstanceError):
        config.poisson_alpha = 1.0  # type: ignore[misc]
    payload = config.to_dict()
    assert payload["rolling_windows"] == [5, 10]
    assert payload["formula"] == config.formula


@pytest.mark.parametrize(
    "changed",
    [
        {"poisson_alpha": 0.1},
        {"rolling_windows": (3, 5)},
        {"candidate_pseudo_xg_feature_count": 11},
        {"logistic_solver": "saga"},
    ],
)
def test_source_defining_changes_are_rejected(changed: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(PseudoXGConfig(), **changed)
