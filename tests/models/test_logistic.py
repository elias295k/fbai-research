from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import fbai.models.logistic as logistic_module
from fbai.features.build import build_feature_table
from fbai.features.labels import TARGET_COLUMNS
from fbai.features.schema import FEATURE_COLUMNS
from fbai.models.logistic import (
    PROBABILITY_CLASS_ORDER,
    PROBABILITY_COLUMNS,
    LR52Config,
    LR52ConvergenceError,
    LR52TargetError,
    build_lr52_pipeline,
    fit_lr52,
    predict_lr52_proba,
)
from fbai.models.preprocessing import LR52InputError
from fbai.testing.synthetic import make_synthetic_canonical_matches


def feature_table() -> pd.DataFrame:
    canonical = make_synthetic_canonical_matches(
        seed=502,
        season_start_years=(2021, 2022, 2023),
        divisions=("E0", "D1"),
        teams_per_division=6,
    )
    return build_feature_table(canonical)


def test_source_verified_configuration_is_explicit() -> None:
    config = LR52Config()
    pipeline = build_lr52_pipeline(config)
    classifier = pipeline.named_steps["logistic"]

    assert isinstance(classifier, LogisticRegression)
    assert config.penalty == "l2"
    assert classifier.solver == "lbfgs"
    assert classifier.C == 1.0
    assert classifier.max_iter == 2000
    assert classifier.fit_intercept is True
    assert classifier.class_weight is None
    assert classifier.random_state == 42
    assert classifier.tol == 1e-4


def test_fit_rejects_missing_or_substituted_features_and_invalid_targets() -> None:
    frame = feature_table()
    with pytest.raises(LR52InputError, match="Missing"):
        fit_lr52(frame.drop(columns=FEATURE_COLUMNS[-1]))
    substituted = frame.drop(columns=FEATURE_COLUMNS[-1]).assign(Unknown_pre=1.0)
    with pytest.raises(LR52InputError):
        fit_lr52(substituted)

    invalid = frame.copy()
    invalid.loc[0, "target_1x2"] = "X"
    with pytest.raises(LR52TargetError, match="outside H/D/A"):
        fit_lr52(invalid)

    two_class = frame.loc[frame["target_1x2"] != "A"].copy()
    with pytest.raises(LR52TargetError, match="requires all H/D/A"):
        fit_lr52(two_class)


def test_same_match_odds_unknown_and_target_substitutions_fail_before_fit() -> None:
    frame = feature_table()
    for column in ("HomeShots_pre", "AvgH_pre", "UnknownNumeric"):
        with pytest.raises(LR52InputError, match="Unexpected"):
            fit_lr52(frame.assign(**{column: 1.0}))

    target_as_feature = frame.drop(columns=FEATURE_COLUMNS[-1]).rename(
        columns={"target_1x2": FEATURE_COLUMNS[-1]}
    )
    with pytest.raises(LR52InputError, match="must be numeric"):
        fit_lr52(target_as_feature)


def test_fit_invokes_semantic_guard_immediately_before_pipeline_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    original = logistic_module.assert_model_inputs_safe

    def recording_guard(
        columns: tuple[str, ...],
        *,
        approved_pre_features: tuple[str, ...],
    ) -> None:
        calls.append(tuple(columns))
        original(columns, approved_pre_features=approved_pre_features)

    monkeypatch.setattr(logistic_module, "assert_model_inputs_safe", recording_guard)

    fit_lr52(feature_table())

    assert calls == [FEATURE_COLUMNS]


def test_prediction_needs_no_targets_and_explicitly_remaps_internal_classes() -> None:
    frame = feature_table()
    train = frame.loc[frame["SeasonStartYear"] <= 2022]
    test = frame.loc[frame["SeasonStartYear"] == 2023]
    fitted = fit_lr52(train)
    prediction_input = test.drop(columns=list(TARGET_COLUMNS))
    before = prediction_input.copy(deep=True)

    probabilities = predict_lr52_proba(fitted, prediction_input)
    raw = fitted.pipeline.predict_proba(prediction_input.loc[:, list(FEATURE_COLUMNS)])
    order = [fitted.sklearn_classes.index(label) for label in PROBABILITY_CLASS_ORDER]

    assert fitted.sklearn_classes == ("A", "D", "H")
    assert tuple(probabilities.columns) == PROBABILITY_COLUMNS
    np.testing.assert_allclose(probabilities.to_numpy(), raw[:, order], rtol=0.0, atol=0.0)
    assert np.isfinite(probabilities.to_numpy()).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all().all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-8)
    pd.testing.assert_frame_equal(prediction_input, before)


def test_repeated_fit_and_prediction_are_deterministic() -> None:
    frame = feature_table()
    train = frame.loc[frame["SeasonStartYear"] <= 2022]
    test = frame.loc[frame["SeasonStartYear"] == 2023]

    first = predict_lr52_proba(fit_lr52(train), test)
    second = predict_lr52_proba(fit_lr52(train), test)

    pd.testing.assert_frame_equal(first, second)


def test_non_convergence_raises_explicit_failure() -> None:
    with pytest.raises(LR52ConvergenceError, match="failed to converge"):
        fit_lr52(feature_table(), config=LR52Config(max_iter=1))
