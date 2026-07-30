from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbai.evaluation.baselines import (
    BaselineTargetError,
    fit_training_prior,
    uniform_probabilities,
)
from fbai.models.logistic import PROBABILITY_COLUMNS


def test_uniform_baseline_is_exactly_one_third_in_hda_order() -> None:
    frame = pd.DataFrame(index=[7, 9, 11])

    probabilities = uniform_probabilities(frame)

    assert tuple(probabilities.columns) == PROBABILITY_COLUMNS
    assert probabilities.index.tolist() == [7, 9, 11]
    np.testing.assert_array_equal(probabilities.to_numpy(), np.full((3, 3), 1.0 / 3.0))


def test_training_prior_uses_only_training_labels_and_hda_order() -> None:
    train = pd.DataFrame({"target_1x2": ["H", "H", "H", "D", "A"]})
    test = pd.DataFrame({"target_1x2": ["A", "A"]}, index=[4, 5])

    fitted = fit_training_prior(train)
    first = fitted.predict(test)
    changed = test.assign(target_1x2=["H", "D"])
    second = fitted.predict(changed)

    assert fitted.probabilities == (0.6, 0.2, 0.2)
    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=1), 1.0)


def test_training_prior_requires_all_classes() -> None:
    with pytest.raises(BaselineTargetError, match="requires all H/D/A"):
        fit_training_prior(pd.DataFrame({"target_1x2": ["H", "D"]}))
