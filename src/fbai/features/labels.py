"""Target labels derived from canonical full-time outcomes."""

from __future__ import annotations

import pandas as pd

TARGET_COLUMNS: tuple[str, ...] = (
    "target_1x2",
    "target_home_win",
    "target_draw",
    "target_away_win",
    "target_ou25",
)

_VALID_RESULTS = frozenset({"H", "D", "A"})


class LabelValidationError(ValueError):
    """Raised when canonical outcomes cannot produce valid targets."""


def _one_hot(results: pd.Series, label: str) -> pd.Series:
    return results.eq(label).astype("Int8")


def add_target_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the verified five-label contract attached.

    ``target_ou25`` is one when full-time total goals exceed 2.5. Canonical
    data requires a valid H/D/A result; invalid or missing values fail rather
    than silently creating partial labels.
    """

    required = ("FTR", "FTHome", "FTAway")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise LabelValidationError(f"Missing label source columns: {', '.join(missing)}")

    invalid = ~frame["FTR"].isin(_VALID_RESULTS)
    if bool(invalid.any()):
        values = sorted(set(frame.loc[invalid, "FTR"].astype(str)))
        raise LabelValidationError(f"FTR contains values outside H/D/A: {values}")

    out = frame.copy(deep=True)
    results = out["FTR"]
    out["target_1x2"] = results.astype("string")
    out["target_home_win"] = _one_hot(results, "H")
    out["target_draw"] = _one_hot(results, "D")
    out["target_away_win"] = _one_hot(results, "A")

    home_goals = pd.to_numeric(out["FTHome"], errors="coerce").astype("Float64")
    away_goals = pd.to_numeric(out["FTAway"], errors="coerce").astype("Float64")
    total_goals = home_goals + away_goals
    over = total_goals.gt(2.5).astype("Int8")
    over[total_goals.isna()] = pd.NA
    out["target_ou25"] = over
    return out


# Narrow compatibility alias for the verified source API.
add_labels = add_target_labels
