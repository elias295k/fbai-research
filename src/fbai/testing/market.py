"""Deterministic, separate synthetic closing-market records for tests."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from fbai.evaluation.market import (
    CANONICAL_MARKET_COLUMNS,
    CANONICAL_ODDS_COLUMNS,
    MARKET_NATURAL_KEY,
)

SyntheticMarketMode = Literal[
    "complete",
    "missing_rows",
    "incomplete",
    "duplicate",
    "invalid",
    "shuffled",
]


def make_synthetic_closing_market(
    matches: pd.DataFrame,
    *,
    seed: int = 42,
    mode: SyntheticMarketMode = "complete",
) -> pd.DataFrame:
    """Create invented decimal closing odds keyed to synthetic matches.

    The output remains separate from the canonical and feature tables. Modes
    intentionally produce deterministic contract violations for negative
    tests; ``complete`` and ``shuffled`` contain only valid rows.
    """

    missing = [column for column in MARKET_NATURAL_KEY if column not in matches.columns]
    if missing:
        raise ValueError(f"Missing synthetic market keys: {', '.join(missing)}")
    if matches.empty:
        raise ValueError("Synthetic market generation requires at least one match")
    keys = (
        matches.loc[:, list(MARKET_NATURAL_KEY)]
        .copy(deep=True)
        .sort_values(list(MARKET_NATURAL_KEY), kind="mergesort")
        .reset_index(drop=True)
    )
    if keys.loc[:, list(MARKET_NATURAL_KEY)].isna().any().any():
        raise ValueError("Synthetic market keys must not contain null values")
    if keys.duplicated(list(MARKET_NATURAL_KEY)).any():
        raise ValueError("Synthetic market keys must be unique")

    rng = np.random.default_rng(seed)
    logits = rng.normal(loc=(0.35, -0.05, -0.20), scale=0.35, size=(len(keys), 3))
    logits = np.clip(logits, -1.0, 1.0)
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    fair_probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    overround = rng.uniform(1.02, 1.08, size=len(keys))
    odds = np.reciprocal(fair_probabilities * overround[:, np.newaxis])

    market = keys.copy(deep=True)
    for index, column in enumerate(CANONICAL_ODDS_COLUMNS):
        market[column] = odds[:, index]
    market = market.loc[:, list(CANONICAL_MARKET_COLUMNS)]

    if mode == "complete":
        return market
    if mode == "missing_rows":
        return market.drop(index=market.index[::7]).reset_index(drop=True)
    if mode == "incomplete":
        changed = market.copy(deep=True)
        changed.loc[0, "ClosingDrawOdds"] = np.nan
        return changed
    if mode == "duplicate":
        return pd.concat([market, market.iloc[[0]]], ignore_index=True)
    if mode == "invalid":
        changed = market.copy(deep=True)
        changed.loc[0, "ClosingHomeOdds"] = 1.0
        return changed
    if mode == "shuffled":
        return market.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    raise ValueError(f"Unsupported synthetic market mode: {mode}")
