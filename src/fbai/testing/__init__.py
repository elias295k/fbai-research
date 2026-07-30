"""Deterministic test support for Football Outcome Lab."""

from fbai.testing.market import SyntheticMarketMode, make_synthetic_closing_market
from fbai.testing.synthetic import (
    SYNTHETIC_FEATURE_COLUMNS,
    make_synthetic_canonical_matches,
    make_synthetic_fixtures,
    make_synthetic_raw_matches,
)

__all__ = [
    "SYNTHETIC_FEATURE_COLUMNS",
    "SyntheticMarketMode",
    "make_synthetic_canonical_matches",
    "make_synthetic_closing_market",
    "make_synthetic_fixtures",
    "make_synthetic_raw_matches",
]
