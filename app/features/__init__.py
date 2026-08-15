"""Feature engineering.

Layout:
    indicators.py  pure, causal technical indicators
    structure.py   swing points, support/resistance, breakouts, trend structure
    volume.py      volume-derived features (suppressed where volume is meaningless)
    engine.py      FeatureEngine: MarketData -> FeatureSet

**Every function here is causal.** A value at bar ``t`` uses only bars
``<= t``. No rolling window is centred, no series is shifted backwards, and
anything requiring future confirmation (swing pivots) is published only at the
bar where confirmation actually arrives.

This is not a stylistic preference. A single non-causal feature makes every
backtest downstream of it worthless while looking better than ever.
"""

from app.features.engine import FeatureEngine, FeatureSet
from app.features.indicators import (
    adx,
    atr,
    bollinger_bands,
    ema,
    macd,
    momentum,
    rate_of_change,
    rolling_percentile,
    rsi,
    slope,
    sma,
    true_range,
)
from app.features.structure import (
    breakout_levels,
    donchian_channel,
    swing_points,
    trend_structure,
)
from app.features.volume import relative_volume, volume_features

__all__ = [
    "FeatureEngine",
    "FeatureSet",
    "adx",
    "atr",
    "bollinger_bands",
    "breakout_levels",
    "donchian_channel",
    "ema",
    "macd",
    "momentum",
    "rate_of_change",
    "relative_volume",
    "rolling_percentile",
    "rsi",
    "slope",
    "sma",
    "swing_points",
    "trend_structure",
    "true_range",
    "volume_features",
]
