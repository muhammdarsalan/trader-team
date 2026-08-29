"""Volume-derived features.

These are computed only where volume means something. For OTC spot FX there is
no consolidated tape: the "volume" a retail feed reports is that broker's tick
count, which says more about the broker's client base than about the market.

Rather than computing plausible-looking numbers on a meaningless series, the
feature engine suppresses this whole family when ``has_reliable_volume`` is
false for the instrument. A missing feature is honest; a fabricated one silently
poisons every model trained on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.schema import VOLUME
from app.features.indicators import rolling_percentile, sma


def relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume as a multiple of its own trailing average.

    2.0 means twice the typical volume of the last ``period`` bars. The average
    excludes the current bar, so a single huge bar does not dilute the baseline
    it is being measured against.
    """
    baseline = sma(volume.shift(1), period)
    return volume / baseline.replace(0.0, np.nan)


def volume_features(
    df: pd.DataFrame,
    period: int = 20,
    spike_threshold: float = 2.0,
) -> pd.DataFrame:
    """Volume moving average, relative volume, spike flag and percentile.

    Args:
        df: canonical OHLCV frame.
        period: lookback for averages and percentile ranking.
        spike_threshold: relative-volume multiple that counts as a spike.

    Returns:
        DataFrame with ``volume_sma``, ``relative_volume``, ``volume_spike``,
        ``volume_percentile``, ``volume_trend``. All NaN (or False) when the
        series carries no usable volume.
    """
    volume = df[VOLUME]
    index = df.index

    usable = volume.notna().any() and (volume.fillna(0) > 0).any()
    if not usable:
        return pd.DataFrame(
            {
                "volume_sma": np.nan,
                "relative_volume": np.nan,
                "volume_spike": False,
                "volume_percentile": np.nan,
                "volume_trend": np.nan,
            },
            index=index,
        )

    rel = relative_volume(volume, period)
    return pd.DataFrame(
        {
            "volume_sma": sma(volume, period),
            "relative_volume": rel,
            "volume_spike": (rel >= spike_threshold).fillna(False),
            "volume_percentile": rolling_percentile(volume, max(period * 5, 100)),
            # Is volume expanding or contracting relative to a slower baseline?
            "volume_trend": sma(volume, period) / sma(volume, period * 3).replace(0.0, np.nan),
        }
    )
