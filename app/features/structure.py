"""Market structure: swing points, support/resistance, breakouts, trend structure.

**This module is where look-ahead bias enters most trading platforms.**

A swing high at bar ``i`` is defined by bars on *both* sides of it: bar ``i`` is
a swing high only if the ``right`` bars after it are lower. That fact is not
knowable at bar ``i``. It becomes knowable at bar ``i + right``.

The naive implementation marks the pivot at bar ``i`` and moves on. Every
strategy reading it then trades on a pivot the market had not yet confirmed,
and the backtest reports superb results that evaporate live.

Everything in this module publishes confirmed structure at the bar where
confirmation actually arrives. The pivot's *price* and its original *timestamp*
are both preserved - only the moment it becomes visible is corrected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.data.schema import CLOSE, HIGH, LOW


@dataclass(frozen=True)
class SwingConfig:
    """Pivot detection parameters.

    Attributes:
        left: bars to the left that must be lower (for a high).
        right: bars to the right that must be lower. Also the confirmation
            delay: a pivot is published ``right`` bars after it occurred.
    """

    left: int = 3
    right: int = 3

    def __post_init__(self) -> None:
        if self.left < 1 or self.right < 1:
            raise ValueError(f"left and right must be >= 1, got left={self.left}, right={self.right}")


def swing_points(df: pd.DataFrame, config: SwingConfig | None = None) -> pd.DataFrame:
    """Confirmed swing highs and lows, published when confirmation arrives.

    Args:
        df: canonical OHLCV frame.
        config: pivot geometry. Defaults to 3 bars either side.

    Returns:
        DataFrame indexed like ``df`` with columns:

        - ``swing_high`` / ``swing_low``: pivot price, on the confirmation bar
          only (NaN elsewhere).
        - ``last_swing_high`` / ``last_swing_low``: most recent confirmed pivot
          price, forward-filled. **This is what strategies should use.**
        - ``last_swing_high_age`` / ``last_swing_low_age``: bars since that
          pivot actually occurred, so a strategy can ignore stale structure.
        - ``prev_swing_high`` / ``prev_swing_low``: the pivot before the last
          one, for higher-high / lower-low comparisons.

    A pivot detected at bar ``i`` appears at bar ``i + right``. The reported
    ``age`` counts from ``i``, not from publication, so distance-to-structure
    logic stays truthful.
    """
    cfg = config or SwingConfig()
    n = len(df)

    out = pd.DataFrame(
        {
            "swing_high": np.nan,
            "swing_low": np.nan,
            "last_swing_high": np.nan,
            "last_swing_low": np.nan,
            "last_swing_high_age": np.nan,
            "last_swing_low_age": np.nan,
            "prev_swing_high": np.nan,
            "prev_swing_low": np.nan,
        },
        index=df.index,
    )
    if n < cfg.left + cfg.right + 1:
        return out

    high, low = df[HIGH].to_numpy(), df[LOW].to_numpy()

    # A pivot high must be strictly greater than the left window (so a flat
    # shelf does not produce a pivot at every bar) and greater than or equal to
    # the right window. Requiring strictness on both sides misses pivots that
    # end in a doji, which are perfectly real.
    is_pivot_high = np.zeros(n, dtype=bool)
    is_pivot_low = np.zeros(n, dtype=bool)

    for i in range(cfg.left, n - cfg.right):
        window_left_h = high[i - cfg.left : i]
        window_right_h = high[i + 1 : i + 1 + cfg.right]
        if high[i] > window_left_h.max() and high[i] >= window_right_h.max():
            is_pivot_high[i] = True

        window_left_l = low[i - cfg.left : i]
        window_right_l = low[i + 1 : i + 1 + cfg.right]
        if low[i] < window_left_l.min() and low[i] <= window_right_l.min():
            is_pivot_low[i] = True

    # --- publish at the confirmation bar, not at the pivot -------------------
    for kind, is_pivot, price_source in (
        ("high", is_pivot_high, high),
        ("low", is_pivot_low, low),
    ):
        published_price = np.full(n, np.nan)
        pivot_origin = np.full(n, np.nan)

        pivot_indices = np.flatnonzero(is_pivot)
        for idx in pivot_indices:
            confirm_at = idx + cfg.right
            if confirm_at < n:
                published_price[confirm_at] = price_source[idx]
                pivot_origin[confirm_at] = idx

        published = pd.Series(published_price, index=df.index)
        origin = pd.Series(pivot_origin, index=df.index)

        out[f"swing_{kind}"] = published
        last = published.ffill()
        out[f"last_swing_{kind}"] = last

        last_origin = origin.ffill()
        bar_number = pd.Series(np.arange(n, dtype="float64"), index=df.index)
        out[f"last_swing_{kind}_age"] = bar_number - last_origin

        # The pivot before the current one: shift the sparse series, then fill.
        out[f"prev_swing_{kind}"] = published.dropna().shift(1).reindex(df.index).ffill()

    return out


def donchian_channel(df: pd.DataFrame, period: int = 20, exclude_current: bool = True) -> pd.DataFrame:
    """Highest high and lowest low over a trailing window.

    Args:
        df: canonical OHLCV frame.
        period: lookback in bars.
        exclude_current: shift the window back one bar so the current bar is not
            part of its own channel. **This matters.** Without it, "close breaks
            above the 20-bar high" can never be true at the moment of the break,
            because the current bar's own high is inside the maximum it must
            exceed. Every breakout signal would be delayed by one bar or lost.

    Returns:
        DataFrame with ``donchian_upper``, ``donchian_lower``, ``donchian_mid``.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    highs, lows = df[HIGH], df[LOW]
    if exclude_current:
        highs, lows = highs.shift(1), lows.shift(1)

    upper = highs.rolling(window=period, min_periods=period).max()
    lower = lows.rolling(window=period, min_periods=period).min()

    return pd.DataFrame(
        {
            "donchian_upper": upper,
            "donchian_lower": lower,
            "donchian_mid": (upper + lower) / 2.0,
        }
    )


def breakout_levels(
    df: pd.DataFrame,
    period: int = 20,
    atr_series: pd.Series | None = None,
    buffer_atr: float = 0.0,
    freshness_window: int = 3,
) -> pd.DataFrame:
    """Breakout detection against a trailing Donchian channel.

    Args:
        df: canonical OHLCV frame.
        period: channel lookback.
        atr_series: ATR, required when ``buffer_atr`` is non-zero.
        buffer_atr: require the close to clear the level by this many ATRs.
            A small buffer filters marginal pokes through a level that are
            noise rather than a break; it is a parameter to be tested, not a
            truth.
        freshness_window: how many prior bars must be free of a break for the
            current one to count as *fresh*.

    Returns:
        DataFrame with ``breakout_up``, ``breakout_down`` (bool),
        ``breakout_up_fresh``, ``breakout_down_fresh`` (bool),
        ``consolidation_ratio`` (channel width relative to its own median - low
        values mean a tight range), and the underlying channel columns.

    On the distinction between ``breakout_up`` and ``breakout_up_fresh``: in a
    sustained trend every bar sets a new N-bar high, so ``breakout_up`` is true
    continuously for hundreds of bars. That is a trend, not a breakout. A
    breakout is a *transition* out of a range, and only the fresh variant
    captures it. Anything deciding "is this a breakout right now" - the regime
    detector, the breakout strategy - must use the fresh columns.
    """
    channel = donchian_channel(df, period=period, exclude_current=True)
    close = df[CLOSE]

    buffer = 0.0
    if buffer_atr:
        if atr_series is None:
            raise ValueError("buffer_atr requires atr_series")
        buffer = atr_series * buffer_atr

    upper, lower = channel["donchian_upper"], channel["donchian_lower"]

    out = channel.copy()
    breakout_up = (close > upper + buffer).fillna(False)
    breakout_down = (close < lower - buffer).fillna(False)
    out["breakout_up"] = breakout_up
    out["breakout_down"] = breakout_down

    # A break is "fresh" only if the channel was intact over the preceding
    # window. Shifting first keeps this causal: the current bar is judged
    # against bars strictly before it.
    if freshness_window >= 1:
        recent_up = (
            breakout_up.shift(1).fillna(False).rolling(freshness_window, min_periods=1).max() > 0
        )
        recent_down = (
            breakout_down.shift(1).fillna(False).rolling(freshness_window, min_periods=1).max() > 0
        )
        out["breakout_up_fresh"] = breakout_up & ~recent_up
        out["breakout_down_fresh"] = breakout_down & ~recent_down
    else:
        out["breakout_up_fresh"] = breakout_up
        out["breakout_down_fresh"] = breakout_down

    width = (upper - lower) / close.replace(0.0, np.nan)
    median_width = width.rolling(window=max(period * 3, 20), min_periods=period).median()
    out["channel_width"] = width
    out["consolidation_ratio"] = width / median_width.replace(0.0, np.nan)

    return out


def trend_structure(swings: pd.DataFrame) -> pd.DataFrame:
    """Higher-highs / lower-lows classification from confirmed swings.

    Args:
        swings: output of :func:`swing_points`.

    Returns:
        DataFrame with booleans ``higher_high``, ``lower_high``, ``higher_low``,
        ``lower_low``, and ``structure`` in {UPTREND, DOWNTREND, UNCLEAR}.

    Classic reading: higher highs *and* higher lows is an uptrend. Anything
    mixed is UNCLEAR, which is a legitimate and common answer - forcing a
    direction on sideways structure is how strategies end up trading noise.
    """
    last_high, prev_high = swings["last_swing_high"], swings["prev_swing_high"]
    last_low, prev_low = swings["last_swing_low"], swings["prev_swing_low"]

    higher_high = (last_high > prev_high).fillna(False)
    lower_high = (last_high < prev_high).fillna(False)
    higher_low = (last_low > prev_low).fillna(False)
    lower_low = (last_low < prev_low).fillna(False)

    structure = pd.Series("UNCLEAR", index=swings.index, dtype="object")
    structure[higher_high & higher_low] = "UPTREND"
    structure[lower_high & lower_low] = "DOWNTREND"

    # Without two confirmed pivots on each side there is nothing to compare.
    incomplete = prev_high.isna() | prev_low.isna()
    structure[incomplete] = "UNCLEAR"

    return pd.DataFrame(
        {
            "higher_high": higher_high,
            "lower_high": lower_high,
            "higher_low": higher_low,
            "lower_low": lower_low,
            "structure": structure,
        }
    )


def support_resistance_distance(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    atr_series: pd.Series,
) -> pd.DataFrame:
    """How far price sits from the nearest confirmed structure, in ATR units.

    ATR units rather than price units, so a single threshold is meaningful on
    both gold and EURUSD.

    Returns:
        DataFrame with ``dist_to_resistance_atr`` and ``dist_to_support_atr``.
        Positive means the level is above (resistance) or below (support) the
        current close. Negative means price has passed through the level, which
        is itself informative.
    """
    close = df[CLOSE]
    safe_atr = atr_series.replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "dist_to_resistance_atr": (swings["last_swing_high"] - close) / safe_atr,
            "dist_to_support_atr": (close - swings["last_swing_low"]) / safe_atr,
        }
    )
