"""Technical indicators.

All functions are pure and **causal**: the value at bar ``t`` is computed from
bars ``<= t`` only. Warm-up periods return NaN rather than a partially-computed
value, because a half-formed indicator is not a smaller version of the real
one - it is a different number that will not appear in live trading.

Smoothing conventions follow Wilder for RSI, ATR and ADX (alpha = 1/n), which
is what the original definitions specify and what charting platforms display.
Using a standard EMA instead produces visibly different values and silently
breaks comparisons against any external reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.schema import CLOSE, HIGH, LOW


def _validate_period(period: int, name: str = "period") -> int:
    if not isinstance(period, (int, np.integer)) or period < 1:
        raise ValueError(f"{name} must be a positive integer, got {period!r}")
    return int(period)


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average.

    Returns NaN until ``period`` observations exist.
    """
    period = _validate_period(period)
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, seeded so early values are not misleading.

    ``adjust=False`` gives the recursive form used by charting platforms. The
    first ``period - 1`` values are masked to NaN: pandas would otherwise emit
    a value from a single observation, which is not an EMA of anything.
    """
    period = _validate_period(period)
    out = series.ewm(span=period, adjust=False, min_periods=period).mean()
    return out


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (alpha = 1/n), the basis of RSI, ATR and ADX."""
    period = _validate_period(period)
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def slope(series: pd.Series, period: int = 5, normalize_by: pd.Series | None = None) -> pd.Series:
    """Rate of change of a series per bar, over ``period`` bars.

    Args:
        series: input, typically a moving average.
        period: lookback in bars.
        normalize_by: divide the raw slope by this (usually price or ATR) to get
            a scale-free measure. Gold at 2000 and EURUSD at 1.08 produce raw
            slopes three orders of magnitude apart, so any fixed threshold on an
            un-normalised slope is meaningless across instruments.

    Returns:
        Slope per bar. Positive means rising.
    """
    period = _validate_period(period)
    raw = (series - series.shift(period)) / period
    if normalize_by is None:
        return raw
    denom = normalize_by.replace(0.0, np.nan)
    return raw / denom


def rate_of_change(series: pd.Series, period: int = 10) -> pd.Series:
    """Percentage change over ``period`` bars, as a fraction (0.05 = +5%)."""
    period = _validate_period(period)
    previous = series.shift(period).replace(0.0, np.nan)
    return (series - previous) / previous


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Absolute price change over ``period`` bars, in price units."""
    period = _validate_period(period)
    return series - series.shift(period)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder), bounded [0, 100].

    A period of pure gains yields 100 rather than a division by zero.
    """
    period = _validate_period(period)
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    avg_gain = wilder_smooth(gains, period)
    avg_loss = wilder_smooth(losses, period)

    # avg_loss == 0 means no down moves in the window: RSI is 100 by definition.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
    return out.where(avg_gain.notna() & avg_loss.notna())


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line and histogram.

    Returns:
        DataFrame with columns ``macd``, ``macd_signal``, ``macd_hist``.

    Raises:
        ValueError: if ``fast >= slow``, which would invert the indicator's meaning.
    """
    fast = _validate_period(fast, "fast")
    slow = _validate_period(slow, "slow")
    signal = _validate_period(signal, "signal")
    if fast >= slow:
        raise ValueError(f"fast period ({fast}) must be shorter than slow period ({slow})")

    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        }
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range: the greater of the bar's own range and its gap from the prior close.

    The first bar has no previous close, so its true range is simply high - low.
    """
    high, low, close = df[HIGH], df[LOW], df[CLOSE]
    previous_close = close.shift(1)

    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    tr = ranges.max(axis=1)
    tr.iloc[0] = (high.iloc[0] - low.iloc[0]) if len(df) else np.nan
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder), in price units."""
    return wilder_smooth(true_range(df), _validate_period(period))


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands plus bandwidth and %B.

    Returns:
        DataFrame with ``bb_middle``, ``bb_upper``, ``bb_lower``, ``bb_width``
        (bandwidth as a fraction of the middle band) and ``bb_pct_b`` (position
        within the bands: 0 at the lower band, 1 at the upper).
    """
    period = _validate_period(period)
    if num_std <= 0:
        raise ValueError(f"num_std must be positive, got {num_std}")

    middle = sma(close, period)
    # ddof=0: the population standard deviation, matching the original definition.
    deviation = close.rolling(window=period, min_periods=period).std(ddof=0)

    upper = middle + num_std * deviation
    lower = middle - num_std * deviation
    span = (upper - lower).replace(0.0, np.nan)

    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": (upper - lower) / middle.replace(0.0, np.nan),
            "bb_pct_b": (close - lower) / span,
        }
    )


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI and -DI (Wilder).

    ADX measures trend *strength* regardless of direction; +DI/-DI carry the
    direction. Conventionally ADX > 25 indicates a trending market and < 20 a
    ranging one, but those numbers are folklore, not law - the platform treats
    them as configurable thresholds to be tested, not assumed.

    Returns:
        DataFrame with ``adx``, ``di_plus``, ``di_minus``.
    """
    period = _validate_period(period)
    high, low = df[HIGH], df[LOW]

    up_move = high.diff()
    down_move = -low.diff()

    # Directional movement counts only when one side clearly dominates.
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )

    smoothed_tr = wilder_smooth(true_range(df), period).replace(0.0, np.nan)
    di_plus = 100.0 * wilder_smooth(plus_dm, period) / smoothed_tr
    di_minus = 100.0 * wilder_smooth(minus_dm, period) / smoothed_tr

    di_sum = (di_plus + di_minus).replace(0.0, np.nan)
    dx = 100.0 * (di_plus - di_minus).abs() / di_sum

    return pd.DataFrame(
        {
            "adx": wilder_smooth(dx, period),
            "di_plus": di_plus,
            "di_minus": di_minus,
        }
    )


def rolling_std(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling sample standard deviation."""
    period = _validate_period(period)
    return series.rolling(window=period, min_periods=period).std()


def rolling_percentile(series: pd.Series, period: int = 100) -> pd.Series:
    """Where the current value sits within its own trailing distribution, in [0, 1].

    A volatility percentile of 0.9 means current volatility exceeds 90% of the
    last ``period`` observations. Percentiles travel across instruments and
    epochs in a way that raw ATR values never will: "ATR is 12" means nothing
    on its own, "ATR is at its 95th percentile" means something everywhere.

    The window includes the current bar, which is causal - the bar's own value
    is known at the bar.
    """
    period = _validate_period(period)
    return series.rolling(window=period, min_periods=period).rank(pct=True)


def distance_from(series: pd.Series, reference: pd.Series, normalizer: pd.Series) -> pd.Series:
    """Signed distance from a reference, in units of ``normalizer`` (usually ATR).

    Expressing "price is far above its moving average" in ATR units is what
    makes a threshold comparable between a 2,000-point gold move and a 0.008
    move in EURUSD.
    """
    return (series - reference) / normalizer.replace(0.0, np.nan)


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on the bar where ``fast`` crosses above ``slow``.

    Requires the previous bar to be at or below, so a series that starts above
    does not register a phantom cross on its first valid bar.
    """
    above = fast > slow
    return above & ~above.shift(1, fill_value=False) & fast.shift(1).notna() & slow.shift(1).notna()
