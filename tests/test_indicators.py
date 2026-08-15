"""Technical indicators: correctness, edge cases and causality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import indicators as ind
from tests.conftest import make_ohlcv
from tests.helpers import assert_causal


@pytest.fixture
def prices() -> pd.Series:
    return pd.Series([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 13.0])


# ------------------------------------------------------------------ averages

def test_sma_matches_hand_calculation(prices):
    result = ind.sma(prices, 3)
    assert result.iloc[2] == pytest.approx((10 + 11 + 12) / 3)
    assert result.iloc[3] == pytest.approx((11 + 12 + 11) / 3)


def test_sma_is_nan_during_warmup(prices):
    result = ind.sma(prices, 3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].notna().all()


def test_ema_masks_partial_warmup_values(prices):
    """pandas would emit an 'EMA' from one observation; that is not an EMA."""
    result = ind.ema(prices, 5)
    assert result.iloc[:4].isna().all()
    assert result.notna().iloc[4]


def test_ema_reacts_faster_than_sma():
    rising = pd.Series(np.arange(1.0, 51.0))
    assert ind.ema(rising, 10).iloc[-1] > ind.sma(rising, 10).iloc[-1]


@pytest.mark.parametrize("period", [0, -1, 2.5, "10"])
def test_period_must_be_a_positive_integer(prices, period):
    with pytest.raises(ValueError, match="positive integer"):
        ind.sma(prices, period)


# ----------------------------------------------------------------------- RSI

def test_rsi_is_bounded():
    df = make_ohlcv(periods=300)
    rsi = ind.rsi(df["close"], 14).dropna()
    assert rsi.between(0, 100).all()


def test_rsi_of_uninterrupted_gains_is_100():
    """No down moves means no average loss; the definition gives 100, not a crash."""
    rising = pd.Series(np.arange(1.0, 60.0))
    assert ind.rsi(rising, 14).iloc[-1] == pytest.approx(100.0)


def test_rsi_of_uninterrupted_losses_is_zero():
    falling = pd.Series(np.arange(60.0, 1.0, -1.0))
    assert ind.rsi(falling, 14).iloc[-1] == pytest.approx(0.0)


def test_rsi_of_flat_prices_is_neutral():
    flat = pd.Series([50.0] * 40)
    assert ind.rsi(flat, 14).iloc[-1] == pytest.approx(50.0)


def test_rsi_uses_wilder_smoothing():
    """Wilder's alpha is 1/n, not the 2/(n+1) of a standard EMA."""
    df = make_ohlcv(periods=200, seed=3)
    close = df["close"]
    wilder = ind.rsi(close, 14)

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    standard_ema_rs = (
        gains.ewm(span=14, adjust=False, min_periods=14).mean()
        / losses.ewm(span=14, adjust=False, min_periods=14).mean()
    )
    standard = 100 - 100 / (1 + standard_ema_rs)

    assert not np.isclose(wilder.iloc[-1], standard.iloc[-1])


# ---------------------------------------------------------------------- MACD

def test_macd_returns_three_series():
    df = make_ohlcv(periods=200)
    result = ind.macd(df["close"])
    assert list(result.columns) == ["macd", "macd_signal", "macd_hist"]


def test_macd_histogram_is_the_difference():
    df = make_ohlcv(periods=200)
    result = ind.macd(df["close"]).dropna()
    pd.testing.assert_series_equal(
        result["macd_hist"], result["macd"] - result["macd_signal"], check_names=False
    )


def test_macd_rejects_inverted_periods():
    df = make_ohlcv(periods=100)
    with pytest.raises(ValueError, match="must be shorter"):
        ind.macd(df["close"], fast=26, slow=12)


# ------------------------------------------------------------------ true range

def test_true_range_accounts_for_gaps():
    """A gap makes the true range larger than the bar's own high-low."""
    df = pd.DataFrame(
        {
            "open": [100.0, 120.0], "high": [105.0, 125.0],
            "low": [95.0, 118.0], "close": [100.0, 120.0], "volume": [1.0, 1.0],
        },
        index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
    )
    tr = ind.true_range(df)
    assert tr.iloc[0] == pytest.approx(10.0)      # first bar: high - low
    assert tr.iloc[1] == pytest.approx(25.0)      # gap from prior close of 100


def test_atr_is_positive():
    df = make_ohlcv(periods=200)
    assert (ind.atr(df, 14).dropna() > 0).all()


# ------------------------------------------------------------ Bollinger Bands

def test_bollinger_bands_are_ordered():
    df = make_ohlcv(periods=200)
    bands = ind.bollinger_bands(df["close"], 20, 2.0).dropna()
    assert (bands["bb_upper"] >= bands["bb_middle"]).all()
    assert (bands["bb_middle"] >= bands["bb_lower"]).all()


def test_bollinger_pct_b_locates_price_within_the_bands():
    df = make_ohlcv(periods=200)
    bands = ind.bollinger_bands(df["close"], 20, 2.0)
    combined = bands.join(df["close"]).dropna()

    at_upper = np.isclose(combined["close"], combined["bb_upper"])
    assert np.allclose(combined.loc[at_upper, "bb_pct_b"], 1.0, atol=1e-6)


def test_bollinger_width_is_relative_to_price():
    """Bandwidth must be scale-free, or no threshold works across instruments."""
    df = make_ohlcv(periods=200, start_price=100.0, seed=11)
    scaled = df * 10.0

    width_small = ind.bollinger_bands(df["close"], 20).dropna()["bb_width"]
    width_large = ind.bollinger_bands(scaled["close"], 20).dropna()["bb_width"]
    assert np.allclose(width_small.to_numpy(), width_large.to_numpy(), rtol=1e-9)


def test_bollinger_rejects_non_positive_std():
    df = make_ohlcv(periods=100)
    with pytest.raises(ValueError, match="num_std must be positive"):
        ind.bollinger_bands(df["close"], 20, 0.0)


# ----------------------------------------------------------------------- ADX

def test_adx_is_bounded():
    df = make_ohlcv(periods=400)
    result = ind.adx(df, 14).dropna()
    assert result["adx"].between(0, 100).all()
    assert result["di_plus"].between(0, 100).all()


def test_adx_is_high_in_a_strong_trend():
    """A clean staircase upward should register as strongly trending."""
    n = 200
    close = np.linspace(100, 300, n)
    index = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    df = pd.DataFrame(
        {
            "open": close - 0.5, "high": close + 1.0, "low": close - 1.0,
            "close": close, "volume": np.ones(n),
        },
        index=index,
    )
    assert ind.adx(df, 14)["adx"].iloc[-1] > 40


def test_adx_is_low_in_a_choppy_market():
    n = 200
    base = np.tile([100.0, 101.0], n // 2)
    index = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    df = pd.DataFrame(
        {
            "open": base, "high": base + 0.5, "low": base - 0.5,
            "close": base, "volume": np.ones(n),
        },
        index=index,
    )
    assert ind.adx(df, 14)["adx"].iloc[-1] < 30


# -------------------------------------------------------------- normalisation

def test_slope_normalisation_makes_instruments_comparable():
    """Un-normalised slopes differ by orders of magnitude between instruments."""
    n = 100
    index = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    gold = pd.Series(np.linspace(1800, 1980, n), index=index)      # +10%
    euro = pd.Series(np.linspace(1.08, 1.188, n), index=index)     # +10%

    raw_gold = ind.slope(gold, 5).iloc[-1]
    raw_euro = ind.slope(euro, 5).iloc[-1]
    assert raw_gold > raw_euro * 100  # wildly different scales

    norm_gold = ind.slope(gold, 5, normalize_by=gold).iloc[-1]
    norm_euro = ind.slope(euro, 5, normalize_by=euro).iloc[-1]
    assert norm_gold == pytest.approx(norm_euro, rel=0.05)


def test_rolling_percentile_is_bounded_and_ranks_correctly():
    series = pd.Series(np.arange(100.0))
    result = ind.rolling_percentile(series, 20).dropna()
    assert result.between(0, 1).all()
    # A strictly increasing series is always at the top of its own window.
    assert result.iloc[-1] == pytest.approx(1.0)


def test_rate_of_change_is_a_fraction():
    series = pd.Series([100.0] * 10 + [110.0])
    assert ind.rate_of_change(series, 10).iloc[-1] == pytest.approx(0.10)


# ---------------------------------------------------------------- crossovers

def test_crossover_fires_once_at_the_crossing():
    fast = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    slow = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])
    crosses = ind.crossover(fast, slow)
    assert crosses.sum() == 1
    assert crosses.iloc[3]


def test_crossover_does_not_fire_when_already_above():
    fast = pd.Series([5.0, 6.0, 7.0])
    slow = pd.Series([1.0, 1.0, 1.0])
    assert not ind.crossover(fast, slow).any()


# ------------------------------------------------------------------ causality

@pytest.mark.parametrize(
    ("name", "fn"),
    [
        ("sma", lambda df: ind.sma(df["close"], 20)),
        ("ema", lambda df: ind.ema(df["close"], 21)),
        ("rsi", lambda df: ind.rsi(df["close"], 14)),
        ("macd", lambda df: ind.macd(df["close"])),
        ("atr", lambda df: ind.atr(df, 14)),
        ("adx", lambda df: ind.adx(df, 14)),
        ("bollinger", lambda df: ind.bollinger_bands(df["close"], 20)),
        ("roc", lambda df: ind.rate_of_change(df["close"], 10)),
        ("momentum", lambda df: ind.momentum(df["close"], 10)),
        ("true_range", ind.true_range),
        ("rolling_percentile", lambda df: ind.rolling_percentile(df["close"], 50)),
        ("slope", lambda df: ind.slope(ind.sma(df["close"], 20), 5)),
        ("rolling_std", lambda df: ind.rolling_std(df["close"], 20)),
    ],
)
def test_indicator_is_causal(name, fn):
    """Truncating the input must not change any value that remains."""
    df = make_ohlcv(periods=400, seed=17)
    assert_causal(fn, df, label=name)
