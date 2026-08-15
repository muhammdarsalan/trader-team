"""Market structure: swing confirmation delay, channels, breakouts, trend structure.

The confirmation-delay tests are the most important in the phase. A swing high
is defined by bars on both sides of it; publishing it at the pivot bar would
hand every strategy information the market had not yet produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import structure as struct
from app.features.indicators import atr
from tests.conftest import make_ohlcv
from tests.helpers import assert_causal


def ohlc_from_highs_lows(highs: list[float], lows: list[float]) -> pd.DataFrame:
    """Build a minimal frame with exact highs and lows for pivot testing."""
    n = len(highs)
    index = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    mid = [(h + low) / 2 for h, low in zip(highs, lows, strict=True)]
    df = pd.DataFrame(
        {
            "open": mid, "high": highs, "low": lows, "close": mid,
            "volume": np.ones(n),
        },
        index=index,
    )
    df.index.name = "timestamp"
    return df


# ------------------------------------------------------- confirmation delay

def test_swing_high_is_published_after_confirmation_not_at_the_pivot():
    """The defining test of this module.

    A peak at bar 5 needs 2 lower bars after it. That is knowable at bar 7,
    not at bar 5. Publishing at bar 5 is look-ahead bias.
    """
    highs = [10, 11, 12, 13, 14, 20, 14, 13, 12, 11, 10]
    lows = [h - 2 for h in highs]
    df = ohlc_from_highs_lows(highs, lows)

    swings = struct.swing_points(df, struct.SwingConfig(left=2, right=2))

    assert pd.isna(swings["swing_high"].iloc[5]), "pivot must NOT be published at its own bar"
    assert pd.isna(swings["swing_high"].iloc[6]), "still unconfirmed one bar later"
    assert swings["swing_high"].iloc[7] == 20.0, "published at pivot + right"


def test_last_swing_high_is_unknown_before_confirmation():
    highs = [10, 11, 12, 13, 14, 20, 14, 13, 12, 11, 10]
    lows = [h - 2 for h in highs]
    df = ohlc_from_highs_lows(highs, lows)

    swings = struct.swing_points(df, struct.SwingConfig(left=2, right=2))

    assert pd.isna(swings["last_swing_high"].iloc[6])
    assert swings["last_swing_high"].iloc[7] == 20.0
    assert swings["last_swing_high"].iloc[8] == 20.0  # forward-filled thereafter


def test_swing_age_counts_from_the_pivot_not_from_publication():
    """Distance-to-structure logic depends on the true age of the level."""
    highs = [10, 11, 12, 13, 14, 20, 14, 13, 12, 11, 10]
    lows = [h - 2 for h in highs]
    df = ohlc_from_highs_lows(highs, lows)

    swings = struct.swing_points(df, struct.SwingConfig(left=2, right=2))

    assert swings["last_swing_high_age"].iloc[7] == 2   # pivot at 5, now at 7
    assert swings["last_swing_high_age"].iloc[9] == 4


def test_swing_low_is_published_after_confirmation():
    lows = [20, 19, 18, 17, 16, 10, 16, 17, 18, 19, 20]
    highs = [low + 2 for low in lows]
    df = ohlc_from_highs_lows(highs, lows)

    swings = struct.swing_points(df, struct.SwingConfig(left=2, right=2))

    assert pd.isna(swings["swing_low"].iloc[5])
    assert swings["swing_low"].iloc[7] == 10.0


def test_confirmation_delay_scales_with_right_window():
    highs = [10, 11, 12, 13, 14, 20, 14, 13, 12, 11, 10, 9, 8]
    lows = [h - 2 for h in highs]
    df = ohlc_from_highs_lows(highs, lows)

    for right in (1, 2, 3):
        swings = struct.swing_points(df, struct.SwingConfig(left=2, right=right))
        published_at = swings["swing_high"].first_valid_index()
        assert df.index.get_loc(published_at) == 5 + right


def test_pivot_near_the_end_is_never_published():
    """Without enough bars after it, a peak is not yet a confirmed pivot."""
    highs = [10, 11, 12, 13, 20, 14]
    lows = [h - 2 for h in highs]
    df = ohlc_from_highs_lows(highs, lows)

    swings = struct.swing_points(df, struct.SwingConfig(left=2, right=3))
    assert swings["swing_high"].isna().all()


def test_flat_shelf_does_not_produce_a_pivot_at_every_bar():
    highs = [10.0] * 10
    lows = [8.0] * 10
    df = ohlc_from_highs_lows(highs, lows)

    swings = struct.swing_points(df, struct.SwingConfig(left=2, right=2))
    assert swings["swing_high"].isna().all()


def test_previous_swing_tracks_the_pivot_before_last():
    highs = [10, 11, 15, 11, 10, 11, 18, 11, 10, 11, 12, 11, 10]
    lows = [h - 3 for h in highs]
    df = ohlc_from_highs_lows(highs, lows)

    swings = struct.swing_points(df, struct.SwingConfig(left=2, right=2))
    published = swings["swing_high"].dropna()
    assert len(published) >= 2

    second_bar = published.index[1]
    assert swings.loc[second_bar, "last_swing_high"] == published.iloc[1]
    assert swings.loc[second_bar, "prev_swing_high"] == published.iloc[0]


def test_swing_config_rejects_zero_windows():
    with pytest.raises(ValueError, match="must be >= 1"):
        struct.SwingConfig(left=0, right=2)


def test_short_series_returns_empty_structure():
    df = make_ohlcv(periods=3)
    swings = struct.swing_points(df, struct.SwingConfig(left=3, right=3))
    assert len(swings) == 3
    assert swings["swing_high"].isna().all()


# --------------------------------------------------------- Donchian channel

def test_donchian_excludes_the_current_bar_by_default():
    """Otherwise a bar can never exceed a maximum that contains its own high."""
    highs = [10, 11, 12, 13, 20]
    lows = [h - 2 for h in highs]
    df = ohlc_from_highs_lows(highs, lows)

    channel = struct.donchian_channel(df, period=4, exclude_current=True)
    # At the final bar the channel reflects bars 0-3, so the high of 20 breaks it.
    assert channel["donchian_upper"].iloc[4] == 13.0


def test_donchian_including_current_bar_cannot_be_broken():
    highs = [10, 11, 12, 13, 20]
    lows = [h - 2 for h in highs]
    df = ohlc_from_highs_lows(highs, lows)

    channel = struct.donchian_channel(df, period=4, exclude_current=False)
    assert channel["donchian_upper"].iloc[4] == 20.0  # the bar's own high


def test_donchian_bounds_are_ordered():
    df = make_ohlcv(periods=200)
    channel = struct.donchian_channel(df, 20).dropna()
    assert (channel["donchian_upper"] >= channel["donchian_lower"]).all()


# ------------------------------------------------------------------ breakout

def test_breakout_up_fires_when_close_exceeds_the_channel():
    highs = [10.0] * 20 + [20.0]
    lows = [8.0] * 20 + [10.0]
    df = ohlc_from_highs_lows(highs, lows)
    df.loc[df.index[-1], "close"] = 19.0

    result = struct.breakout_levels(df, period=10)
    assert bool(result["breakout_up"].iloc[-1])
    assert not bool(result["breakout_down"].iloc[-1])


def test_breakout_requires_clearing_the_atr_buffer():
    highs = [10.0] * 20 + [10.4]
    lows = [8.0] * 20 + [9.0]
    df = ohlc_from_highs_lows(highs, lows)
    df.loc[df.index[-1], "close"] = 10.2

    atr_series = atr(df, 5)
    without = struct.breakout_levels(df, period=10, atr_series=atr_series, buffer_atr=0.0)
    with_buffer = struct.breakout_levels(df, period=10, atr_series=atr_series, buffer_atr=2.0)

    assert bool(without["breakout_up"].iloc[-1])
    assert not bool(with_buffer["breakout_up"].iloc[-1])


def test_breakout_buffer_requires_atr():
    df = make_ohlcv(periods=100)
    with pytest.raises(ValueError, match="requires atr_series"):
        struct.breakout_levels(df, period=20, buffer_atr=0.5)


def test_no_breakout_inside_the_range():
    df = make_ohlcv(periods=200, seed=5)
    result = struct.breakout_levels(df, period=20)
    # Not every bar can be a breakout; most must be inside the channel.
    assert result["breakout_up"].mean() < 0.2


# ----------------------------------------------------------- trend structure

def test_uptrend_needs_both_higher_highs_and_higher_lows():
    swings = pd.DataFrame(
        {
            "last_swing_high": [12.0], "prev_swing_high": [10.0],
            "last_swing_low": [8.0], "prev_swing_low": [6.0],
        }
    )
    assert struct.trend_structure(swings)["structure"].iloc[0] == "UPTREND"


def test_downtrend_needs_both_lower_highs_and_lower_lows():
    swings = pd.DataFrame(
        {
            "last_swing_high": [8.0], "prev_swing_high": [10.0],
            "last_swing_low": [4.0], "prev_swing_low": [6.0],
        }
    )
    assert struct.trend_structure(swings)["structure"].iloc[0] == "DOWNTREND"


def test_mixed_structure_is_unclear():
    """Higher highs with lower lows is a broadening range, not a trend."""
    swings = pd.DataFrame(
        {
            "last_swing_high": [12.0], "prev_swing_high": [10.0],
            "last_swing_low": [4.0], "prev_swing_low": [6.0],
        }
    )
    assert struct.trend_structure(swings)["structure"].iloc[0] == "UNCLEAR"


def test_structure_is_unclear_without_two_confirmed_pivots():
    swings = pd.DataFrame(
        {
            "last_swing_high": [12.0], "prev_swing_high": [np.nan],
            "last_swing_low": [8.0], "prev_swing_low": [6.0],
        }
    )
    assert struct.trend_structure(swings)["structure"].iloc[0] == "UNCLEAR"


# ---------------------------------------------------- support/resistance distance

def test_distance_is_measured_in_atr_units():
    df = make_ohlcv(periods=200)
    swings = struct.swing_points(df)
    distances = struct.support_resistance_distance(df, swings, atr(df, 14))
    assert "dist_to_resistance_atr" in distances
    assert "dist_to_support_atr" in distances


def test_negative_distance_means_price_passed_through_the_level():
    df = make_ohlcv(periods=100)
    swings = pd.DataFrame(
        {"last_swing_high": 50.0, "last_swing_low": 10.0}, index=df.index
    )
    df = df.copy()
    df["close"] = 60.0  # above the resistance level
    distances = struct.support_resistance_distance(df, swings, atr(df, 14))
    assert (distances["dist_to_resistance_atr"].dropna() < 0).all()


# ------------------------------------------------------------------ causality

@pytest.mark.parametrize(
    ("name", "fn"),
    [
        ("swing_points", lambda df: struct.swing_points(df).drop(columns=["structure"], errors="ignore")),
        ("donchian", lambda df: struct.donchian_channel(df, 20)),
        ("breakout_levels", lambda df: struct.breakout_levels(df, 20)),
        (
            "trend_structure",
            lambda df: struct.trend_structure(struct.swing_points(df)),
        ),
        (
            "sr_distance",
            lambda df: struct.support_resistance_distance(df, struct.swing_points(df), atr(df, 14)),
        ),
    ],
)
def test_structure_feature_is_causal(name, fn):
    df = make_ohlcv(periods=400, seed=23)
    assert_causal(fn, df, label=name)
