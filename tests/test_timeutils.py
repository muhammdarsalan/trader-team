"""Timeframe normalisation and UTC handling."""

from __future__ import annotations

import pandas as pd
import pytest

from app.utils.timeutils import (
    TIMEFRAMES,
    UnknownTimeframeError,
    is_upsampling,
    normalize_timeframe,
    timeframe_minutes,
    to_utc,
    to_utc_index,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1h", "1H"), ("H1", "1H"), ("60min", "1H"),
        ("4h", "4H"), ("H4", "4H"), ("240m", "4H"),
        ("1d", "1D"), ("daily", "1D"), ("D1", "1D"),
        ("15m", "15M"), ("M15", "15M"), (" 5M ", "5M"),
    ],
)
def test_normalize_timeframe_accepts_common_spellings(raw, expected):
    assert normalize_timeframe(raw).code == expected


def test_normalize_timeframe_rejects_unknown():
    with pytest.raises(UnknownTimeframeError, match="Unsupported timeframe"):
        normalize_timeframe("3H")


def test_normalize_timeframe_is_idempotent():
    tf = normalize_timeframe("4H")
    assert normalize_timeframe(tf) is tf


def test_timeframe_minutes_are_ordered():
    minutes = [timeframe_minutes(code) for code in TIMEFRAMES]
    assert minutes == sorted(minutes)


def test_is_upsampling_detects_direction():
    assert is_upsampling("4H", "1H") is True     # coarse -> fine: fabrication
    assert is_upsampling("1H", "4H") is False    # fine -> coarse: aggregation
    assert is_upsampling("1H", "1H") is False


def test_to_utc_treats_naive_as_utc_not_local():
    # Assuming local time here would silently shift every bar on a non-UTC machine.
    assert to_utc("2024-03-01 12:00").isoformat() == "2024-03-01T12:00:00+00:00"


def test_to_utc_converts_aware_timestamps():
    ny = pd.Timestamp("2024-03-01 09:30", tz="America/New_York")
    assert to_utc(ny).isoformat() == "2024-03-01T14:30:00+00:00"


def test_to_utc_index_localises_naive():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
    out = to_utc_index(idx)
    assert str(out.tz) == "UTC"


def test_to_utc_index_honours_assume_tz():
    idx = pd.DatetimeIndex(["2024-01-01 09:30"])
    out = to_utc_index(idx, assume_tz="America/New_York")
    assert out[0].isoformat() == "2024-01-01T14:30:00+00:00"
