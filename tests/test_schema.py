"""The canonical OHLCV contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.schema import (
    OHLCV_COLUMNS,
    MarketData,
    SchemaError,
    coerce_schema,
    empty_frame,
    validate_schema,
)
from tests.conftest import make_ohlcv

# --------------------------------------------------------------- coerce_schema

def test_coerce_renames_vendor_columns():
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "Open": [1.0, 2.0], "High": [2.0, 3.0],
            "Low": [0.5, 1.5], "Close": [1.5, 2.5], "Vol": [10, 20],
        }
    )
    out = coerce_schema(raw)
    assert list(out.columns) == list(OHLCV_COLUMNS)
    assert str(out.index.tz) == "UTC"
    assert out.index.name == "timestamp"


def test_coerce_flattens_multiindex_columns():
    # yfinance-style (field, ticker) headers.
    cols = pd.MultiIndex.from_tuples(
        [("Open", "GC=F"), ("High", "GC=F"), ("Low", "GC=F"), ("Close", "GC=F"), ("Volume", "GC=F")]
    )
    raw = pd.DataFrame([[1.0, 2.0, 0.5, 1.5, 10]], index=pd.DatetimeIndex(["2024-01-01"]), columns=cols)
    out = coerce_schema(raw)
    assert list(out.columns) == list(OHLCV_COLUMNS)


def test_coerce_sorts_out_of_order_rows():
    raw = pd.DataFrame(
        {
            "time": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "open": [3.0, 1.0, 2.0], "high": [3, 1, 2], "low": [3, 1, 2], "close": [3, 1, 2],
        }
    )
    out = coerce_schema(raw)
    assert out.index.is_monotonic_increasing


def test_coerce_adds_missing_volume_as_nan():
    raw = pd.DataFrame(
        {"date": ["2024-01-01"], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}
    )
    out = coerce_schema(raw)
    assert out["volume"].isna().all()


def test_coerce_rejects_missing_required_column():
    raw = pd.DataFrame({"date": ["2024-01-01"], "open": [1.0], "high": [1.0], "low": [1.0]})
    with pytest.raises(SchemaError, match="Missing required column"):
        coerce_schema(raw)


def test_coerce_rejects_frame_without_timestamp():
    raw = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]})
    with pytest.raises(SchemaError, match="No timestamp column"):
        coerce_schema(raw)


def test_coerce_drops_unparseable_timestamps():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "not-a-date", "2024-01-02"],
            "open": [1.0, 2.0, 3.0], "high": [1, 2, 3], "low": [1, 2, 3], "close": [1, 2, 3],
        }
    )
    out = coerce_schema(raw)
    assert len(out) == 2


def test_coerce_localises_naive_using_assume_tz():
    raw = pd.DataFrame(
        {"date": ["2024-01-01 09:30"], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}
    )
    out = coerce_schema(raw, assume_tz="America/New_York")
    assert out.index[0].isoformat() == "2024-01-01T14:30:00+00:00"


def test_coerce_prefers_close_over_adj_close_on_collision():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01"], "open": [1.0], "high": [1.0], "low": [1.0],
            "close": [10.0], "adj close": [99.0],
        }
    )
    out = coerce_schema(raw)
    assert out["close"].iloc[0] == 10.0


# -------------------------------------------------------------- validate_schema

def test_validate_accepts_clean_frame(clean_daily):
    assert validate_schema(clean_daily) is clean_daily


def test_validate_rejects_naive_index(clean_daily):
    bad = clean_daily.copy()
    bad.index = bad.index.tz_localize(None)
    with pytest.raises(SchemaError, match="timezone-aware"):
        validate_schema(bad)


def test_validate_rejects_non_utc_index(clean_daily):
    bad = clean_daily.tz_convert("America/New_York")
    with pytest.raises(SchemaError, match="must be UTC"):
        validate_schema(bad)


def test_validate_rejects_duplicate_index(clean_daily):
    bad = pd.concat([clean_daily, clean_daily.iloc[[0]]]).sort_index()
    with pytest.raises(SchemaError, match="duplicate timestamps"):
        validate_schema(bad)


def test_validate_rejects_nan_in_ohlc(clean_daily):
    bad = clean_daily.copy()
    bad.iloc[5, bad.columns.get_loc("close")] = np.nan
    with pytest.raises(SchemaError, match="NaN values"):
        validate_schema(bad)


def test_validate_allows_nan_volume(clean_daily):
    ok = clean_daily.copy()
    ok["volume"] = np.nan
    assert validate_schema(ok) is ok


def test_validate_non_strict_skips_row_level_checks(clean_daily):
    loose = clean_daily.copy()
    loose.iloc[5, loose.columns.get_loc("close")] = np.nan
    assert validate_schema(loose, strict=False) is loose


# ------------------------------------------------------------------ empty_frame

def test_empty_frame_satisfies_the_contract():
    df = empty_frame()
    assert validate_schema(df) is df
    assert df.empty


# ------------------------------------------------------------------ MarketData

def test_marketdata_normalises_symbol_and_timeframe(clean_daily):
    md = MarketData(symbol=" xauusd ", timeframe="1d", df=clean_daily)
    assert md.symbol == "XAUUSD"
    assert md.timeframe.code == "1D"


def test_marketdata_reports_range(clean_daily):
    md = MarketData(symbol="X", timeframe="1D", df=clean_daily)
    assert md.start == clean_daily.index[0]
    assert md.end == clean_daily.index[-1]
    assert len(md) == len(clean_daily)


def test_marketdata_has_volume_is_false_when_all_nan():
    df = make_ohlcv(periods=50, with_volume=False)
    assert MarketData(symbol="X", timeframe="1D", df=df).has_volume is False


def test_marketdata_has_volume_is_true_when_present(clean_daily):
    assert MarketData(symbol="X", timeframe="1D", df=clean_daily).has_volume is True


def test_marketdata_empty_is_handled():
    md = MarketData(symbol="X", timeframe="1D", df=empty_frame())
    assert md.is_empty
    assert md.start is None
    assert "empty" in md.describe()


def test_marketdata_slice_is_inclusive(clean_daily):
    md = MarketData(symbol="X", timeframe="1D", df=clean_daily)
    start, end = clean_daily.index[10], clean_daily.index[20]
    out = md.slice(start, end)
    assert out.start == start
    assert out.end == end
    assert len(out) == 11


def test_marketdata_is_immutable(clean_daily):
    md = MarketData(symbol="X", timeframe="1D", df=clean_daily)
    with pytest.raises((AttributeError, TypeError)):
        md.symbol = "Y"  # type: ignore[misc]
