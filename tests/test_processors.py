"""Cleaning and resampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.processors.normalize import clean_ohlcv
from app.data.processors.resample import ResampleError, best_source_timeframe, resample_ohlcv
from app.data.schema import empty_frame, validate_schema
from tests.conftest import make_ohlcv

# ------------------------------------------------------------------- cleaning

def test_clean_leaves_good_data_untouched(clean_daily):
    out, report = clean_ohlcv(clean_daily)
    assert report.changed is False
    assert len(out) == len(clean_daily)
    pd.testing.assert_frame_equal(out, clean_daily)


def test_clean_handles_empty_frame():
    out, report = clean_ohlcv(empty_frame())
    assert out.empty
    assert report.rows_in == 0 and report.rows_out == 0


def test_clean_drops_duplicate_timestamps(clean_daily):
    dirty = pd.concat([clean_daily, clean_daily.iloc[[10]]]).sort_index()
    out, report = clean_ohlcv(dirty)
    assert report.duplicates_dropped == 1
    assert not out.index.has_duplicates


def test_clean_duplicate_policy_drop_removes_all_copies(clean_daily):
    dirty = pd.concat([clean_daily, clean_daily.iloc[[10]]]).sort_index()
    out, report = clean_ohlcv(dirty, duplicate_policy="drop")
    assert report.duplicates_dropped == 2
    assert clean_daily.index[10] not in out.index


def test_clean_rejects_bad_duplicate_policy(clean_daily):
    dirty = pd.concat([clean_daily, clean_daily.iloc[[10]]]).sort_index()
    with pytest.raises(ValueError, match="duplicate_policy"):
        clean_ohlcv(dirty, duplicate_policy="whatever")


def test_clean_sorts_unordered_rows(clean_daily):
    shuffled = clean_daily.iloc[::-1]
    out, report = clean_ohlcv(shuffled)
    assert report.reordered is True
    assert out.index.is_monotonic_increasing


def test_clean_drops_rows_with_missing_prices(clean_daily):
    dirty = clean_daily.copy()
    dirty.iloc[5, dirty.columns.get_loc("close")] = np.nan
    out, report = clean_ohlcv(dirty)
    assert report.nan_rows_dropped == 1
    assert len(out) == len(clean_daily) - 1


def test_clean_drops_nonpositive_prices(clean_daily):
    dirty = clean_daily.copy()
    dirty.iloc[3, dirty.columns.get_loc("low")] = -1.0
    dirty.iloc[7, dirty.columns.get_loc("open")] = 0.0
    out, report = clean_ohlcv(dirty)
    assert report.nonpositive_rows_dropped == 2
    assert (out[["open", "high", "low", "close"]] > 0).all().all()


def test_clean_repairs_high_below_body(clean_daily):
    dirty = clean_daily.copy()
    row = 12
    dirty.iloc[row, dirty.columns.get_loc("high")] = dirty.iloc[row]["low"]
    out, report = clean_ohlcv(dirty)
    assert report.ohlc_bounds_repaired == 1
    fixed = out.iloc[row]
    assert fixed["high"] >= max(fixed["open"], fixed["close"])


def test_clean_repairs_low_above_body(clean_daily):
    dirty = clean_daily.copy()
    row = 15
    dirty.iloc[row, dirty.columns.get_loc("low")] = dirty.iloc[row]["high"]
    out, report = clean_ohlcv(dirty)
    assert report.ohlc_bounds_repaired == 1
    fixed = out.iloc[row]
    assert fixed["low"] <= min(fixed["open"], fixed["close"])


def test_clean_can_drop_instead_of_repair(clean_daily):
    dirty = clean_daily.copy()
    dirty.iloc[12, dirty.columns.get_loc("high")] = dirty.iloc[12]["low"]
    out, _ = clean_ohlcv(dirty, repair_ohlc_bounds=False)
    assert len(out) == len(clean_daily) - 1


def test_clean_drops_inverted_high_low(clean_daily):
    """high < low is unrepairable: which value is wrong cannot be known."""
    dirty = clean_daily.copy()
    row = 20
    dirty.iloc[row, dirty.columns.get_loc("high")] = 1.0
    dirty.iloc[row, dirty.columns.get_loc("low")] = 500.0
    dirty.iloc[row, dirty.columns.get_loc("open")] = 1.0
    dirty.iloc[row, dirty.columns.get_loc("close")] = 1.0
    out, report = clean_ohlcv(dirty)
    assert any("unrepairable" in n for n in report.notes)
    assert (out["high"] >= out["low"]).all()


def test_clean_nulls_negative_volume(clean_daily):
    dirty = clean_daily.copy()
    dirty.iloc[4, dirty.columns.get_loc("volume")] = -100.0
    out, report = clean_ohlcv(dirty)
    assert report.negative_volume_nulled == 1
    assert np.isnan(out.iloc[4]["volume"])


def test_clean_never_interpolates_prices(clean_daily):
    """Dropped bars must stay dropped. An invented price is untradeable."""
    dirty = clean_daily.copy()
    dirty.iloc[5, dirty.columns.get_loc("close")] = np.nan
    out, _ = clean_ohlcv(dirty)
    assert clean_daily.index[5] not in out.index


def test_clean_output_passes_strict_validation(clean_daily):
    dirty = clean_daily.copy()
    dirty.iloc[5, dirty.columns.get_loc("close")] = np.nan
    out, _ = clean_ohlcv(dirty)
    assert validate_schema(out) is out


def test_cleaning_report_summary_is_readable(clean_daily):
    dirty = pd.concat([clean_daily, clean_daily.iloc[[10]]]).sort_index()
    _, report = clean_ohlcv(dirty)
    assert "duplicate" in report.summary()


# ----------------------------------------------------------------- resampling

def test_resample_hourly_to_4h_aggregates_correctly(clean_hourly):
    out = resample_ohlcv(clean_hourly, "1H", "4H")
    first_window = clean_hourly.iloc[:4]
    first_bar = out.iloc[0]

    assert first_bar["open"] == first_window["open"].iloc[0]
    assert first_bar["high"] == first_window["high"].max()
    assert first_bar["low"] == first_window["low"].min()
    assert first_bar["close"] == first_window["close"].iloc[-1]
    assert first_bar["volume"] == first_window["volume"].sum()


def test_resample_labels_bars_by_open_time(clean_hourly):
    """A 4H bar must be stamped at its window's start, never its end."""
    out = resample_ohlcv(clean_hourly, "1H", "4H")
    assert out.index[0] == clean_hourly.index[0].floor("4h")
    assert out.index[0] <= clean_hourly.index[0]


def test_resample_refuses_to_upsample(clean_daily):
    with pytest.raises(ResampleError, match="Refusing to upsample"):
        resample_ohlcv(clean_daily, "1D", "1H")


def test_resample_same_timeframe_is_identity(clean_daily):
    out = resample_ohlcv(clean_daily, "1D", "1D")
    pd.testing.assert_frame_equal(out, clean_daily)


def test_resample_empty_frame_stays_empty():
    out = resample_ohlcv(empty_frame(), "1H", "4H")
    assert out.empty


def test_resample_drops_empty_windows(clean_daily):
    """Weekend windows contain no bars and must not appear as NaN rows."""
    out = resample_ohlcv(clean_daily, "1D", "1D")
    assert out[["open", "high", "low", "close"]].notna().all().all()


def test_resample_keeps_missing_volume_as_nan():
    """Summing an all-NaN volume window must not produce a fake 0.0."""
    df = make_ohlcv(periods=96, freq="1h", with_volume=False, weekdays_only=False)
    out = resample_ohlcv(df, "1H", "4H")
    assert out["volume"].isna().all()


def test_resample_output_passes_validation(clean_hourly):
    out = resample_ohlcv(clean_hourly, "1H", "4H")
    assert validate_schema(out) is out


def test_resample_conserves_total_volume(clean_hourly):
    out = resample_ohlcv(clean_hourly, "1H", "4H")
    assert out["volume"].sum() == pytest.approx(clean_hourly["volume"].sum())


def test_resample_high_bounds_all_source_highs(clean_hourly):
    out = resample_ohlcv(clean_hourly, "1H", "4H")
    assert out["high"].max() == pytest.approx(clean_hourly["high"].max())
    assert out["low"].min() == pytest.approx(clean_hourly["low"].min())


# ------------------------------------------------------- source timeframe choice

def test_best_source_prefers_exact_match():
    assert best_source_timeframe("1H", ["1M", "1H", "1D"]) == "1H"


def test_best_source_picks_coarsest_divisor():
    # Building 4H from 1H beats building it from 1M.
    assert best_source_timeframe("4H", ["1M", "5M", "1H"]) == "1H"


def test_best_source_returns_none_when_nothing_fine_enough():
    assert best_source_timeframe("1H", ["1D"]) is None
