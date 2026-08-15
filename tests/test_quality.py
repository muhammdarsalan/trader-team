"""The data-quality engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.models import AssetConfig, QualityThresholds
from app.data.schema import MarketData, empty_frame
from app.data.validators.quality import (
    DataQualityEngine,
    DataQualityError,
    QualityStatus,
)
from tests.conftest import make_ohlcv


def wrap(df: pd.DataFrame, symbol: str = "TEST", timeframe: str = "1D") -> MarketData:
    return MarketData(symbol=symbol, timeframe=timeframe, df=df, provider="synthetic")


@pytest.fixture
def engine() -> DataQualityEngine:
    return DataQualityEngine(QualityThresholds())


def _asset(**overrides) -> AssetConfig:
    base = {
        "symbol": "TEST", "name": "Test", "asset_class": "FX", "quote_currency": "USD",
        "tick_size": 0.0001, "typical_spread": 0.0002,
    }
    return AssetConfig(**{**base, **overrides})


# ------------------------------------------------------------------ happy path

def test_clean_data_passes(engine, clean_daily):
    report = engine.validate(wrap(clean_daily))
    # Stale-data warnings are expected: the fixture is historical.
    assert report.status is not QualityStatus.FAIL
    assert report.stats["invalid_ohlc"] == 0
    assert report.stats["duplicates"] == 0


def test_report_renders_all_key_sections(engine, clean_daily):
    text = engine.validate(wrap(clean_daily)).render()
    for expected in ("DATA QUALITY REPORT", "Symbol:", "Timeframe:", "Rows:", "Status:"):
        assert expected in text


def test_report_serialises(engine, clean_daily):
    payload = engine.validate(wrap(clean_daily)).to_dict()
    assert payload["symbol"] == "TEST"
    assert payload["status"] in {"PASS", "WARNING", "FAIL"}
    assert isinstance(payload["issues"], list)


# ---------------------------------------------------------------- fatal defects

def test_empty_dataset_fails(engine):
    report = engine.validate(wrap(empty_frame()))
    assert report.status is QualityStatus.FAIL
    assert any(i.code == "empty_dataset" for i in report.issues)


def test_insufficient_history_fails(engine):
    report = engine.validate(wrap(make_ohlcv(periods=10)))
    assert report.status is QualityStatus.FAIL
    assert any(i.code == "insufficient_history" for i in report.issues)


def test_short_history_warns(engine):
    report = engine.validate(wrap(make_ohlcv(periods=60)))
    assert any(i.code == "short_history" for i in report.issues)


def test_negative_prices_fail(engine, clean_daily):
    dirty = clean_daily.copy()
    dirty.iloc[10, dirty.columns.get_loc("low")] = -5.0
    report = engine.validate(wrap(dirty))
    assert report.status is QualityStatus.FAIL
    assert any(i.code == "nonpositive_prices" for i in report.issues)


def test_unordered_timestamps_fail(engine, clean_daily):
    report = engine.validate(wrap(clean_daily.iloc[::-1]))
    assert report.status is QualityStatus.FAIL
    assert any(i.code == "unordered_timestamps" for i in report.issues)


def test_many_duplicates_fail(engine, clean_daily):
    dirty = pd.concat([clean_daily, clean_daily.iloc[:20]]).sort_index()
    report = engine.validate(wrap(dirty))
    assert report.status is QualityStatus.FAIL
    assert report.stats["duplicates"] == 20


def test_single_duplicate_warns(engine, clean_daily):
    dirty = pd.concat([clean_daily, clean_daily.iloc[[5]]]).sort_index()
    report = engine.validate(wrap(dirty))
    assert any(i.code == "duplicate_timestamps" and i.severity is QualityStatus.WARNING
               for i in report.issues)


# ------------------------------------------------------------- OHLC consistency

def test_inverted_high_low_is_detected(engine, clean_daily):
    dirty = clean_daily.copy()
    dirty.iloc[10, dirty.columns.get_loc("high")] = 1.0
    dirty.iloc[10, dirty.columns.get_loc("low")] = 999.0
    report = engine.validate(wrap(dirty))
    assert report.stats["invalid_ohlc"] >= 1
    assert any(i.code == "invalid_ohlc" for i in report.issues)


def test_high_below_body_is_detected(engine, clean_daily):
    dirty = clean_daily.copy()
    dirty.iloc[10, dirty.columns.get_loc("high")] = dirty.iloc[10]["low"]
    report = engine.validate(wrap(dirty))
    assert report.stats["invalid_ohlc"] == 1


def test_invalid_ohlc_message_explains_which_rule_broke(engine, clean_daily):
    dirty = clean_daily.copy()
    dirty.iloc[10, dirty.columns.get_loc("high")] = dirty.iloc[10]["low"]
    report = engine.validate(wrap(dirty))
    issue = next(i for i in report.issues if i.code == "invalid_ohlc")
    assert "high below open/close" in issue.message


# ------------------------------------------------------------------- price jumps

def test_bad_print_is_flagged(engine, clean_daily):
    """A 10x spike in one bar is a bad print, and MAD-based detection must catch it."""
    dirty = clean_daily.copy()
    row = 100
    for col in ("open", "high", "low", "close"):
        dirty.iloc[row, dirty.columns.get_loc(col)] = dirty.iloc[row][col] * 10
    report = engine.validate(wrap(dirty))
    assert report.stats["price_jumps"] >= 1


def test_normal_volatility_is_not_flagged(engine, clean_daily):
    report = engine.validate(wrap(clean_daily))
    assert report.stats.get("price_jumps", 0) == 0


def test_jump_detection_resists_masking_by_the_outlier_itself(engine):
    """Two huge prints must both be flagged; a std-dev test would hide them."""
    df = make_ohlcv(periods=300)
    dirty = df.copy()
    for row in (100, 200):
        for col in ("open", "high", "low", "close"):
            dirty.iloc[row, dirty.columns.get_loc(col)] = dirty.iloc[row][col] * 20
    report = engine.validate(wrap(dirty))
    assert report.stats["price_jumps"] >= 2


# ----------------------------------------------------------------- missing bars

def test_missing_daily_bars_are_counted(engine, clean_daily):
    dropped = clean_daily.drop(clean_daily.index[50:55])
    report = engine.validate(wrap(dropped))
    assert report.stats["missing_candles"] >= 5


def test_missing_intraday_bars_are_counted(engine, clean_hourly):
    dropped = clean_hourly.drop(clean_hourly.index[100:102])
    report = engine.validate(wrap(dropped, timeframe="1H"))
    assert report.stats["missing_candles"] >= 1


def test_session_breaks_are_not_counted_as_missing(engine):
    """An overnight gap is a closed market, not lost data."""
    days = []
    for day in pd.date_range("2024-01-01", periods=20, freq="B", tz="UTC"):
        session = pd.date_range(day + pd.Timedelta(hours=13), periods=7, freq="1h", tz="UTC")
        days.append(session)
    index = pd.DatetimeIndex(np.concatenate([d.values for d in days])).tz_localize("UTC")
    df = make_ohlcv(periods=len(index), freq="1h", weekdays_only=False)
    df.index = index
    df.index.name = "timestamp"

    report = engine.validate(wrap(df, timeframe="1H"))
    # 140 bars over 20 sessions: overnight gaps must not be reported as missing.
    assert report.stats["missing_candles"] == 0
    assert report.stats["large_gaps"] > 0


# ---------------------------------------------------------------------- volume

def test_missing_volume_warns_when_asset_claims_reliable(engine):
    df = make_ohlcv(periods=300, with_volume=False)
    report = engine.validate(wrap(df), _asset(has_reliable_volume=True))
    assert any(i.code == "no_volume" for i in report.issues)


def test_missing_volume_is_silent_when_asset_declares_it_unreliable(engine):
    """Spot FX has no volume by nature. That is not a defect."""
    df = make_ohlcv(periods=300, with_volume=False)
    report = engine.validate(wrap(df), _asset(has_reliable_volume=False))
    assert not any(i.code == "no_volume" for i in report.issues)
    assert report.stats["volume_reliable"] is False


def test_negative_volume_fails(engine, clean_daily):
    dirty = clean_daily.copy()
    dirty.iloc[10, dirty.columns.get_loc("volume")] = -1.0
    report = engine.validate(wrap(dirty), _asset(has_reliable_volume=True))
    assert report.status is QualityStatus.FAIL


def test_mostly_zero_volume_warns(engine, clean_daily):
    dirty = clean_daily.copy()
    dirty.iloc[:100, dirty.columns.get_loc("volume")] = 0.0
    report = engine.validate(wrap(dirty), _asset(has_reliable_volume=True))
    assert any(i.code == "zero_volume" for i in report.issues)


# ----------------------------------------------------------------- stale / fresh

def test_frozen_feed_warns(engine, clean_daily):
    dirty = clean_daily.copy()
    for col in ("open", "high", "low", "close"):
        dirty.iloc[:100, dirty.columns.get_loc(col)] = 100.0
    report = engine.validate(wrap(dirty))
    assert any(i.code == "stale_bars" for i in report.issues)


def test_old_data_warns_about_freshness(engine, clean_daily):
    report = engine.validate(wrap(clean_daily))
    assert any(i.code == "stale_data" for i in report.issues)
    assert report.stats["data_age_hours"] > 0


# --------------------------------------------------------------- status ordering

def test_status_is_the_worst_finding(engine, clean_daily):
    dirty = clean_daily.copy()
    dirty.iloc[10, dirty.columns.get_loc("low")] = -1.0  # FAIL
    report = engine.validate(wrap(dirty))
    assert report.status is QualityStatus.FAIL
    assert report.passed is False


def test_error_message_names_the_failures(engine):
    report = engine.validate(wrap(empty_frame()))
    err = DataQualityError(report)
    assert "empty" in str(err).lower()
