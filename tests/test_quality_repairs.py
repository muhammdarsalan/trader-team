"""Cleaning must not hide vendor defects from the quality gate.

Cleaning runs before validation, so a repaired series looks pristine by the
time it is graded. These tests pin the behaviour that makes the original damage
visible in the report.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.config.models import QualityThresholds
from app.data.processors.normalize import clean_ohlcv
from app.data.schema import MarketData
from app.data.validators.quality import DataQualityEngine, QualityStatus
from tests.conftest import make_ohlcv


@pytest.fixture
def engine() -> DataQualityEngine:
    return DataQualityEngine(QualityThresholds())


def wrap(df: pd.DataFrame) -> MarketData:
    return MarketData(symbol="TEST", timeframe="1D", df=df, provider="test")


def test_repairs_are_invisible_without_the_cleaning_report(engine):
    """Baseline: this is the blind spot the cleaning report closes."""
    df = make_ohlcv(periods=400)
    dirty = df.copy()
    for row in range(0, 100):
        dirty.iloc[row, dirty.columns.get_loc("high")] = dirty.iloc[row]["low"]

    cleaned, _ = clean_ohlcv(dirty)
    report = engine.validate(wrap(cleaned))  # cleaning report withheld

    assert report.stats["invalid_ohlc"] == 0
    assert not any(i.code == "repaired_bars" for i in report.issues)


def test_repairs_are_reported_when_the_cleaning_report_is_passed(engine):
    df = make_ohlcv(periods=400)
    dirty = df.copy()
    for row in range(0, 20):  # 5%: enough to warn, below the fail threshold
        dirty.iloc[row, dirty.columns.get_loc("high")] = dirty.iloc[row]["low"]

    cleaned, cleaning = clean_ohlcv(dirty)
    report = engine.validate(wrap(cleaned), cleaning=cleaning)

    issue = next(i for i in report.issues if i.code == "repaired_bars")
    assert issue.count == 20
    assert report.stats["bars_repaired"] == 20
    assert issue.severity is QualityStatus.WARNING


def test_clean_data_produces_no_repair_finding(engine, clean_daily):
    cleaned, cleaning = clean_ohlcv(clean_daily)
    report = engine.validate(wrap(cleaned), cleaning=cleaning)
    assert not any(i.code == "repaired_bars" for i in report.issues)
    assert report.stats["bars_repaired"] == 0


def test_dropped_rows_are_reported(engine):
    df = make_ohlcv(periods=400)
    dirty = pd.concat([df, df.iloc[:30]]).sort_index()

    cleaned, cleaning = clean_ohlcv(dirty)
    report = engine.validate(wrap(cleaned), cleaning=cleaning)

    issue = next(i for i in report.issues if i.code == "dropped_bars")
    assert issue.count == 30


def test_catastrophic_repair_rate_fails(engine):
    """A feed where a quarter of bars need surgery is not research-grade."""
    df = make_ohlcv(periods=400)
    dirty = df.copy()
    for row in range(0, 200):
        dirty.iloc[row, dirty.columns.get_loc("high")] = dirty.iloc[row]["low"]

    cleaned, cleaning = clean_ohlcv(dirty)
    report = engine.validate(wrap(cleaned), cleaning=cleaning)

    assert report.status is QualityStatus.FAIL


def test_service_passes_cleaning_report_to_the_gate(config_dir):
    """End-to-end: the facade must not drop the cleaning report on the floor."""
    from app.data.service import MarketDataService
    from tests.test_service import StubProvider

    df = make_ohlcv(periods=400)
    dirty = df.copy()
    for row in range(0, 50):
        dirty.iloc[row, dirty.columns.get_loc("high")] = dirty.iloc[row]["low"]

    result = MarketDataService(
        provider=StubProvider(dirty), config_dir=config_dir
    ).get_historical_data("XAUUSD", "1D")

    assert result.quality.stats["bars_repaired"] == 50
    assert any(i.code == "repaired_bars" for i in result.quality.issues)
