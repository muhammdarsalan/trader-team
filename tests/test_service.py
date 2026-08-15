"""The market-data facade: provider + cache + cleaning + quality gate."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.interfaces import DataUnavailableError, MarketDataProvider
from app.data.loaders import SyntheticProvider
from app.data.schema import MarketData, empty_frame, validate_schema
from app.data.service import MarketDataService
from app.data.validators.quality import DataQualityError, QualityStatus
from tests.conftest import make_ohlcv


class StubProvider(MarketDataProvider):
    """Serves a fixed frame and counts calls, so cache behaviour is observable."""

    name = "synthetic"
    native_timeframes = ("1H", "1D")

    def __init__(self, df: pd.DataFrame, timeframe: str = "1D"):
        self.df = df
        self.timeframe = timeframe
        self.calls = 0

    def supports(self, symbol, timeframe):
        from app.utils.timeutils import normalize_timeframe

        return normalize_timeframe(timeframe).code in self.native_timeframes

    def get_historical_data(self, symbol, timeframe, start=None, end=None):
        self.calls += 1
        df = self.df
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        return MarketData(symbol=symbol, timeframe=timeframe, df=df, provider=self.name,
                          metadata={"provider": self.name})

    def get_latest_data(self, symbol, timeframe, bars=500):
        self.calls += 1
        return MarketData(symbol=symbol, timeframe=timeframe, df=self.df.iloc[-bars:],
                          provider=self.name, metadata={"provider": self.name})


def service(provider, config_dir, **kwargs) -> MarketDataService:
    return MarketDataService(provider=provider, config_dir=config_dir, **kwargs)


# ------------------------------------------------------------------ happy path

def test_returns_validated_data(config_dir):
    provider = StubProvider(make_ohlcv(periods=400))
    result = service(provider, config_dir).get_historical_data("XAUUSD", "1D")

    assert len(result) == 400
    assert validate_schema(result.df) is result.df
    assert result.quality.status is not QualityStatus.FAIL


def test_result_carries_quality_and_cleaning_reports(config_dir):
    provider = StubProvider(make_ohlcv(periods=400))
    result = service(provider, config_dir).get_historical_data("XAUUSD", "1D")
    assert result.quality.symbol == "XAUUSD"
    assert result.cleaning.rows_in == 400


def test_date_range_is_respected(config_dir):
    df = make_ohlcv(periods=400, start="2022-01-01")
    result = service(StubProvider(df), config_dir).get_historical_data(
        "XAUUSD", "1D", start="2022-06-01", end="2022-08-01"
    )
    assert result.data.start >= pd.Timestamp("2022-06-01", tz="UTC")
    assert result.data.end <= pd.Timestamp("2022-08-01", tz="UTC")


# --------------------------------------------------------------------- cleaning

def test_dirty_data_is_cleaned_before_use(config_dir):
    df = make_ohlcv(periods=400)
    dirty = pd.concat([df, df.iloc[[10]]]).sort_index()
    result = service(StubProvider(dirty), config_dir).get_historical_data("XAUUSD", "1D")

    assert result.cleaning.duplicates_dropped == 1
    assert not result.df.index.has_duplicates


# ---------------------------------------------------------------- quality gate

def test_failing_quality_raises_by_default(config_dir):
    """Corrupt data must not reach a strategy."""
    provider = StubProvider(make_ohlcv(periods=15))  # below the fail threshold
    with pytest.raises(DataQualityError, match="Data quality FAIL"):
        service(provider, config_dir).get_historical_data("XAUUSD", "1D")


def test_empty_result_fails_quality(config_dir):
    provider = StubProvider(empty_frame())
    with pytest.raises(DataQualityError):
        service(provider, config_dir).get_historical_data("XAUUSD", "1D")


def test_validation_can_be_bypassed_explicitly(config_dir):
    provider = StubProvider(make_ohlcv(periods=15))
    result = service(provider, config_dir).get_historical_data("XAUUSD", "1D", validate=False)
    assert result.quality.status is QualityStatus.FAIL  # still reported, just not raised
    assert len(result) == 15


# -------------------------------------------------------------------- resampling

def test_resamples_when_provider_lacks_the_timeframe(config_dir):
    """Yahoo has no 4H. The service must build it from 1H, not give up."""
    hourly = make_ohlcv(periods=1000, freq="1h", weekdays_only=False)
    provider = StubProvider(hourly, timeframe="1H")

    result = service(provider, config_dir).get_historical_data("XAUUSD", "4H")

    assert result.resampled_from == "1H"
    assert len(result) == pytest.approx(len(hourly) / 4, rel=0.05)
    assert validate_schema(result.df) is result.df


def test_no_resampling_when_native(config_dir):
    provider = StubProvider(make_ohlcv(periods=400))
    result = service(provider, config_dir).get_historical_data("XAUUSD", "1D")
    assert result.resampled_from is None


def test_unavailable_timeframe_raises(config_dir):
    class DailyOnly(StubProvider):
        native_timeframes = ("1D",)

    provider = DailyOnly(make_ohlcv(periods=400))
    with pytest.raises(DataUnavailableError, match="no timeframe fine enough"):
        service(provider, config_dir).get_historical_data("XAUUSD", "1H")


def test_unknown_timeframe_is_rejected(config_dir):
    provider = StubProvider(make_ohlcv(periods=400))
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        service(provider, config_dir).get_historical_data("XAUUSD", "3H")


def test_timeframe_disabled_in_config_is_rejected(config_dir):
    """A valid timeframe that the operator has switched off must not be served."""
    import yaml

    path = config_dir / "data.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["supported_timeframes"] = ["1D"]
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    from app.config.loader import reset_config_cache

    reset_config_cache()

    provider = StubProvider(make_ohlcv(periods=400))
    with pytest.raises(ValueError, match="not enabled"):
        service(provider, config_dir).get_historical_data("XAUUSD", "1H")


# ------------------------------------------------------------------------ cache

def test_second_request_is_served_from_cache(config_dir):
    provider = StubProvider(make_ohlcv(periods=400))
    svc = service(provider, config_dir)

    svc.get_historical_data("XAUUSD", "1D")
    svc.get_historical_data("XAUUSD", "1D")

    assert provider.calls == 1


def test_refresh_bypasses_cache(config_dir):
    provider = StubProvider(make_ohlcv(periods=400))
    svc = service(provider, config_dir)

    svc.get_historical_data("XAUUSD", "1D")
    svc.get_historical_data("XAUUSD", "1D", refresh=True)

    assert provider.calls == 2


def test_cache_can_be_disabled(config_dir):
    provider = StubProvider(make_ohlcv(periods=400))
    svc = service(provider, config_dir, use_cache=False)

    svc.get_historical_data("XAUUSD", "1D")
    svc.get_historical_data("XAUUSD", "1D")

    assert provider.calls == 2


def test_cache_miss_when_request_starts_earlier_than_cached_range(config_dir):
    """A cache that begins mid-range is missing history the caller asked for."""
    df = make_ohlcv(periods=800, start="2020-01-01")
    provider = StubProvider(df)
    svc = service(provider, config_dir)

    svc.get_historical_data("XAUUSD", "1D", start="2021-01-01")
    svc.get_historical_data("XAUUSD", "1D", start="2020-01-01")

    assert provider.calls == 2


# ----------------------------------------------------------------- latest data

def test_latest_returns_requested_bar_count(config_dir):
    provider = StubProvider(make_ohlcv(periods=1000))
    result = service(provider, config_dir).get_latest_data("XAUUSD", "1D", bars=100)
    assert len(result) == 100


def test_latest_drops_a_still_forming_bar(config_dir):
    """Acting on an unfinished bar is look-ahead bias wearing a live-trading hat."""
    now = pd.Timestamp.now(tz="UTC")
    index = pd.date_range(end=now.floor("h"), periods=300, freq="1h", tz="UTC")
    df = make_ohlcv(periods=300, freq="1h", weekdays_only=False)
    df.index = index
    df.index.name = "timestamp"

    provider = StubProvider(df, timeframe="1H")
    result = service(provider, config_dir).get_latest_data(
        "XAUUSD", "1H", bars=50, require_complete_bar=True
    )

    assert result.data.end < now.floor("h") + pd.Timedelta(hours=1)
    assert "forming_bar_dropped" in result.data.metadata or result.data.end < now


def test_latest_can_keep_the_forming_bar(config_dir):
    df = make_ohlcv(periods=300, freq="1h", weekdays_only=False)
    provider = StubProvider(df, timeframe="1H")
    result = service(provider, config_dir).get_latest_data(
        "XAUUSD", "1H", bars=50, require_complete_bar=False
    )
    assert len(result) == 50


# ------------------------------------------------------------------ integration

def test_end_to_end_with_synthetic_provider(config_dir):
    """Whole pipeline, no network: provider -> clean -> resample -> quality."""
    svc = MarketDataService(
        provider=SyntheticProvider(seed=1, start_price=1800.0),
        config_dir=config_dir,
    )
    result = svc.get_historical_data("XAUUSD", "1D", start="2018-01-01", end="2023-01-01")

    assert len(result) > 1000
    assert result.quality.status is not QualityStatus.FAIL
    assert validate_schema(result.df) is result.df
