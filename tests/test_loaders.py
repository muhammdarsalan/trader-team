"""Providers: registry, CSV, synthetic, and Yahoo response parsing.

Yahoo is exercised against a recorded response shape rather than the live
endpoint, so the parsing logic is covered without a network dependency. The one
genuinely live test is marked ``network`` and excluded from the default run.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.config.models import AssetUniverse, ProviderConfig
from app.data.interfaces import DataFetchError, DataUnavailableError
from app.data.loaders import (
    CsvProvider,
    ProviderNotRegisteredError,
    SyntheticProvider,
    YahooProvider,
    available_providers,
    get_provider,
    register_provider,
)
from app.data.schema import validate_schema

# ------------------------------------------------------------------- registry

def test_builtin_providers_are_registered():
    assert {"yahoo", "csv", "synthetic"} <= set(available_providers())


def test_get_provider_returns_instance():
    assert isinstance(get_provider("synthetic"), SyntheticProvider)


def test_unknown_provider_lists_alternatives():
    with pytest.raises(ProviderNotRegisteredError, match="synthetic"):
        get_provider("bloomberg")


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_provider("synthetic", SyntheticProvider)


def test_factory_drops_unaccepted_kwargs():
    """The service passes assets/config uniformly; simple providers ignore them."""
    provider = get_provider("synthetic", assets=None, config=ProviderConfig())
    assert isinstance(provider, SyntheticProvider)


# ------------------------------------------------------------------ synthetic

def test_synthetic_produces_valid_schema():
    data = SyntheticProvider().get_historical_data("XAUUSD", "1D", "2020-01-01", "2021-01-01")
    assert validate_schema(data.df) is data.df
    assert len(data) > 200


def test_synthetic_is_deterministic():
    a = SyntheticProvider(seed=5).get_historical_data("XAUUSD", "1D", "2020-01-01", "2020-06-01")
    b = SyntheticProvider(seed=5).get_historical_data("XAUUSD", "1D", "2020-01-01", "2020-06-01")
    pd.testing.assert_frame_equal(a.df, b.df)


def test_synthetic_differs_by_symbol():
    a = SyntheticProvider(seed=5).get_historical_data("XAUUSD", "1D", "2020-01-01", "2020-06-01")
    b = SyntheticProvider(seed=5).get_historical_data("EURUSD", "1D", "2020-01-01", "2020-06-01")
    assert not a.df["close"].equals(b.df["close"])


def test_synthetic_prices_stay_positive_and_ordered():
    df = SyntheticProvider().get_historical_data("X", "1D", "2015-01-01", "2025-01-01").df
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()


def test_synthetic_flags_itself_as_synthetic():
    """Results computed on this must never be mistaken for real research."""
    data = SyntheticProvider().get_historical_data("X", "1D", "2020-01-01", "2020-06-01")
    assert data.metadata["synthetic"] is True
    assert "SYNTHETIC" in data.metadata["warning"]


def test_synthetic_latest_returns_requested_bars():
    data = SyntheticProvider().get_latest_data("X", "1D", bars=50)
    assert len(data) == 50


# ------------------------------------------------------------------------ CSV

def write_csv(path, *, sep=",", decimal=".", header=None, rows=None) -> None:
    header = header or "timestamp,open,high,low,close,volume"
    rows = rows or [
        "2024-01-01 00:00:00,100.0,101.0,99.0,100.5,1000",
        "2024-01-02 00:00:00,100.5,102.0,100.0,101.5,1200",
        "2024-01-03 00:00:00,101.5,103.0,101.0,102.5,900",
    ]
    text = "\n".join([header, *rows])
    if sep != ",":
        text = text.replace(",", sep)
    if decimal == ",":
        text = text.replace(".", ",")
    path.write_text(text, encoding="utf-8")


def test_csv_reads_standard_file(tmp_path):
    write_csv(tmp_path / "XAUUSD_1D.csv")
    data = CsvProvider(directory=tmp_path).get_historical_data("XAUUSD", "1D")
    assert len(data) == 3
    assert validate_schema(data.df) is data.df
    assert data.df["close"].iloc[0] == 100.5


def test_csv_accepts_vendor_column_names(tmp_path):
    write_csv(
        tmp_path / "XAUUSD_1D.csv",
        header="Date,Open,High,Low,Close,Vol",
    )
    data = CsvProvider(directory=tmp_path).get_historical_data("XAUUSD", "1D")
    assert list(data.df.columns) == ["open", "high", "low", "close", "volume"]


def test_csv_handles_european_format(tmp_path):
    """';' separator with ',' decimals is what most EU broker exports look like."""
    path = tmp_path / "XAUUSD_1D.csv"
    path.write_text(
        "timestamp;open;high;low;close;volume\n"
        "2024-01-01;100,5;101,5;99,5;100,8;1000\n"
        "2024-01-02;100,8;102,5;100,1;101,9;1100\n"
        "2024-01-03;101,9;103,0;101,5;102,7;1300\n",
        encoding="utf-8",
    )
    data = CsvProvider(directory=tmp_path).get_historical_data("XAUUSD", "1D")
    assert data.df["close"].iloc[0] == pytest.approx(100.8)


def test_csv_missing_file_raises_with_expected_name(tmp_path):
    provider = CsvProvider(directory=tmp_path)
    with pytest.raises(DataUnavailableError, match="XAUUSD_1D.csv"):
        provider.get_historical_data("XAUUSD", "1D")


def test_csv_empty_file_raises(tmp_path):
    (tmp_path / "XAUUSD_1D.csv").write_text("", encoding="utf-8")
    with pytest.raises(DataFetchError, match="empty"):
        CsvProvider(directory=tmp_path).get_historical_data("XAUUSD", "1D")


def test_csv_header_only_raises(tmp_path):
    (tmp_path / "XAUUSD_1D.csv").write_text("timestamp,open,high,low,close\n", encoding="utf-8")
    with pytest.raises(DataFetchError, match="no rows"):
        CsvProvider(directory=tmp_path).get_historical_data("XAUUSD", "1D")


def test_csv_respects_date_range(tmp_path):
    write_csv(tmp_path / "XAUUSD_1D.csv")
    data = CsvProvider(directory=tmp_path).get_historical_data(
        "XAUUSD", "1D", start="2024-01-02", end="2024-01-02"
    )
    assert len(data) == 1


def test_csv_assumes_utc_by_default(tmp_path):
    """Guessing a venue timezone would silently shift a UTC export by hours."""
    write_csv(tmp_path / "XAUUSD_1D.csv")
    data = CsvProvider(directory=tmp_path).get_historical_data("XAUUSD", "1D")
    assert data.df.index[0].isoformat() == "2024-01-01T00:00:00+00:00"


def test_csv_honours_explicit_timezone_option(tmp_path):
    write_csv(tmp_path / "XAUUSD_1D.csv")
    provider = CsvProvider(
        directory=tmp_path,
        config=ProviderConfig(options={"assume_timezone": "America/New_York"}),
    )
    data = provider.get_historical_data("XAUUSD", "1D")
    assert data.df.index[0].isoformat() == "2024-01-01T05:00:00+00:00"


def test_csv_records_source_provenance(tmp_path):
    write_csv(tmp_path / "XAUUSD_1D.csv")
    data = CsvProvider(directory=tmp_path).get_historical_data("XAUUSD", "1D")
    assert data.metadata["source_file"].endswith("XAUUSD_1D.csv")
    assert data.metadata["last_bar_complete"] is True


def test_csv_supports_reports_availability(tmp_path):
    write_csv(tmp_path / "XAUUSD_1D.csv")
    provider = CsvProvider(directory=tmp_path)
    assert provider.supports("XAUUSD", "1D") is True
    assert provider.supports("EURUSD", "1D") is False


# ---------------------------------------------------------------------- Yahoo

def yahoo_payload(
    *,
    timestamps: list[int] | None = None,
    tz: str = "America/New_York",
    with_nulls: bool = False,
) -> dict:
    timestamps = timestamps or [1704153600, 1704240000, 1704326400]
    n = len(timestamps)
    quote = {
        "open": [100.0 + i for i in range(n)],
        "high": [101.0 + i for i in range(n)],
        "low": [99.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)],
        "volume": [1000 + i for i in range(n)],
    }
    if with_nulls:
        for key in quote:
            quote[key] = [*quote[key], None]
        timestamps = [*timestamps, timestamps[-1] + 86400]
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "currency": "USD", "symbol": "GC=F", "instrumentType": "FUTURE",
                        "fullExchangeName": "COMEX", "exchangeTimezoneName": tz,
                        "firstTradeDate": 967608000,
                    },
                    "timestamp": timestamps,
                    "indicators": {"quote": [quote]},
                }
            ],
            "error": None,
        }
    }


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(payload) if isinstance(payload, dict) else str(payload)

    def json(self):
        if not isinstance(self._payload, dict):
            raise ValueError("not json")
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.response


def yahoo_provider(response, **kwargs) -> YahooProvider:
    return YahooProvider(
        config=ProviderConfig(rate_limit_seconds=0, max_retries=0, **kwargs),
        session=FakeSession(response),
    )


def test_yahoo_parses_chart_response():
    provider = yahoo_provider(FakeResponse(yahoo_payload()))
    data = provider.get_historical_data("XAUUSD", "1D")
    assert len(data) == 3
    assert validate_schema(data.df) is data.df
    assert data.metadata["exchange"] == "COMEX"


def test_yahoo_normalises_daily_bars_to_midnight_utc():
    """Yahoo stamps daily bars at the local session open; cross-asset joins need midnight."""
    provider = yahoo_provider(FakeResponse(yahoo_payload()))
    data = provider.get_historical_data("XAUUSD", "1D")
    assert all(ts.hour == 0 and ts.minute == 0 for ts in data.df.index)


def test_yahoo_drops_all_null_holiday_rows():
    provider = yahoo_provider(FakeResponse(yahoo_payload(with_nulls=True)))
    data = provider.get_historical_data("XAUUSD", "1D")
    assert len(data) == 3


def test_yahoo_maps_canonical_symbol_to_vendor_ticker():
    universe = AssetUniverse(
        assets={
            "XAUUSD": {
                "name": "Gold", "asset_class": "METAL", "quote_currency": "USD",
                "tick_size": 0.01, "typical_spread": 0.3,
                "provider_symbols": {"yahoo": "GC=F"},
            }
        }
    )
    session = FakeSession(FakeResponse(yahoo_payload()))
    provider = YahooProvider(
        assets=universe, config=ProviderConfig(rate_limit_seconds=0), session=session
    )
    provider.get_historical_data("XAUUSD", "1D")
    assert "GC%3DF" in session.calls[0][0] or "GC=F" in session.calls[0][0]


def test_yahoo_unknown_ticker_raises_unavailable():
    provider = yahoo_provider(FakeResponse({}, status_code=404))
    with pytest.raises(DataUnavailableError, match="does not know ticker"):
        provider.get_historical_data("NOPE", "1D")


def test_yahoo_server_error_raises_after_retries():
    provider = yahoo_provider(FakeResponse({}, status_code=503))
    with pytest.raises(DataFetchError):
        provider.get_historical_data("XAUUSD", "1D")


def test_yahoo_reports_api_level_error():
    payload = {"chart": {"result": None,
                         "error": {"code": "Not Found", "description": "No data found"}}}
    provider = yahoo_provider(FakeResponse(payload))
    with pytest.raises(DataUnavailableError, match="No data found"):
        provider.get_historical_data("XAUUSD", "1D")


def test_yahoo_has_no_native_4h():
    """4H must come from resampling 1H, never from a vendor that lacks it."""
    provider = yahoo_provider(FakeResponse(yahoo_payload()))
    assert provider.supports("XAUUSD", "4H") is False
    with pytest.raises(DataUnavailableError, match="no native 4H"):
        provider.get_historical_data("XAUUSD", "4H")


def test_yahoo_refuses_out_of_retention_intraday_request():
    """Silently returning a truncated series would corrupt a backtest."""
    provider = yahoo_provider(FakeResponse(yahoo_payload()))
    provider.config.options.update({"max_days_1m": 7, "max_days_intraday": 730})
    with pytest.raises(DataUnavailableError, match="retains only"):
        provider.get_historical_data("XAUUSD", "1M", start="2015-01-01")


def test_yahoo_empty_timestamps_yield_empty_series():
    payload = yahoo_payload()
    payload["chart"]["result"][0]["timestamp"] = []
    provider = yahoo_provider(FakeResponse(payload))
    data = provider.get_historical_data("XAUUSD", "1D")
    assert data.is_empty


@pytest.mark.network
def test_yahoo_live_fetch_returns_gold_history():
    """The one live test. Run with: pytest -m network"""
    provider = YahooProvider(config=ProviderConfig(rate_limit_seconds=0))
    data = provider.get_historical_data("GC=F", "1D", start="2020-01-01", end="2020-12-31")
    assert len(data) > 200
    assert validate_schema(data.df) is data.df
