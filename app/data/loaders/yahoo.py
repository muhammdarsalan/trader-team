"""Yahoo Finance chart-API provider.

Free, no account, no API key, no rate-limit token. Uses the public
``/v8/finance/chart`` JSON endpoint directly rather than a wrapper library, so
the dependency surface stays small and failures are legible.

Vendor limitations, enforced rather than silently absorbed:

* No spot XAUUSD series exists. ``configs/assets.yaml`` maps XAUUSD to ``GC=F``
  (COMEX front-month gold futures) as a proxy. See ``docs/data.md``.
* Intraday retention is capped by Yahoo: ~7 days of 1-minute bars and ~730 days
  of other intraday intervals. Requests beyond that raise rather than return a
  quietly truncated series.
* There is no native 4-hour interval; the service layer resamples 1H -> 4H.
* Daily bars are stamped by Yahoo at the session's local open. They are
  re-stamped here to midnight UTC of the session date, which is the
  conventional daily-bar label and what makes cross-asset joins line up.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from app.config.models import AssetUniverse, ProviderConfig
from app.data.interfaces import DataFetchError, DataUnavailableError, MarketDataProvider
from app.data.schema import MarketData, coerce_schema
from app.utils.logging import get_logger
from app.utils.timeutils import Timeframe, normalize_timeframe, to_utc, utcnow

logger = get_logger(__name__)

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

# Canonical timeframe -> Yahoo interval string. Only these are native.
_NATIVE_INTERVALS: dict[str, str] = {
    "1M": "1m",
    "5M": "5m",
    "15M": "15m",
    "30M": "30m",
    "1H": "1h",
    "1D": "1d",
}

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class YahooProvider(MarketDataProvider):
    """Bars from Yahoo Finance's public chart endpoint."""

    name = "yahoo"

    def __init__(
        self,
        assets: AssetUniverse | None = None,
        config: ProviderConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.assets = assets
        self.config = config or ProviderConfig()
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT, "Accept": "application/json"})
        self._last_request_at: float = 0.0

    # ------------------------------------------------------------------ API

    @property
    def native_timeframes(self) -> tuple[str, ...]:
        """Timeframes Yahoo serves directly. Others must be resampled."""
        return tuple(_NATIVE_INTERVALS)

    def supports(self, symbol: str, timeframe: str | Timeframe) -> bool:  # noqa: ARG002
        try:
            tf = normalize_timeframe(timeframe)
        except ValueError:
            return False
        return tf.code in _NATIVE_INTERVALS

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        start: datetime | pd.Timestamp | str | None = None,
        end: datetime | pd.Timestamp | str | None = None,
    ) -> MarketData:
        tf = normalize_timeframe(timeframe)
        ticker = self._ticker(symbol)
        self._check_retention(tf, start)

        params: dict[str, Any] = {
            "interval": _NATIVE_INTERVALS[self._require_native(tf, symbol)],
            "includePrePost": "false",
            "events": "div,splits",
        }
        if start is None and end is None:
            # period1=0 asks for the entire available history.
            params["period1"] = 0
            params["period2"] = int(utcnow().timestamp())
        else:
            start_ts = to_utc(start) if start is not None else pd.Timestamp("1970-01-01", tz="UTC")
            end_ts = to_utc(end) if end is not None else utcnow()
            if end_ts < start_ts:
                raise ValueError(f"end ({end_ts}) precedes start ({start_ts})")
            params["period1"] = int(start_ts.timestamp())
            # Yahoo's period2 is exclusive at bar granularity; pad by one day so
            # a request ending on date D actually includes D's bar.
            params["period2"] = int((end_ts + pd.Timedelta(days=1)).timestamp())

        payload = self._request(ticker, params)
        df, meta = self._parse(payload, tf, symbol=symbol, ticker=ticker)

        # Trim the pad and honour the caller's inclusive bounds exactly.
        if start is not None:
            df = df[df.index >= to_utc(start)]
        if end is not None:
            df = df[df.index <= to_utc(end)]

        data = MarketData(symbol=symbol, timeframe=tf, df=df, provider=self.name, metadata=meta)
        logger.info("Fetched market data", extra={"summary": data.describe(), "provider": self.name})
        return data

    def get_latest_data(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        bars: int = 500,
    ) -> MarketData:
        tf = normalize_timeframe(timeframe)
        # Ask for a generous window (weekends and holidays mean calendar span
        # exceeds bar count), then keep the tail.
        span_minutes = tf.minutes * max(bars, 1) * 2.2
        start = utcnow() - pd.Timedelta(minutes=span_minutes) - pd.Timedelta(days=5)
        data = self.get_historical_data(symbol, tf, start=start, end=None)
        if len(data) > bars:
            data = data.replace(df=data.df.iloc[-bars:])
        return data

    # -------------------------------------------------------------- internals

    def _require_native(self, tf: Timeframe, symbol: str) -> str:
        if tf.code not in _NATIVE_INTERVALS:
            raise DataUnavailableError(
                f"Yahoo has no native {tf.code} interval for {symbol}. "
                f"Native intervals: {', '.join(_NATIVE_INTERVALS)}. "
                "Request a native timeframe and resample (MarketDataService does this "
                "automatically)."
            )
        return tf.code

    def _ticker(self, symbol: str) -> str:
        """Map the canonical platform symbol to the Yahoo ticker."""
        key = symbol.strip().upper()
        if self.assets is not None and key in self.assets.assets:
            return self.assets.get(key).provider_symbol(self.name)
        return key

    def _check_retention(self, tf: Timeframe, start: Any) -> None:
        """Refuse requests Yahoo cannot serve, instead of returning short data."""
        if tf.code == "1D" or start is None:
            return
        opts = self.config.options
        limit_days = (
            float(opts.get("max_days_1m", 7))
            if tf.code == "1M"
            else float(opts.get("max_days_intraday", 730))
        )
        age_days = (utcnow() - to_utc(start)).total_seconds() / 86_400
        if age_days > limit_days:
            raise DataUnavailableError(
                f"Yahoo retains only ~{limit_days:.0f} days of {tf.code} bars; "
                f"requested start is {age_days:.0f} days old. "
                "Use a longer timeframe, or supply an intraday CSV via the 'csv' provider."
            )

    def _request(self, ticker: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET with rate limiting, bounded retries and exponential backoff."""
        url = _CHART_URL.format(ticker=requests.utils.quote(ticker, safe=""))
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            self._throttle()
            try:
                response = self._session.get(url, params=params, timeout=self.config.timeout_seconds)
            except requests.RequestException as exc:
                last_error = exc
            else:
                if response.status_code == 404:
                    raise DataUnavailableError(
                        f"Yahoo does not know ticker {ticker!r}. Check provider_symbols "
                        "in configs/assets.yaml."
                    )
                if response.status_code == 429:
                    last_error = DataFetchError(f"Rate limited by Yahoo for {ticker}")
                elif response.status_code >= 500:
                    last_error = DataFetchError(f"Yahoo server error {response.status_code}")
                elif not response.ok:
                    # 4xx other than 404/429: retrying will not help.
                    raise DataFetchError(
                        f"Yahoo returned HTTP {response.status_code} for {ticker}: "
                        f"{response.text[:200]}"
                    )
                else:
                    try:
                        return response.json()
                    except ValueError as exc:
                        last_error = DataFetchError(f"Non-JSON response from Yahoo: {exc}")

            if attempt < attempts - 1:
                delay = self.config.retry_backoff_seconds * (2**attempt)
                logger.warning(
                    "Retrying Yahoo request",
                    extra={"ticker": ticker, "attempt": attempt + 1, "delay_s": delay,
                           "error": str(last_error)},
                )
                time.sleep(delay)

        raise DataFetchError(
            f"Failed to fetch {ticker} from Yahoo after {attempts} attempt(s): {last_error}"
        ) from last_error

    def _throttle(self) -> None:
        wait = self.config.rate_limit_seconds - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _parse(
        self,
        payload: dict[str, Any],
        tf: Timeframe,
        *,
        symbol: str,
        ticker: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Turn the chart JSON into a canonical frame plus provenance metadata."""
        chart = payload.get("chart") or {}
        if chart.get("error"):
            err = chart["error"]
            raise DataUnavailableError(
                f"Yahoo error for {ticker}: {err.get('code')} - {err.get('description')}"
            )

        results = chart.get("result")
        if not results:
            raise DataFetchError(f"Yahoo returned no result block for {ticker}")

        result = results[0]
        meta_in = result.get("meta", {})
        timestamps = result.get("timestamp") or []

        quote_blocks = (result.get("indicators") or {}).get("quote") or [{}]
        quote = quote_blocks[0]

        exchange_tz = meta_in.get("exchangeTimezoneName", "UTC")

        if not timestamps:
            from app.data.schema import empty_frame

            return empty_frame(), self._metadata(meta_in, ticker, tf, exchange_tz, complete=True)

        raw = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume"),
            }
        )

        df = coerce_schema(raw, assume_tz="UTC")

        # Yahoo pads holidays and halts with all-null rows. They are absence of
        # trading, not missing data, so drop them here rather than let the
        # quality engine report thousands of false "invalid OHLC" rows.
        df = df.dropna(subset=["open", "high", "low", "close"], how="all")

        if tf.code == "1D":
            df = self._normalise_daily_index(df, exchange_tz)

        complete = self._last_bar_complete(df, tf)
        meta = self._metadata(meta_in, ticker, tf, exchange_tz, complete=complete)
        meta["symbol_requested"] = symbol.upper()
        return df, meta

    @staticmethod
    def _normalise_daily_index(df: pd.DataFrame, exchange_tz: str) -> pd.DataFrame:
        """Re-stamp daily bars from local session open to midnight UTC.

        Yahoo labels a daily bar with the session's local opening instant
        (e.g. 13:30 UTC for a New York session). Keeping that would make daily
        series from different venues fail to align on the same trading date.
        """
        if df.empty:
            return df
        local_dates = df.index.tz_convert(exchange_tz).normalize().tz_localize(None)
        out = df.copy()
        out.index = pd.DatetimeIndex(local_dates).tz_localize("UTC")
        out.index.name = df.index.name
        # A venue-local date can theoretically collide after normalisation; keep
        # the last observation for a date.
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    @staticmethod
    def _last_bar_complete(df: pd.DataFrame, tf: Timeframe) -> bool:
        """Whether the final bar has closed.

        Paper trading must not act on a bar that is still forming.
        """
        if df.empty:
            return True
        bar_end = df.index[-1] + pd.Timedelta(minutes=tf.minutes)
        return bool(utcnow() >= bar_end)

    def _metadata(
        self,
        meta_in: dict[str, Any],
        ticker: str,
        tf: Timeframe,
        exchange_tz: str,
        *,
        complete: bool,
    ) -> dict[str, Any]:
        return {
            "provider": self.name,
            "provider_ticker": ticker,
            "timeframe": tf.code,
            "instrument_type": meta_in.get("instrumentType"),
            "exchange": meta_in.get("fullExchangeName"),
            "exchange_timezone": exchange_tz,
            "currency": meta_in.get("currency"),
            "first_trade_date": meta_in.get("firstTradeDate"),
            "last_bar_complete": complete,
            "retrieved_at": utcnow().isoformat(),
        }
