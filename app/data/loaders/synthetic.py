"""Synthetic market-data provider.

Exists so the entire pipeline is testable offline and deterministically. It is
a *test fixture*, not a research tool: geometric Brownian motion with an
optional drift and regime schedule resembles price data superficially, but it
has none of the fat tails, volatility clustering or autocorrelation structure
of a real market.

Never report a backtest result computed on synthetic data as evidence that a
strategy works. Its legitimate uses are: exercising code paths, testing edge
cases, and sanity-checking that a strategy is *not* profitable on noise.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from app.data.interfaces import MarketDataProvider
from app.data.schema import MarketData, coerce_schema
from app.utils.timeutils import Timeframe, normalize_timeframe, to_utc, utcnow


class SyntheticProvider(MarketDataProvider):
    """Deterministic pseudo-random OHLCV generator."""

    name = "synthetic"

    def __init__(
        self,
        seed: int = 42,
        start_price: float = 1_800.0,
        annual_drift: float = 0.0,
        annual_volatility: float = 0.15,
        with_volume: bool = True,
        weekdays_only: bool = True,
    ) -> None:
        self.seed = seed
        self.start_price = start_price
        self.annual_drift = annual_drift
        self.annual_volatility = annual_volatility
        self.with_volume = with_volume
        self.weekdays_only = weekdays_only

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        start: datetime | pd.Timestamp | str | None = None,
        end: datetime | pd.Timestamp | str | None = None,
    ) -> MarketData:
        tf = normalize_timeframe(timeframe)
        end_ts = to_utc(end) if end is not None else utcnow().normalize()
        start_ts = to_utc(start) if start is not None else end_ts - pd.Timedelta(days=1000)

        index = pd.date_range(start=start_ts, end=end_ts, freq=tf.pandas_freq, tz="UTC")
        if self.weekdays_only:
            index = index[index.dayofweek < 5]

        df = self._generate(index, tf, symbol)
        return MarketData(
            symbol=symbol,
            timeframe=tf,
            df=df,
            provider=self.name,
            metadata={
                "provider": self.name,
                "synthetic": True,
                "seed": self.seed,
                "warning": "SYNTHETIC DATA - not a market. Never cite results computed on it.",
                "last_bar_complete": True,
                "retrieved_at": utcnow().isoformat(),
            },
        )

    def get_latest_data(
        self, symbol: str, timeframe: str | Timeframe, bars: int = 500
    ) -> MarketData:
        tf = normalize_timeframe(timeframe)
        end = utcnow()
        start = end - pd.Timedelta(minutes=tf.minutes * bars * 2)
        data = self.get_historical_data(symbol, tf, start=start, end=end)
        return data.replace(df=data.df.iloc[-bars:]) if len(data) > bars else data

    # -------------------------------------------------------------- internals

    def _generate(self, index: pd.DatetimeIndex, tf: Timeframe, symbol: str) -> pd.DataFrame:
        n = len(index)
        if n == 0:
            from app.data.schema import empty_frame

            return empty_frame()

        # Seed off the symbol too, so different instruments differ but each is
        # reproducible run to run.
        rng = np.random.default_rng(self.seed + (hash(symbol.upper()) % 10_000))

        bars_per_year = (365.25 * 24 * 60) / tf.minutes
        dt = 1.0 / bars_per_year
        sigma = self.annual_volatility * np.sqrt(dt)
        mu = (self.annual_drift - 0.5 * self.annual_volatility**2) * dt

        log_returns = rng.normal(loc=mu, scale=sigma, size=n)
        close = self.start_price * np.exp(np.cumsum(log_returns))

        # Open = previous close, so bars chain without artificial gaps.
        open_ = np.empty(n)
        open_[0] = self.start_price
        open_[1:] = close[:-1]

        # Intrabar extremes: a half-range drawn from the same volatility scale.
        wick = np.abs(rng.normal(0.0, sigma, size=n)) * close
        high = np.maximum(open_, close) + wick
        low = np.minimum(open_, close) - wick
        low = np.maximum(low, 1e-9)  # prices stay strictly positive

        volume = (
            rng.lognormal(mean=10.0, sigma=0.4, size=n).round()
            if self.with_volume
            else np.full(n, np.nan)
        )

        return coerce_schema(
            pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                index=index,
            )
        )
