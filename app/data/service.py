"""The market-data facade.

This is the only data entry point the rest of the platform uses. It composes
provider -> cache -> cleaning -> quality gate -> resampling, so that features,
strategies and the backtester receive validated, canonical bars and nothing
else. Everything below this line is an implementation detail they never see.

    service.get_historical_data("XAUUSD", "4H", start="2015-01-01")
    service.get_latest_data("XAUUSD", "1H", bars=300)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from app.config.loader import get_config
from app.data.cache import MarketDataCache
from app.data.interfaces import DataUnavailableError, MarketDataProvider
from app.data.loaders.registry import get_provider
from app.data.processors.normalize import CleaningReport, clean_ohlcv
from app.data.processors.resample import best_source_timeframe, resample_ohlcv
from app.data.schema import MarketData
from app.data.validators.quality import (
    DataQualityEngine,
    DataQualityError,
    DataQualityReport,
    QualityStatus,
)
from app.utils.logging import get_logger
from app.utils.timeutils import Timeframe, normalize_timeframe, to_utc, utcnow

logger = get_logger(__name__)


@dataclass(frozen=True)
class DataRequestResult:
    """Everything one data request produced, including how good the data was."""

    data: MarketData
    quality: DataQualityReport
    cleaning: CleaningReport
    resampled_from: str | None = None

    @property
    def df(self) -> pd.DataFrame:
        return self.data.df

    def __len__(self) -> int:
        return len(self.data)


class MarketDataService:
    """Composes providers, cache, cleaning and validation."""

    def __init__(
        self,
        provider: MarketDataProvider | None = None,
        *,
        config_dir: Any = None,
        cache: MarketDataCache | None = None,
        quality_engine: DataQualityEngine | None = None,
        use_cache: bool | None = None,
    ) -> None:
        cfg = get_config(config_dir)
        self.assets = cfg.assets
        self.data_config = cfg.data

        self.provider = provider or self._build_provider(self.data_config.default_provider)
        self.cache = cache or MarketDataCache(fmt=self.data_config.cache_format)
        self.quality_engine = quality_engine or DataQualityEngine(self.data_config.quality)
        self.use_cache = self.data_config.cache_enabled if use_cache is None else use_cache

    def _build_provider(self, name: str) -> MarketDataProvider:
        return get_provider(
            name,
            assets=self.assets,
            config=self.data_config.provider(name),
        )

    # ------------------------------------------------------------------- API

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        start: datetime | pd.Timestamp | str | None = None,
        end: datetime | pd.Timestamp | str | None = None,
        *,
        refresh: bool = False,
        validate: bool = True,
    ) -> DataRequestResult:
        """Validated bars for ``symbol`` over ``[start, end]``.

        Args:
            symbol: canonical platform symbol.
            timeframe: any supported timeframe; resampled from a finer native
                one when the provider has no native support.
            start, end: inclusive bounds. None means "as far as available".
            refresh: bypass the cache and re-download.
            validate: run the quality gate. Disabling it is for inspecting known-
                bad data deliberately, never for production research paths.

        Raises:
            DataQualityError: quality FAIL while ``fail_on_quality_fail`` is set.
        """
        tf = normalize_timeframe(timeframe)
        self._check_timeframe_supported(tf)

        source_tf = self._resolve_source_timeframe(symbol, tf)
        raw = self._fetch(symbol, source_tf, start, end, refresh=refresh)

        cleaned_df, cleaning = clean_ohlcv(raw.df)
        if cleaning.changed:
            logger.info(
                "Cleaned market data",
                extra={"symbol": symbol, "timeframe": source_tf.code,
                       "summary": cleaning.summary()},
            )
        data = raw.replace(df=cleaned_df)

        resampled_from: str | None = None
        if source_tf.code != tf.code:
            data = data.replace(
                df=resample_ohlcv(data.df, source_tf, tf),
                timeframe=tf,
            )
            resampled_from = source_tf.code
            logger.debug(
                "Resampled market data",
                extra={"symbol": symbol, "from": source_tf.code, "to": tf.code, "bars": len(data)},
            )

        # Resampling can spill past the requested bounds (a 4H window opening
        # before `start`), and the cache may hold more history than was asked
        # for. Clip to exactly the requested range.
        if start is not None or end is not None:
            data = data.slice(
                to_utc(start) if start is not None else None,
                to_utc(end) if end is not None else None,
            )

        quality = self._grade(data, validate=validate, cleaning=cleaning)
        return DataRequestResult(
            data=data, quality=quality, cleaning=cleaning, resampled_from=resampled_from
        )

    def get_latest_data(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        bars: int = 500,
        *,
        validate: bool = True,
        require_complete_bar: bool = True,
    ) -> DataRequestResult:
        """The most recent ``bars`` bars, for paper trading and monitoring.

        Args:
            require_complete_bar: drop a still-forming final bar. Acting on a
                forming bar is look-ahead bias in live clothing: its high, low
                and close are not yet determined.
        """
        tf = normalize_timeframe(timeframe)
        self._check_timeframe_supported(tf)
        source_tf = self._resolve_source_timeframe(symbol, tf)

        # Ask for extra bars so resampling to a coarser timeframe still yields
        # `bars` complete windows.
        multiplier = max(1, tf.minutes // source_tf.minutes)
        raw = self.provider.get_latest_data(symbol, source_tf, bars=bars * multiplier + multiplier)

        cleaned_df, cleaning = clean_ohlcv(raw.df)
        data = raw.replace(df=cleaned_df)

        resampled_from: str | None = None
        if source_tf.code != tf.code:
            data = data.replace(df=resample_ohlcv(data.df, source_tf, tf), timeframe=tf)
            resampled_from = source_tf.code

        if require_complete_bar and not data.df.empty:
            data = self._drop_forming_bar(data, tf)

        if len(data) > bars:
            data = data.replace(df=data.df.iloc[-bars:])

        quality = self._grade(data, validate=validate, cleaning=cleaning)
        return DataRequestResult(
            data=data, quality=quality, cleaning=cleaning, resampled_from=resampled_from
        )

    # -------------------------------------------------------------- internals

    def _check_timeframe_supported(self, tf: Timeframe) -> None:
        if tf.code not in self.data_config.supported_timeframes:
            raise ValueError(
                f"Timeframe {tf.code} is not enabled. Supported: "
                f"{', '.join(self.data_config.supported_timeframes)} "
                "(edit supported_timeframes in configs/data.yaml)."
            )

    def _resolve_source_timeframe(self, symbol: str, tf: Timeframe) -> Timeframe:
        """Pick the timeframe to actually request from the provider.

        When the provider serves ``tf`` natively that is what we ask for.
        Otherwise we find the coarsest native timeframe that can be aggregated
        up to it - never a coarser one, which would require fabricating bars.
        """
        if self.provider.supports(symbol, tf):
            return tf

        native = getattr(self.provider, "native_timeframes", None)
        if not native:
            raise DataUnavailableError(
                f"Provider {self.provider.name!r} cannot serve {symbol} at {tf.code} and "
                "declares no native timeframes to resample from."
            )

        source = best_source_timeframe(tf, list(native))
        if source is None:
            raise DataUnavailableError(
                f"Provider {self.provider.name!r} has no timeframe fine enough to build "
                f"{tf.code} bars for {symbol}. Native: {', '.join(native)}."
            )
        return normalize_timeframe(source)

    def _fetch(
        self,
        symbol: str,
        tf: Timeframe,
        start: Any,
        end: Any,
        *,
        refresh: bool,
    ) -> MarketData:
        """Fetch from cache when fresh enough, otherwise from the provider."""
        if not self.use_cache or refresh:
            data = self.provider.get_historical_data(symbol, tf, start, end)
            if self.use_cache and not data.df.empty:
                self.cache.save(data)
            return data

        cached = self.cache.load(
            symbol, tf, self.provider.name, max_age_hours=self.data_config.cache_max_age_hours
        )
        if cached is not None and self._cache_covers(cached, start, end):
            logger.debug("Cache hit", extra={"symbol": symbol, "timeframe": tf.code})
            return cached

        data = self.provider.get_historical_data(symbol, tf, start, end)

        if cached is not None and not cached.df.empty:
            data = self.cache.merge(cached, data)

        if not data.df.empty:
            self.cache.save(data)
        return data

    @staticmethod
    def _cache_covers(cached: MarketData, start: Any, end: Any) -> bool:
        """Whether the cached range spans the request.

        Deliberately strict about the start: a cache that begins after the
        requested start is missing history the caller asked for. The end is
        allowed to fall short by the cache's own freshness window, which the
        age check has already vetted.
        """
        if cached.df.empty:
            return False
        if start is not None:
            requested_start = pd.Timestamp(start)
            if requested_start.tz is None:
                requested_start = requested_start.tz_localize("UTC")
            if cached.start > requested_start:
                return False
        if end is not None:
            requested_end = pd.Timestamp(end)
            if requested_end.tz is None:
                requested_end = requested_end.tz_localize("UTC")
            if cached.end < requested_end:
                return False
        return True

    @staticmethod
    def _drop_forming_bar(data: MarketData, tf: Timeframe) -> MarketData:
        """Remove the final bar if its window has not closed yet."""
        last_open = data.df.index[-1]
        if utcnow() < last_open + pd.Timedelta(minutes=tf.minutes):
            return data.replace(
                df=data.df.iloc[:-1],
                metadata={**data.metadata, "forming_bar_dropped": str(last_open)},
            )
        return data

    def _grade(
        self, data: MarketData, *, validate: bool, cleaning: CleaningReport | None = None
    ) -> DataQualityReport:
        """Run the quality gate and enforce the configured policy."""
        asset = self.assets.assets.get(data.symbol.upper())
        report = self.quality_engine.validate(data, asset, cleaning=cleaning)

        if not validate:
            return report

        if report.status is QualityStatus.FAIL and self.data_config.fail_on_quality_fail:
            logger.error(
                "Data quality FAIL",
                extra={"symbol": data.symbol, "timeframe": data.timeframe.code,
                       "issues": [i.code for i in report.issues_of(QualityStatus.FAIL)]},
            )
            raise DataQualityError(report)

        if report.status is QualityStatus.WARNING and self.data_config.warn_on_quality_warning:
            logger.warning(
                "Data quality WARNING",
                extra={"symbol": data.symbol, "timeframe": data.timeframe.code,
                       "issues": [i.code for i in report.issues_of(QualityStatus.WARNING)]},
            )
        return report
