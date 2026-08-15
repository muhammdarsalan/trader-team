"""Pydantic models describing every configuration file the platform reads."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.utils.timeutils import TIMEFRAMES, normalize_timeframe


class TradingMode(StrEnum):
    """Supported run modes.

    ``LIVE`` exists as a named constant purely so that the rest of the code can
    reject it explicitly. Real-money execution is not implemented.
    """

    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    RESEARCH = "RESEARCH"
    ANALYSIS = "ANALYSIS"
    LIVE = "LIVE"  # permanently unsupported; see PlatformConfig validation


class AssetClass(StrEnum):
    FX = "FX"
    METAL = "METAL"
    INDEX = "INDEX"
    CRYPTO = "CRYPTO"
    EQUITY = "EQUITY"
    FUTURE = "FUTURE"


class StrictModel(BaseModel):
    """Base model that rejects unknown keys.

    A silently-ignored misspelled config key is a bug that surfaces as
    mysteriously wrong research results three weeks later.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class AssetConfig(StrictModel):
    """One tradable instrument."""

    symbol: str = Field(description="Canonical platform symbol, e.g. XAUUSD")
    name: str = Field(description="Human-readable instrument name")
    asset_class: AssetClass
    quote_currency: str = Field(min_length=3, max_length=3)
    base_currency: str | None = Field(
        default=None, description="For FX pairs: the base leg, e.g. EUR in EURUSD"
    )

    # Per-provider symbol mapping. Keeps vendor tickers out of strategy code.
    provider_symbols: dict[str, str] = Field(default_factory=dict)

    tick_size: float = Field(gt=0, description="Smallest price increment")
    contract_size: float = Field(default=1.0, gt=0, description="Units per 1.0 position size")

    # Execution assumptions, used from phase 3 onward.
    typical_spread: float = Field(ge=0, description="Typical spread in price units")
    commission_per_unit: float = Field(default=0.0, ge=0)

    # Data characteristics.
    venue_timezone: str = Field(default="UTC")
    has_reliable_volume: bool = Field(
        default=False,
        description=(
            "False for OTC spot instruments where 'volume' is a broker tick count "
            "rather than traded size. Volume features are suppressed when False "
            "rather than computed on meaningless numbers."
        ),
    )
    trading_days: Literal["WEEKDAYS", "ALL"] = "WEEKDAYS"

    default_timeframe: str = "1D"
    enabled: bool = True

    @field_validator("symbol")
    @classmethod
    def _upper_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("quote_currency", "base_currency")
    @classmethod
    def _upper_ccy(cls, v: str | None) -> str | None:
        return v.strip().upper() if v else v

    @field_validator("default_timeframe")
    @classmethod
    def _known_timeframe(cls, v: str) -> str:
        return normalize_timeframe(v).code

    def provider_symbol(self, provider: str) -> str:
        """Vendor ticker for ``provider``, falling back to the canonical symbol."""
        return self.provider_symbols.get(provider, self.symbol)


class AssetUniverse(StrictModel):
    """The full configured instrument universe."""

    assets: dict[str, AssetConfig]

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        """Accept either a mapping keyed by symbol or a list of asset blocks."""
        if isinstance(data, dict) and "assets" in data:
            raw = data["assets"]
            if isinstance(raw, list):
                data = {**data, "assets": {a["symbol"].upper(): a for a in raw}}
            elif isinstance(raw, dict):
                # Inject the key as `symbol` when the block omits it.
                data = {
                    **data,
                    "assets": {
                        k.upper(): {"symbol": k.upper(), **v} if "symbol" not in v else v
                        for k, v in raw.items()
                    },
                }
        return data

    def get(self, symbol: str) -> AssetConfig:
        """Look up an asset, raising a helpful error when absent."""
        key = symbol.strip().upper()
        if key not in self.assets:
            known = ", ".join(sorted(self.assets)) or "<none>"
            raise KeyError(f"Unknown symbol {symbol!r}. Configured assets: {known}")
        return self.assets[key]

    def enabled_symbols(self) -> list[str]:
        return sorted(s for s, a in self.assets.items() if a.enabled)


class QualityThresholds(StrictModel):
    """When a data-quality report escalates from PASS to WARNING to FAIL.

    Ratios are fractions of total rows in [0, 1].
    """

    max_missing_ratio_warn: float = Field(default=0.01, ge=0, le=1)
    max_missing_ratio_fail: float = Field(default=0.10, ge=0, le=1)
    max_duplicate_ratio_warn: float = Field(default=0.0, ge=0, le=1)
    max_duplicate_ratio_fail: float = Field(default=0.01, ge=0, le=1)
    max_invalid_ohlc_ratio_warn: float = Field(default=0.0, ge=0, le=1)
    max_invalid_ohlc_ratio_fail: float = Field(default=0.001, ge=0, le=1)

    # Cleaning repairs some defects before validation runs. Without this check
    # the gate would grade the *repaired* series and never see how damaged the
    # vendor data actually was.
    max_repaired_ratio_warn: float = Field(default=0.001, ge=0, le=1)
    max_repaired_ratio_fail: float = Field(default=0.20, ge=0, le=1)

    # A single bar moving more than this many robust standard deviations is
    # flagged as a suspected bad print rather than a real market move.
    price_jump_sigma: float = Field(default=12.0, gt=0)
    max_price_jump_ratio_warn: float = Field(default=0.001, ge=0, le=1)
    max_price_jump_ratio_fail: float = Field(default=0.01, ge=0, le=1)

    # Gaps longer than this multiple of the bar interval are reported. Weekend
    # and holiday gaps are excluded before this test is applied.
    gap_tolerance_multiple: float = Field(default=3.0, gt=1)

    min_rows_warn: int = Field(default=250, ge=0)
    min_rows_fail: int = Field(default=30, ge=0)

    @model_validator(mode="after")
    def _fail_above_warn(self) -> QualityThresholds:
        pairs = [
            ("missing", self.max_missing_ratio_warn, self.max_missing_ratio_fail),
            ("duplicate", self.max_duplicate_ratio_warn, self.max_duplicate_ratio_fail),
            ("invalid_ohlc", self.max_invalid_ohlc_ratio_warn, self.max_invalid_ohlc_ratio_fail),
            ("price_jump", self.max_price_jump_ratio_warn, self.max_price_jump_ratio_fail),
            ("repaired", self.max_repaired_ratio_warn, self.max_repaired_ratio_fail),
        ]
        for name, warn, fail in pairs:
            if fail < warn:
                raise ValueError(
                    f"{name}: fail threshold ({fail}) must be >= warn threshold ({warn})"
                )
        if self.min_rows_fail > self.min_rows_warn:
            raise ValueError("min_rows_fail must be <= min_rows_warn")
        return self


class ProviderConfig(StrictModel):
    """Settings for one market-data provider."""

    enabled: bool = True
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_backoff_seconds: float = Field(default=1.5, ge=0)
    rate_limit_seconds: float = Field(
        default=0.5, ge=0, description="Minimum delay between consecutive requests"
    )
    options: dict[str, Any] = Field(default_factory=dict)


class DataConfig(StrictModel):
    """Market-data subsystem configuration."""

    default_provider: str = "yahoo"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    cache_enabled: bool = True
    cache_format: Literal["parquet", "csv"] = "parquet"
    # Re-download rather than serve cache once the cached copy is older than this.
    cache_max_age_hours: float = Field(default=24.0, gt=0)

    quality: QualityThresholds = Field(default_factory=QualityThresholds)

    # Refuse to hand corrupt data to strategies. When True a FAIL-grade quality
    # report raises instead of returning the frame.
    fail_on_quality_fail: bool = True
    warn_on_quality_warning: bool = True

    supported_timeframes: list[str] = Field(default_factory=lambda: list(TIMEFRAMES))

    @field_validator("supported_timeframes")
    @classmethod
    def _known_timeframes(cls, v: list[str]) -> list[str]:
        return [normalize_timeframe(tf).code for tf in v]

    def provider(self, name: str | None = None) -> ProviderConfig:
        key = name or self.default_provider
        return self.providers.get(key, ProviderConfig())


class FeatureConfig(StrictModel):
    """Indicator periods used by the feature engine.

    These are defaults, not truths. Every one of them is a parameter to be
    tested for robustness in phase 4 - a strategy that only works at exactly
    RSI(14) is overfit to a number Wilder picked in 1978.
    """

    # Trend
    sma_periods: list[int] = Field(default_factory=lambda: [20, 50, 200])
    ema_periods: list[int] = Field(default_factory=lambda: [9, 21, 50])
    slope_period: int = Field(default=5, ge=1)
    adx_period: int = Field(default=14, ge=2)

    # Momentum
    rsi_period: int = Field(default=14, ge=2)
    macd_fast: int = Field(default=12, ge=1)
    macd_slow: int = Field(default=26, ge=2)
    macd_signal: int = Field(default=9, ge=1)
    roc_period: int = Field(default=10, ge=1)
    momentum_period: int = Field(default=10, ge=1)

    # Volatility
    atr_period: int = Field(default=14, ge=2)
    bb_period: int = Field(default=20, ge=2)
    bb_std: float = Field(default=2.0, gt=0)
    volatility_lookback: int = Field(
        default=100, ge=10, description="Window for volatility percentile ranking"
    )

    # Structure
    swing_left: int = Field(default=3, ge=1)
    swing_right: int = Field(
        default=3, ge=1, description="Also the confirmation delay for swing pivots"
    )
    donchian_period: int = Field(default=20, ge=2)

    # Volume
    volume_period: int = Field(default=20, ge=2)
    volume_spike_threshold: float = Field(default=2.0, gt=1)

    @model_validator(mode="after")
    def _macd_periods_ordered(self) -> FeatureConfig:
        if self.macd_fast >= self.macd_slow:
            raise ValueError(
                f"macd_fast ({self.macd_fast}) must be shorter than "
                f"macd_slow ({self.macd_slow}); otherwise the indicator inverts"
            )
        return self

    def warmup_bars(self) -> int:
        """Bars needed before every feature is defined.

        Below this, some features are NaN. Strategies must not fire on a
        partially-warm feature set: an indicator computed from three
        observations is not a shorter version of the real one, it is a
        different number that will never occur in live trading.
        """
        candidates = [
            *self.sma_periods,
            *self.ema_periods,
            self.adx_period * 2,  # ADX smooths DX, which is itself smoothed
            self.rsi_period * 2,
            self.macd_slow + self.macd_signal,
            self.atr_period * 2,
            self.bb_period,
            self.volatility_lookback,
            self.donchian_period + self.swing_right,
            self.swing_left + self.swing_right + 1,
            self.volume_period * 3,
        ]
        return int(max(candidates))


class RegimeConfig(StrictModel):
    """Thresholds for deterministic market-regime classification.

    Deliberately rule-based and transparent to start with. A machine-learned
    regime classifier is easy to add later and impossible to debug if it is the
    only thing you have ever had.
    """

    # Trend strength, read from ADX.
    adx_trending: float = Field(default=25.0, gt=0, description="ADX above this = trending")
    adx_ranging: float = Field(default=20.0, gt=0, description="ADX below this = ranging")

    # Trend direction, read from normalised MA slope (per bar, in ATR units).
    slope_threshold: float = Field(
        default=0.05, ge=0, description="Minimum |slope| in ATR/bar to call a direction"
    )

    # Volatility state, read from ATR percentile in [0, 1].
    high_volatility_percentile: float = Field(default=0.80, gt=0, lt=1)
    low_volatility_percentile: float = Field(default=0.20, gt=0, lt=1)

    # Bollinger bandwidth percentile below which a squeeze is declared.
    squeeze_percentile: float = Field(default=0.20, gt=0, lt=1)

    # Below this confidence the regime is reported as UNCERTAIN. Admitting
    # ignorance is more useful than a confident wrong label.
    min_confidence: float = Field(default=0.45, ge=0, le=1)

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> RegimeConfig:
        if self.adx_ranging > self.adx_trending:
            raise ValueError(
                f"adx_ranging ({self.adx_ranging}) must not exceed "
                f"adx_trending ({self.adx_trending})"
            )
        if self.low_volatility_percentile >= self.high_volatility_percentile:
            raise ValueError(
                f"low_volatility_percentile ({self.low_volatility_percentile}) must be below "
                f"high_volatility_percentile ({self.high_volatility_percentile})"
            )
        return self


class StrategyConfig(StrictModel):
    """One strategy's activation state and parameters."""

    enabled: bool = True
    timeframes: list[str] = Field(default_factory=lambda: ["1D"])
    params: dict[str, Any] = Field(default_factory=dict)
    # Signals below this confidence are discarded by the strategy itself.
    min_confidence: float = Field(default=0.5, ge=0, le=1)

    @field_validator("timeframes")
    @classmethod
    def _known_timeframes(cls, v: list[str]) -> list[str]:
        return [normalize_timeframe(tf).code for tf in v]


class StrategiesConfig(StrictModel):
    """The strategy roster.

    Adding a strategy means writing one class, registering it, and adding a
    block here. No other file changes.
    """

    strategies: dict[str, StrategyConfig] = Field(default_factory=dict)

    # Risk defaults applied when a strategy does not specify its own.
    default_stop_atr_multiple: float = Field(default=2.0, gt=0)
    default_target_atr_multiple: float = Field(default=3.0, gt=0)

    def enabled_names(self) -> list[str]:
        return sorted(name for name, cfg in self.strategies.items() if cfg.enabled)

    def get(self, name: str) -> StrategyConfig:
        if name not in self.strategies:
            known = ", ".join(sorted(self.strategies)) or "<none>"
            raise KeyError(f"Unknown strategy {name!r}. Configured: {known}")
        return self.strategies[name]


class PlatformConfig(StrictModel):
    """Global runtime settings."""

    mode: TradingMode = TradingMode.RESEARCH

    # Global kill switch. Even in PAPER mode, no order is placed while False.
    trading_enabled: bool = False

    base_currency: str = Field(default="USD", min_length=3, max_length=3)
    starting_balance: float = Field(default=10_000.0, gt=0)

    random_seed: int = 42
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_to_file: bool = True

    @field_validator("base_currency")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _reject_live(self) -> PlatformConfig:
        if self.mode is TradingMode.LIVE:
            raise ValueError(
                "LIVE mode is not implemented. This platform is research, backtesting "
                "and paper trading only, and must not be pointed at a real-money account."
            )
        return self
