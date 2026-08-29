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

    is_proxy: bool = Field(
        default=False,
        description=(
            "True when the configured provider series stands in for the instrument "
            "rather than being the instrument - for example front-month futures used "
            "as a proxy for a spot rate. Monitoring surfaces must label proxy series "
            "as proxies, and a YAML comment cannot reach a dashboard."
        ),
    )
    data_caveat: str | None = Field(
        default=None,
        description=(
            "Operator-facing caveat shown wherever this instrument's data is "
            "displayed. Use it for limitations a reader must know to interpret a "
            "number correctly, such as proxy tracking error."
        ),
    )

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
    # Standing weight, before the selector applies regime fit and measured
    # performance. 1.0 means "no opinion". This exists so that the question
    # "does down-weighting a redundant strategy help?" can be answered by
    # running an experiment rather than by deleting the strategy and hoping.
    weight: float = Field(default=1.0, ge=0)

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


class RiskConfig(StrictModel):
    """Position sizing and exposure limits.

    Every number here caps something. None of them is a performance setting -
    a risk engine's job is to bound the damage of being wrong, which it will
    frequently be.
    """

    # --- per-trade sizing ---
    risk_per_trade: float = Field(
        default=0.01, gt=0, le=0.1,
        description="Fraction of equity risked between entry and stop",
    )
    sizing_method: Literal["percent_risk", "fixed_risk", "fixed_units"] = "percent_risk"
    fixed_risk_amount: float = Field(default=100.0, gt=0)
    fixed_units: float = Field(default=1.0, gt=0)

    # --- per-position caps ---
    max_position_notional_pct: float = Field(
        default=0.5, gt=0,
        description="Cap on one position's notional as a fraction of equity. Above 1.0 "
                    "implies leverage.",
    )
    min_position_units: float = Field(default=0.0, ge=0)

    # --- portfolio caps ---
    max_concurrent_positions: int = Field(default=3, ge=1)
    max_positions_per_symbol: int = Field(default=1, ge=1)
    max_portfolio_risk: float = Field(
        default=0.06, gt=0,
        description="Total open risk (sum of per-trade risk) as a fraction of equity",
    )
    max_portfolio_notional_pct: float = Field(default=3.0, gt=0)
    max_risk_per_strategy: float = Field(default=0.03, gt=0)

    # --- correlation ---
    correlation_threshold: float = Field(
        default=0.7, ge=0, le=1,
        description="Above this, two symbols count as effectively one exposure",
    )
    max_correlated_risk: float = Field(default=0.03, gt=0)
    correlation_lookback: int = Field(default=60, ge=10)

    # --- drawdown controls ---
    drawdown_reduce_threshold: float = Field(
        default=0.10, gt=0, lt=1, description="Above this drawdown, size is scaled down"
    )
    drawdown_reduce_factor: float = Field(default=0.5, gt=0, le=1)
    max_drawdown_limit: float = Field(
        default=0.20, gt=0, lt=1, description="Above this drawdown, no new trades open"
    )
    daily_loss_limit: float = Field(
        default=0.03, gt=0, lt=1, description="Loss in one day that halts new trades"
    )

    # --- data quality ---
    # A limit computed from corrupt inputs is not a limit. Every number the risk
    # engine works with - equity, stop distance, ATR - descends from the price
    # series, so the grade that series was given has to reach the sizing
    # decision rather than stopping at the report header.
    block_on_data_quality_fail: bool = Field(
        default=True,
        description="Refuse to size a position on data the quality gate graded FAIL",
    )
    block_on_data_quality_warning: bool = Field(
        default=False,
        description="Also refuse on WARNING-grade data. Off by default because most "
                    "long historical windows carry at least one warning; turn it on for "
                    "paper trading, where a stale or gappy feed is a live hazard.",
    )
    require_known_data_quality: bool = Field(
        default=True,
        description="Treat an unstated data grade as a refusal rather than as consent. "
                    "'Nobody checked' and 'it was checked and passed' must not produce "
                    "the same position size.",
    )

    @model_validator(mode="after")
    def _drawdown_thresholds_ordered(self) -> RiskConfig:
        if self.drawdown_reduce_threshold >= self.max_drawdown_limit:
            raise ValueError(
                f"drawdown_reduce_threshold ({self.drawdown_reduce_threshold}) must be below "
                f"max_drawdown_limit ({self.max_drawdown_limit}); otherwise trading halts "
                "before sizing is ever reduced"
            )
        return self


class ExecutionConfig(StrictModel):
    """Fill simulation assumptions.

    These decide whether a backtest is informative or fiction. Optimistic
    execution assumptions are the most common way a strategy appears profitable
    on paper and is not in practice.
    """

    # A signal is generated from bar t's close, so the earliest tradable price
    # is bar t+1's open. Filling at the signal bar's close is trading on a
    # price that had already passed.
    fill_delay_bars: int = Field(default=1, ge=1)

    apply_spread: bool = True
    spread_multiplier: float = Field(
        default=1.0, ge=0, description="Scales the asset's configured typical spread"
    )

    slippage_model: Literal["none", "fixed_ticks", "atr_fraction", "percent"] = "atr_fraction"
    slippage_value: float = Field(default=0.05, ge=0)

    commission_per_unit: float = Field(default=0.0, ge=0)
    commission_pct: float = Field(default=0.0, ge=0)
    min_commission: float = Field(default=0.0, ge=0)

    # When a bar's range contains both the stop and the target, OHLC data
    # cannot say which was touched first. Assuming the target is a choice to
    # believe the flattering possibility every single time.
    same_bar_resolution: Literal["stop_first", "target_first", "skip"] = "stop_first"

    # A gap through the stop fills at the open, not at the stop price. Pretending
    # otherwise removes exactly the losses that hurt most.
    honour_gaps: bool = True

    allow_shorts: bool = True

    # Reject orders whose notional exceeds this multiple of the bar's traded
    # value, where volume is meaningful.
    max_volume_participation: float = Field(default=0.1, gt=0)
    enforce_volume_limit: bool = False


class BacktestConfig(StrictModel):
    """Backtest run settings."""

    initial_balance: float = Field(default=10_000.0, gt=0)
    warmup_bars: int | None = Field(
        default=None, description="Overrides the feature engine's own warm-up when set"
    )
    # A bar-limited run is for smoke tests, never for research conclusions.
    max_bars: int | None = None

    close_positions_at_end: bool = Field(
        default=True,
        description="Close open positions at the final bar so equity is realised. "
                    "Leaving them open silently omits their outcome from the metrics.",
    )
    bars_per_year: int = Field(
        default=252, gt=0, description="Used to annualise returns; 252 for daily bars"
    )
    risk_free_rate: float = Field(default=0.0, ge=0, description="Annual, for Sharpe")


class WalkForwardSettings(StrictModel):
    """Geometry of the rolling train/test evaluation."""

    enabled: bool = True
    folds: int = Field(default=5, ge=1)
    #: Explicit window sizes. Null derives them from the history available, so a
    #: short series produces a smaller study rather than an error.
    train_bars: int | None = Field(default=None, ge=1)
    test_bars: int | None = Field(default=None, ge=1)
    step_bars: int | None = Field(default=None, ge=1)
    anchored: bool = Field(
        default=False,
        description="Grow the training window from a fixed start instead of rolling it. "
                    "Neither setting is correct in general.",
    )
    test_fraction: float = Field(default=0.25, gt=0, lt=1)


class MonteCarloSettings(StrictModel):
    """Trade-sequence resampling."""

    enabled: bool = True
    simulations: int = Field(default=2_000, ge=1)
    method: Literal["permutation", "bootstrap"] = "permutation"
    #: Resample in blocks to preserve streaks. 1 assumes consecutive trades are
    #: independent, which they are not.
    block_size: int = Field(default=1, ge=1)


class RobustnessSettings(StrictModel):
    """Parameter sensitivity sweeps."""

    enabled: bool = True
    #: Dotted paths into the configuration, e.g. ``features.rsi_period``.
    parameters: list[str] = Field(default_factory=list)
    #: Fractional half-width of the neighbourhood swept around each value.
    span: float = Field(default=0.3, gt=0, lt=1)
    points: int = Field(default=5, ge=3)


class FeedbackSettings(StrictModel):
    """Whether validated research findings may change the running configuration.

    Ships **disabled**, and that is the important part. A research loop that
    silently retunes the thing it is measuring stops being a measurement: the
    next study inherits the previous study's conclusion as its starting point,
    and the evidence that would have contradicted it is never gathered.

    Turning this on does not make the loop trusting. A recommendation still has
    to clear the evidence bar below, and one that does not is *refused out loud*
    rather than skipped - see :func:`app.research.feedback.apply_recommendations`.
    A silent no-op would be indistinguishable from a change that was applied and
    happened to alter nothing.
    """

    #: The gate. False means research output is reporting only.
    enabled: bool = False

    #: Out-of-sample trades required before any finding may move a weight. The
    #: same threshold the validation report uses before it will conclude.
    min_out_of_sample_trades: int = Field(default=30, ge=1)

    #: Walk-forward folds that must have selected a variant before fold
    #: agreement counts as repetition rather than coincidence.
    min_walk_forward_folds: int = Field(default=3, ge=1)

    #: Require per-fold out-of-sample evidence. With this false a single
    #: validation window can move a weight, which is a materially weaker claim;
    #: it is offered because it is a legitimate research choice, not because it
    #: is a good default.
    require_walk_forward: bool = True

    #: The largest cut this loop will make to a standing weight, as a fraction.
    #: A recommendation proposing more is refused, not clamped.
    max_weight_reduction: float = Field(default=0.5, ge=0, le=1)


class ResearchConfig(StrictModel):
    """How a validation study is run.

    None of these settings is a performance dial. They decide how hard the
    evidence is tested, and loosening them makes a result look better without
    making it truer.
    """

    #: (in-sample, validation, out-of-sample) proportions of the usable history.
    split_fractions: tuple[float, float, float] = (0.5, 0.2, 0.3)

    #: Bars dropped between windows so a trade opened in one cannot still be
    #: open in the next. Zero lets one market episode be scored twice.
    embargo_bars: int = Field(default=5, ge=0)

    #: How candidates are ranked on training windows. Return-maximising
    #: objectives are rejected by the objective registry on purpose.
    objective: str = "sortino"

    #: Below this many trades in a window, the study declines to conclude.
    min_trades_for_conclusions: int = Field(default=30, ge=1)

    #: Strategy pairs correlated above this are reported as redundancy
    #: hypotheses. They are never acted on automatically.
    correlation_threshold: float = Field(default=0.7, ge=0, le=1)

    #: Run each strategy alone over the in-sample window, so their returns can
    #: be correlated. The graph aggregates every strategy into one signal before
    #: a position opens, so without this the trade log only ever shows the
    #: ensemble. It is also the most expensive part of a study - one extra
    #: backtest per strategy - so it can be turned off while iterating, at the
    #: cost of the return-correlation half of the analysis.
    isolate_strategies: bool = True

    #: Test the weight/selection variants a redundancy finding suggests, on data
    #: not used to find the redundancy.
    test_weight_variants: bool = True
    variant_down_weight: float = Field(default=0.5, ge=0, le=1)

    walk_forward: WalkForwardSettings = Field(default_factory=WalkForwardSettings)
    monte_carlo: MonteCarloSettings = Field(default_factory=MonteCarloSettings)
    robustness: RobustnessSettings = Field(default_factory=RobustnessSettings)

    #: Whether validated findings may feed back into the strategy configuration.
    #: Disabled by default; see :class:`FeedbackSettings`.
    feedback: FeedbackSettings = Field(default_factory=FeedbackSettings)

    @field_validator("split_fractions")
    @classmethod
    def _positive_fractions(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(f <= 0 for f in v):
            raise ValueError(
                f"Every split fraction must be positive, got {v}. A study with an empty "
                "window is a study missing the part that would have contradicted it."
            )
        return v


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
