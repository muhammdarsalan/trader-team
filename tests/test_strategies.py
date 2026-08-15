"""The five strategies, the base-class contract and the registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.models import FeatureConfig, RegimeConfig, StrategyConfig
from app.data.schema import MarketData, coerce_schema, empty_frame
from app.features.engine import FeatureEngine
from app.regimes.detector import RegimeDetector
from app.regimes.models import MarketRegime, RegimeType, VolatilityState
from app.signals.models import Signal, SignalDirection
from app.strategies import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    Strategy,
    StrategyNotRegisteredError,
    SupportResistanceStrategy,
    TrendFollowingStrategy,
    available_strategies,
    build_enabled_strategies,
    create_strategy,
    register_strategy,
)
from app.strategies.base import StrategyMetadata
from tests.conftest import make_ohlcv

ALL_STRATEGIES = [
    TrendFollowingStrategy,
    SupportResistanceStrategy,
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
]


# ------------------------------------------------------------------- fixtures

def wrap(df: pd.DataFrame) -> MarketData:
    return MarketData(symbol="XAUUSD", timeframe="1D", df=df, provider="synthetic")


def features_for(df: pd.DataFrame):
    return FeatureEngine(FeatureConfig()).compute(wrap(df))


def regime_for(features, timestamp=None) -> MarketRegime:
    return RegimeDetector(RegimeConfig()).detect(features, timestamp)


def fake_regime(regime: RegimeType, confidence: float = 0.8) -> MarketRegime:
    return MarketRegime(
        regime=regime, confidence=confidence,
        volatility=VolatilityState.MEDIUM, trend_strength=0.6,
    )


def trending_market(n: int = 500, slope: float = 1.5) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    index = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    # Start high enough that a sustained downtrend stays positive throughout;
    # a negative price is not a market and the Signal validator rejects it.
    base = 500.0 + max(0.0, -slope) * n * 1.5
    close = base + slope * np.arange(n) + rng.normal(0, 0.5, n)
    return coerce_schema(
        pd.DataFrame(
            {
                "open": close - slope * 0.4,
                "high": close + abs(slope) + 0.5,
                "low": close - abs(slope) - 0.5,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=index,
        )
    )


def ranging_market(n: int = 500, amplitude: float = 8.0) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = 500.0 + amplitude * np.sin(np.arange(n) * 0.35)
    return coerce_schema(
        pd.DataFrame(
            {
                "open": close, "high": close + 1.5, "low": close - 1.5,
                "close": close, "volume": np.full(n, 5000.0),
            },
            index=index,
        )
    )


def _frame(close: np.ndarray, wick: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    open_ = np.concatenate([[close[0]], close[:-1]])
    return coerce_schema(
        pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) + wick,
                "low": np.minimum(open_, close) - wick,
                "close": close,
                "volume": np.full(len(close), 5000.0),
            },
            index=index,
        )
    )


def exponential_trend(n: int = 500, rate: float = 0.004, seed: int = 8) -> pd.DataFrame:
    """A compounding trend.

    A *linear* trend has a decaying percentage rate of change, so a momentum
    threshold expressed in percent stops firing as price rises. Real trends
    compound; this fixture does too.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = 500.0 * (1 + rate) ** np.arange(n) * (1 + rng.normal(0, 0.002, n))
    return _frame(close, close * 0.004, index)


def consolidation_then_breakout(n: int = 700, seed: int = 13) -> pd.DataFrame:
    """Wide chop, then a tight range, then a decisive break.

    The breakout strategy requires the range to have *tightened* relative to
    recent history. A uniformly quiet market never satisfies that, which is
    correct behaviour and worth pinning with a fixture that does.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = np.empty(n)
    close[:350] = 500.0 + 25.0 * np.sin(np.arange(350) * 0.25) + rng.normal(0, 4.0, 350)
    close[350:500] = 500.0 + rng.normal(0, 1.2, 150)
    close[500:] = 500.0 + np.linspace(0, 90, n - 500) + rng.normal(0, 1.2, n - 500)
    return _frame(close, np.full(n, 1.5), index)


def noisy_range_with_wicks(n: int = 600, seed: int = 21) -> pd.DataFrame:
    """A directionless market whose bars have pronounced wicks.

    Support/resistance needs two things a smooth series cannot provide: swing
    pivots close enough to price to be "at" a level, and rejection wicks large
    relative to the candle body.

    The noise is mean-anchored rather than a random walk. A walk drifts away
    from its own old pivots, so by the time a level is confirmed price is no
    longer near it - which is realistic, but tests nothing here.

    Wicks are asymmetric, because a rejection candle is asymmetric by
    definition. With symmetric wicks the rejection ratio cannot exceed 0.5
    even in principle: the wick is measured against the *whole* bar range, so
    an equal wick on each side caps out at exactly half.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = 500.0 + rng.normal(0, 1.0, n)
    open_ = np.concatenate([[close[0]], close[:-1]])

    # Probe down and reject below the mean; probe up and reject above it.
    below = close < 500.0
    lower_wick = np.where(below, 4.0, 0.2)
    upper_wick = np.where(below, 0.2, 4.0)

    return coerce_schema(
        pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) + upper_wick,
                "low": np.minimum(open_, close) - lower_wick,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=index,
        )
    )


def calm_with_shocks(
    n: int = 700, every: int = 60, shock: float = 8.0, calm: float = 0.5, seed: int = 17
) -> pd.DataFrame:
    """A quiet market punctuated by single-bar shocks that revert.

    Bollinger Bands are only breached by a move large relative to *recent*
    volatility. A smooth oscillation never manages it - the bands widen to
    contain the cycle - so mean reversion needs shocks after calm.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = 500.0 + rng.normal(0, calm, n)
    for i in range(60, n, every):
        close[i] += shock if (i // every) % 2 else -shock
    return _frame(close, np.full(n, calm), index)


def run(strategy: Strategy, df: pd.DataFrame, timestamp=None, regime=None) -> Signal:
    features = features_for(df)
    timestamp = timestamp or features.df.index[-1]
    return strategy.generate_signal(
        wrap(df), features, regime or regime_for(features, timestamp), timestamp
    )


# ------------------------------------------------------------------- registry

def test_all_five_strategies_are_registered():
    assert {
        "trend_following", "support_resistance", "breakout",
        "mean_reversion", "momentum",
    } <= set(available_strategies())


def test_create_strategy_returns_an_instance():
    assert isinstance(create_strategy("trend_following"), TrendFollowingStrategy)


def test_unknown_strategy_lists_alternatives():
    with pytest.raises(StrategyNotRegisteredError, match="momentum"):
        create_strategy("does_not_exist")


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_strategy("momentum", MomentumStrategy)


def test_registry_rejects_non_strategy_classes():
    class NotAStrategy:
        pass

    with pytest.raises(TypeError, match="must be a subclass of Strategy"):
        register_strategy("bogus", NotAStrategy)  # type: ignore[arg-type]


def test_build_enabled_strategies_from_config(config_dir):
    from app.config.loader import load_strategies_config

    strategies = build_enabled_strategies(load_strategies_config(config_dir))
    assert len(strategies) == 5


def test_disabled_strategies_are_not_built(config_dir):
    from app.config.models import StrategiesConfig

    config = StrategiesConfig(
        strategies={
            "momentum": StrategyConfig(enabled=True),
            "breakout": StrategyConfig(enabled=False),
        }
    )
    names = [s.name for s in build_enabled_strategies(config)]
    assert names == ["momentum"]


def test_unregistered_config_entry_is_skipped_not_fatal():
    """One bad config entry must not stop the other strategies from running."""
    from app.config.models import StrategiesConfig

    config = StrategiesConfig(
        strategies={
            "momentum": StrategyConfig(enabled=True),
            "not_a_real_strategy": StrategyConfig(enabled=True),
        }
    )
    names = [s.name for s in build_enabled_strategies(config)]
    assert names == ["momentum"]


# ------------------------------------------------------------------- metadata

@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.metadata.name)
def test_strategy_declares_complete_metadata(cls):
    """Section 5 of the brief: a strategy must state its own requirements."""
    meta = cls.metadata
    assert isinstance(meta, StrategyMetadata)
    assert meta.name and meta.description
    assert meta.supported_timeframes
    assert meta.min_history_bars > 0
    assert meta.indicators_used
    assert meta.assumptions, "a strategy must state its assumptions"


@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.metadata.name)
def test_metadata_timeframes_are_canonical(cls):
    from app.utils.timeutils import TIMEFRAMES

    assert all(tf in TIMEFRAMES for tf in cls.metadata.supported_timeframes)


# --------------------------------------------------------------- preconditions

@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.metadata.name)
def test_strategy_waits_on_empty_data(cls):
    strategy = cls()
    features = features_for(empty_frame())
    signal = strategy.generate_signal(wrap(empty_frame()), features, fake_regime(RegimeType.RANGING))
    assert signal.direction is SignalDirection.WAIT


@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.metadata.name)
def test_strategy_waits_on_insufficient_history(cls):
    df = make_ohlcv(periods=40)
    signal = run(cls(), df)
    assert signal.direction is SignalDirection.WAIT
    assert any("history" in r.lower() or "warm-up" in r.lower() for r in signal.reasoning)


@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.metadata.name)
def test_strategy_waits_during_warmup(cls):
    df = make_ohlcv(periods=500)
    features = features_for(df)
    early = features.df.index[10]
    signal = cls().generate_signal(wrap(df), features, fake_regime(RegimeType.RANGING), early)
    assert signal.direction is SignalDirection.WAIT
    assert any("warm-up" in r for r in signal.reasoning)


@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.metadata.name)
def test_strategy_waits_on_unsupported_timeframe(cls):
    df = make_ohlcv(periods=500)
    data = MarketData(symbol="XAUUSD", timeframe="5M", df=df, provider="synthetic")
    features = FeatureEngine().compute(data)
    signal = cls().generate_signal(
        data, features, fake_regime(RegimeType.RANGING), features.df.index[-1]
    )
    if "5M" not in cls.metadata.supported_timeframes:
        assert signal.direction is SignalDirection.WAIT
        assert any("does not support" in r for r in signal.reasoning)


@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.metadata.name)
def test_strategy_never_raises_on_normal_data(cls):
    """One strategy failing must not be able to take down a run."""
    df = make_ohlcv(periods=500, seed=91)
    features = features_for(df)
    for timestamp in features.df.index[::37]:
        cls().generate_signal(wrap(df), features, regime_for(features, timestamp), timestamp)


# ------------------------------------------------------- signals are well-formed

@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.metadata.name)
def test_actionable_signals_are_internally_consistent(cls):
    """Whatever a strategy emits must satisfy the Signal contract."""
    strategy = cls()
    for df in (trending_market(), ranging_market(), make_ohlcv(periods=500, seed=13)):
        features = features_for(df)
        for timestamp in features.df.index[features.warmup_bars :: 11]:
            signal = strategy.generate_signal(
                wrap(df), features, regime_for(features, timestamp), timestamp
            )
            assert signal.strategy == cls.metadata.name
            if not signal.is_actionable:
                continue
            # The Signal constructor validates sides; re-assert the economics.
            assert signal.risk_per_unit > 0
            if signal.direction is SignalDirection.LONG:
                assert signal.stop_loss < signal.entry_price
            else:
                assert signal.stop_loss > signal.entry_price


@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.metadata.name)
def test_every_signal_records_reasoning(cls):
    strategy = cls()
    df = make_ohlcv(periods=500, seed=13)
    features = features_for(df)
    for timestamp in features.df.index[features.warmup_bars :: 23]:
        signal = strategy.generate_signal(
            wrap(df), features, regime_for(features, timestamp), timestamp
        )
        assert signal.reasoning, f"{cls.metadata.name} produced a signal with no reasoning"


@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.metadata.name)
def test_confidence_floor_is_enforced(cls):
    """A strategy's own minimum must be applied without each one remembering to."""
    strategy = cls(StrategyConfig(min_confidence=0.999, timeframes=["1D"]))
    df = trending_market()
    features = features_for(df)
    for timestamp in features.df.index[features.warmup_bars :: 17]:
        signal = strategy.generate_signal(
            wrap(df), features, regime_for(features, timestamp), timestamp
        )
        if signal.is_actionable:
            assert signal.confidence >= 0.999


# ------------------------------------------------- strategies must actually fire

#: Each strategy paired with a market representing the conditions it targets.
FAVOURABLE_MARKETS = [
    (TrendFollowingStrategy, "trending", lambda: trending_market()),
    (SupportResistanceStrategy, "noisy_range_with_wicks", noisy_range_with_wicks),
    (BreakoutStrategy, "consolidation_breakout", consolidation_then_breakout),
    (MeanReversionStrategy, "calm_with_shocks", calm_with_shocks),
    (MomentumStrategy, "exponential_trend", exponential_trend),
]


@pytest.mark.parametrize(
    ("cls", "market_name", "market"), FAVOURABLE_MARKETS, ids=lambda v: getattr(v, "__name__", v)
    if not hasattr(v, "metadata") else v.metadata.name,
)
def test_strategy_fires_under_favourable_conditions(cls, market_name, market):
    """A strategy that never fires is silently broken.

    Every gate in these strategies is a reason to decline, and a subtly wrong
    threshold produces a strategy that simply never trades - which no other
    test would catch, because 'always WAIT' satisfies every safety property.
    """
    strategy = cls(StrategyConfig(min_confidence=0.0, timeframes=["1D"]))
    df = market()
    features = features_for(df)

    actionable = [
        signal
        for timestamp in features.df.index[features.warmup_bars :]
        if (
            signal := strategy.generate_signal(
                wrap(df), features, regime_for(features, timestamp), timestamp
            )
        ).is_actionable
    ]

    assert actionable, (
        f"{cls.metadata.name} produced no actionable signal across {len(features.df)} bars "
        f"of a {market_name} market - check its gates"
    )


# ------------------------------------------------------------ strategy logic

def test_trend_following_goes_long_in_an_uptrend():
    signal = run(TrendFollowingStrategy(StrategyConfig(min_confidence=0.0)), trending_market())
    assert signal.direction is SignalDirection.LONG


def test_trend_following_goes_short_in_a_downtrend():
    signal = run(
        TrendFollowingStrategy(StrategyConfig(min_confidence=0.0)),
        trending_market(slope=-1.5),
    )
    assert signal.direction is SignalDirection.SHORT


def test_trend_following_waits_when_adx_is_weak():
    signal = run(TrendFollowingStrategy(StrategyConfig(min_confidence=0.0)), ranging_market())
    assert signal.direction is SignalDirection.WAIT


def test_trend_following_rejects_inverted_ema_periods():
    with pytest.raises(ValueError, match="must be shorter"):
        TrendFollowingStrategy(StrategyConfig(params={"fast_ema": 50, "slow_ema": 21}))


def test_mean_reversion_refuses_to_fade_a_trend():
    """The single most important rule in that strategy."""
    strategy = MeanReversionStrategy(StrategyConfig(min_confidence=0.0))
    df = trending_market()
    features = features_for(df)

    for timestamp in features.df.index[features.warmup_bars :: 7]:
        signal = strategy.generate_signal(
            wrap(df), features, fake_regime(RegimeType.TRENDING_UP), timestamp
        )
        assert not signal.is_actionable, "mean reversion must never fade a confirmed trend"


def test_mean_reversion_blocks_on_high_adx():
    strategy = MeanReversionStrategy(
        StrategyConfig(min_confidence=0.0, params={"max_adx": 5.0})
    )
    signal = run(strategy, trending_market(), regime=fake_regime(RegimeType.RANGING))
    assert signal.direction is SignalDirection.WAIT
    assert any("ADX" in r for r in signal.reasoning)


def test_mean_reversion_rejects_inverted_rsi_thresholds():
    with pytest.raises(ValueError, match="must be below"):
        MeanReversionStrategy(
            StrategyConfig(params={"rsi_oversold": 70.0, "rsi_overbought": 30.0})
        )


def test_mean_reversion_targets_the_mean():
    """The target is the mean itself - that is the thesis, not an ATR multiple."""
    strategy = MeanReversionStrategy(StrategyConfig(min_confidence=0.0))
    df = calm_with_shocks()
    features = features_for(df)

    checked = 0
    for timestamp in features.df.index[features.warmup_bars :]:
        signal = strategy.generate_signal(
            wrap(df), features, fake_regime(RegimeType.RANGING), timestamp
        )
        if signal.is_actionable:
            middle = float(features.at(timestamp)["bb_middle"])
            assert signal.take_profit == pytest.approx(middle)
            checked += 1

    assert checked > 0, "fixture produced no mean-reversion setup"


def test_breakout_requires_a_fresh_break():
    """Buying every bar of a sustained trend is not a breakout strategy."""
    strategy = BreakoutStrategy(StrategyConfig(min_confidence=0.0))
    df = trending_market(slope=2.0)
    features = features_for(df)

    actionable = 0
    for timestamp in features.df.index[features.warmup_bars :]:
        signal = strategy.generate_signal(
            wrap(df), features, regime_for(features, timestamp), timestamp
        )
        if signal.is_actionable:
            actionable += 1

    bars = len(features.df) - features.warmup_bars
    assert actionable / bars < 0.15, "breakout fired on too many bars of one trend"


def test_breakout_waits_inside_the_channel():
    strategy = BreakoutStrategy(StrategyConfig(min_confidence=0.0))
    df = ranging_market(amplitude=2.0)
    signal = run(strategy, df)
    assert signal.direction is SignalDirection.WAIT


def test_support_resistance_requires_a_rejection_wick():
    strict = SupportResistanceStrategy(
        StrategyConfig(min_confidence=0.0, params={"rejection_wick_ratio": 0.99})
    )
    df = ranging_market()
    features = features_for(df)

    for timestamp in features.df.index[features.warmup_bars :]:
        signal = strict.generate_signal(
            wrap(df), features, fake_regime(RegimeType.RANGING), timestamp
        )
        assert not signal.is_actionable or signal.metadata.get("rejection_ratio", 0) >= 0.99


def test_momentum_requires_macd_agreement_when_configured():
    strategy = MomentumStrategy(
        StrategyConfig(min_confidence=0.0, params={"require_macd_agreement": True})
    )
    df = make_ohlcv(periods=500, seed=71)
    features = features_for(df)

    for timestamp in features.df.index[features.warmup_bars :: 5]:
        signal = strategy.generate_signal(
            wrap(df), features, regime_for(features, timestamp), timestamp
        )
        if signal.is_actionable:
            hist = signal.metadata.get("macd_hist")
            if hist is not None:
                expected = 1 if signal.direction is SignalDirection.LONG else -1
                assert np.sign(hist) == expected


def test_momentum_waits_on_flat_prices():
    n = 400
    index = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    flat = coerce_schema(
        pd.DataFrame(
            {
                "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
                "volume": 1000.0,
            },
            index=index,
        )
    )
    signal = run(MomentumStrategy(StrategyConfig(min_confidence=0.0)), flat)
    assert signal.direction is SignalDirection.WAIT


# ------------------------------------------------------------------ causality

@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.metadata.name)
def test_signal_at_a_bar_is_unchanged_by_later_bars(cls):
    """The end-to-end look-ahead guarantee for the whole phase-2 stack.

    The signal produced for bar t must be identical whether the series ends at
    t or continues for another hundred bars.
    """
    strategy = cls(StrategyConfig(min_confidence=0.0, timeframes=["1D"]))
    df = trending_market(n=500)
    cutoff = 420
    timestamp = df.index[cutoff - 1]

    full_features = features_for(df)
    truncated_features = features_for(df.iloc[:cutoff])

    from_full = strategy.generate_signal(
        wrap(df), full_features, regime_for(full_features, timestamp), timestamp
    )
    from_truncated = strategy.generate_signal(
        wrap(df.iloc[:cutoff]),
        truncated_features,
        regime_for(truncated_features, timestamp),
        timestamp,
    )

    assert from_full.direction is from_truncated.direction
    assert from_full.confidence == pytest.approx(from_truncated.confidence)
    assert from_full.entry_price == from_truncated.entry_price
    assert from_full.stop_loss == from_truncated.stop_loss
    assert from_full.take_profit == from_truncated.take_profit
