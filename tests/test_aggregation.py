"""Signal aggregation and regime-aware strategy selection.

The behaviour that matters most here is the ability to conclude nothing.
Conflicting signals, suppressed strategies and evenly-matched votes must all
produce WAIT with a recorded reason, not a coin-flip dressed as consensus.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.config.models import StrategyConfig
from app.regimes.models import MarketRegime, RegimeType, VolatilityState
from app.signals.aggregator import (
    AggregationMethod,
    ConflictPolicy,
    SignalAggregator,
)
from app.signals.models import Signal, SignalDirection
from app.signals.selector import (
    RegimePerformanceTracker,
    StrategySelector,
    StrategyWeight,
    signals_conflict,
)
from app.strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)

TS = pd.Timestamp("2024-06-03", tz="UTC")


def sig(
    strategy: str,
    direction: SignalDirection,
    confidence: float = 0.8,
    entry: float = 100.0,
    stop: float | None = None,
    target: float | None = None,
) -> Signal:
    if not direction.is_actionable:
        return Signal.wait(strategy, "XAUUSD", "1D", TS, ("nothing to do",))
    if stop is None:
        stop = 98.0 if direction is SignalDirection.LONG else 102.0
    if target is None:
        target = 104.0 if direction is SignalDirection.LONG else 96.0
    return Signal(
        strategy=strategy, symbol="XAUUSD", timeframe="1D", direction=direction,
        confidence=confidence, timestamp=TS, entry_price=entry,
        stop_loss=stop, take_profit=target,
    )


def regime(kind=RegimeType.TRENDING_UP, confidence=0.8) -> MarketRegime:
    return MarketRegime(
        regime=kind, confidence=confidence,
        volatility=VolatilityState.MEDIUM, trend_strength=0.6, timestamp=TS,
    )


def weights(**values: float) -> dict[str, StrategyWeight]:
    return {
        name: StrategyWeight(
            strategy=name, weight=weight, base_weight=1.0,
            regime_factor=1.0, performance_factor=1.0,
        )
        for name, weight in values.items()
    }


# ------------------------------------------------------------------ agreement

def test_unanimous_longs_produce_a_long():
    aggregator = SignalAggregator(min_confidence=0.3)
    decision = aggregator.aggregate(
        [sig("a", SignalDirection.LONG), sig("b", SignalDirection.LONG)],
        weights=weights(a=0.5, b=0.5),
        regime=regime(),
    )
    assert decision.direction is SignalDirection.LONG
    assert set(decision.contributing) == {"a", "b"}


def test_unanimous_shorts_produce_a_short():
    aggregator = SignalAggregator(min_confidence=0.3)
    decision = aggregator.aggregate(
        [sig("a", SignalDirection.SHORT), sig("b", SignalDirection.SHORT)],
        weights=weights(a=0.5, b=0.5),
    )
    assert decision.direction is SignalDirection.SHORT


def test_no_signals_produces_wait():
    decision = SignalAggregator().aggregate([], symbol="XAUUSD", timeframe="1D")
    assert decision.direction is SignalDirection.WAIT
    assert any("No strategy proposed" in r for r in decision.reasoning)


def test_all_waiting_produces_wait_and_records_each_reason():
    decision = SignalAggregator().aggregate(
        [sig("a", SignalDirection.WAIT), sig("b", SignalDirection.WAIT)]
    )
    assert decision.direction is SignalDirection.WAIT
    joined = " ".join(decision.reasoning)
    assert "a: WAIT" in joined and "b: WAIT" in joined


# ------------------------------------------------------------------- conflict

def test_conflicting_signals_abstain_by_default():
    """A split market is a reason to stand aside, not to average two views."""
    aggregator = SignalAggregator(conflict_policy=ConflictPolicy.ABSTAIN)
    decision = aggregator.aggregate(
        [sig("a", SignalDirection.LONG), sig("b", SignalDirection.SHORT)],
        weights=weights(a=0.5, b=0.5),
    )
    assert decision.direction is SignalDirection.WAIT
    assert any("disagree" in r for r in decision.reasoning)


def test_conflict_reason_names_both_sides():
    decision = SignalAggregator().aggregate(
        [sig("bull", SignalDirection.LONG), sig("bear", SignalDirection.SHORT)],
        weights=weights(bull=0.5, bear=0.5),
    )
    joined = " ".join(decision.reasoning)
    assert "bull" in joined and "bear" in joined


def test_net_score_policy_can_resolve_a_lopsided_conflict():
    aggregator = SignalAggregator(
        conflict_policy=ConflictPolicy.NET_SCORE, min_confidence=0.3, min_score_margin=0.1
    )
    decision = aggregator.aggregate(
        [
            sig("a", SignalDirection.LONG, confidence=0.9),
            sig("b", SignalDirection.LONG, confidence=0.9),
            sig("c", SignalDirection.SHORT, confidence=0.3),
        ],
        weights=weights(a=0.4, b=0.4, c=0.2),
    )
    assert decision.direction is SignalDirection.LONG
    assert decision.opposing == ("c",)


def test_evenly_matched_conflict_still_waits_under_net_score():
    aggregator = SignalAggregator(
        conflict_policy=ConflictPolicy.NET_SCORE, min_score_margin=0.3
    )
    decision = aggregator.aggregate(
        [
            sig("a", SignalDirection.LONG, confidence=0.8),
            sig("b", SignalDirection.SHORT, confidence=0.8),
        ],
        weights=weights(a=0.5, b=0.5),
    )
    assert decision.direction is SignalDirection.WAIT
    assert any("evenly matched" in r for r in decision.reasoning)


def test_signals_conflict_helper():
    assert signals_conflict([sig("a", SignalDirection.LONG), sig("b", SignalDirection.SHORT)])
    assert not signals_conflict([sig("a", SignalDirection.LONG), sig("b", SignalDirection.LONG)])
    assert not signals_conflict([sig("a", SignalDirection.WAIT)])


# --------------------------------------------------------------------- methods

def test_unanimous_method_rejects_any_dissent():
    aggregator = SignalAggregator(
        method=AggregationMethod.UNANIMOUS,
        conflict_policy=ConflictPolicy.NET_SCORE,
        min_confidence=0.1,
        min_score_margin=0.0,
    )
    decision = aggregator.aggregate(
        [
            sig("a", SignalDirection.LONG),
            sig("b", SignalDirection.LONG),
            sig("c", SignalDirection.SHORT, confidence=0.1),
        ],
        weights=weights(a=0.45, b=0.45, c=0.1),
    )
    assert decision.direction is SignalDirection.WAIT
    assert any("UNANIMOUS" in r for r in decision.reasoning)


def test_majority_method_counts_strategies_not_conviction():
    aggregator = SignalAggregator(
        method=AggregationMethod.MAJORITY, min_confidence=0.1, min_score_margin=0.0,
        conflict_policy=ConflictPolicy.NET_SCORE,
    )
    decision = aggregator.aggregate(
        [
            sig("a", SignalDirection.LONG, confidence=0.2),
            sig("b", SignalDirection.LONG, confidence=0.2),
            sig("c", SignalDirection.SHORT, confidence=0.99),
        ],
        weights=weights(a=1 / 3, b=1 / 3, c=1 / 3),
    )
    assert decision.direction is SignalDirection.LONG


def test_weighted_method_lets_conviction_matter():
    aggregator = SignalAggregator(
        method=AggregationMethod.WEIGHTED, min_confidence=0.1, min_score_margin=0.0,
        conflict_policy=ConflictPolicy.NET_SCORE,
    )
    decision = aggregator.aggregate(
        [
            sig("a", SignalDirection.LONG, confidence=0.2),
            sig("b", SignalDirection.SHORT, confidence=0.95),
        ],
        weights=weights(a=0.5, b=0.5),
    )
    assert decision.direction is SignalDirection.SHORT


# ------------------------------------------------------------------ thresholds

def test_low_combined_confidence_waits():
    aggregator = SignalAggregator(min_confidence=0.9)
    decision = aggregator.aggregate(
        [sig("a", SignalDirection.LONG, confidence=0.5)], weights=weights(a=1.0)
    )
    assert decision.direction is SignalDirection.WAIT
    assert any("below the" in r for r in decision.reasoning)


def test_suppressed_strategies_leave_no_opinion():
    """A strategy silenced by the selector must not vote."""
    aggregator = SignalAggregator(min_confidence=0.1)
    decision = aggregator.aggregate(
        [sig("a", SignalDirection.LONG)], weights=weights(a=0.0)
    )
    assert decision.direction is SignalDirection.WAIT
    assert any("suppressed" in r for r in decision.reasoning)


# ---------------------------------------------------------------------- levels

def test_combined_stop_is_the_most_conservative():
    """Taking the widest stop would silently increase everyone's intended risk."""
    aggregator = SignalAggregator(min_confidence=0.1)
    decision = aggregator.aggregate(
        [
            sig("a", SignalDirection.LONG, entry=100.0, stop=95.0),
            sig("b", SignalDirection.LONG, entry=100.0, stop=98.0),
        ],
        weights=weights(a=0.5, b=0.5),
    )
    assert decision.stop_loss == pytest.approx(98.0)


def test_combined_short_stop_is_the_tightest():
    aggregator = SignalAggregator(min_confidence=0.1)
    decision = aggregator.aggregate(
        [
            sig("a", SignalDirection.SHORT, entry=100.0, stop=105.0),
            sig("b", SignalDirection.SHORT, entry=100.0, stop=102.0),
        ],
        weights=weights(a=0.5, b=0.5),
    )
    assert decision.stop_loss == pytest.approx(102.0)


def test_combined_target_is_the_nearest():
    aggregator = SignalAggregator(min_confidence=0.1)
    decision = aggregator.aggregate(
        [
            sig("a", SignalDirection.LONG, target=110.0),
            sig("b", SignalDirection.LONG, target=104.0),
        ],
        weights=weights(a=0.5, b=0.5),
    )
    assert decision.take_profit == pytest.approx(104.0)


def test_aggregated_decision_converts_to_a_valid_signal():
    aggregator = SignalAggregator(min_confidence=0.1)
    decision = aggregator.aggregate(
        [sig("a", SignalDirection.LONG), sig("b", SignalDirection.LONG)],
        weights=weights(a=0.5, b=0.5),
    )
    signal = decision.to_signal()

    assert signal.strategy == "ensemble"
    assert signal.is_actionable
    assert signal.stop_loss < signal.entry_price   # validated by the Signal contract


def test_wait_decision_converts_to_a_wait_signal():
    decision = SignalAggregator().aggregate([], symbol="XAUUSD", timeframe="1D")
    assert decision.to_signal().direction is SignalDirection.WAIT


def test_decision_describes_itself():
    aggregator = SignalAggregator(min_confidence=0.1)
    text = aggregator.aggregate(
        [sig("a", SignalDirection.LONG)], weights=weights(a=1.0)
    ).describe()
    assert "FINAL DECISION" in text and "Reasons:" in text


# ----------------------------------------------------------------- the selector

def test_preferred_regime_outweighs_neutral():
    selector = StrategySelector()
    result = selector.select(
        [TrendFollowingStrategy(), MomentumStrategy()], regime(RegimeType.TRENDING_UP)
    )
    assert all(w.weight > 0 for w in result.values())
    assert sum(w.weight for w in result.values()) == pytest.approx(1.0)


def test_hostile_regime_suppresses_a_strategy():
    """Mean reversion in a strong trend is not a weak opinion, it is a wrong one."""
    selector = StrategySelector(suppress_avoided=True)
    result = selector.select(
        [TrendFollowingStrategy(), MeanReversionStrategy()], regime(RegimeType.TRENDING_UP)
    )
    assert result["mean_reversion"].weight == 0.0
    assert result["trend_following"].weight > 0


def test_weights_sum_to_one_when_any_survive():
    selector = StrategySelector()
    result = selector.select(
        [TrendFollowingStrategy(), MomentumStrategy(), MeanReversionStrategy()],
        regime(RegimeType.RANGING),
    )
    total = sum(w.weight for w in result.values())
    assert total == pytest.approx(1.0)


def test_all_suppressed_yields_zero_weights():
    selector = StrategySelector(suppress_avoided=True)
    result = selector.select([MeanReversionStrategy()], regime(RegimeType.TRENDING_UP))
    assert sum(w.weight for w in result.values()) == 0.0


def test_a_configured_standing_weight_tilts_the_selection():
    """The lever the phase-4 redundancy experiments pull.

    Testing whether down-weighting a correlated strategy helps requires the
    weight to actually reach the selector; without this the whole comparison
    would be between identical configurations.
    """
    from app.config.models import StrategyConfig

    selector = StrategySelector()
    full = selector.select(
        [TrendFollowingStrategy(), MomentumStrategy()], regime(RegimeType.TRENDING_UP)
    )
    halved = selector.select(
        [
            TrendFollowingStrategy(),
            MomentumStrategy(config=StrategyConfig(weight=0.5)),
        ],
        regime(RegimeType.TRENDING_UP),
    )

    assert halved["momentum"].base_weight == 0.5
    assert halved["momentum"].weight < full["momentum"].weight
    assert halved["trend_following"].weight > full["trend_following"].weight
    assert sum(w.weight for w in halved.values()) == pytest.approx(1.0)


def test_a_zero_standing_weight_silences_a_strategy():
    from app.config.models import StrategyConfig

    result = StrategySelector().select(
        [
            TrendFollowingStrategy(),
            MomentumStrategy(config=StrategyConfig(weight=0.0)),
        ],
        regime(RegimeType.TRENDING_UP),
    )
    assert result["momentum"].weight == 0.0
    assert result["trend_following"].weight == pytest.approx(1.0)


def test_the_default_standing_weight_changes_nothing():
    selector = StrategySelector()
    result = selector.select([TrendFollowingStrategy()], regime(RegimeType.TRENDING_UP))
    assert result["trend_following"].base_weight == 1.0


def test_selector_records_its_reasoning():
    selector = StrategySelector()
    result = selector.select([TrendFollowingStrategy()], regime(RegimeType.TRENDING_UP))
    assert result["trend_following"].reasoning


# ------------------------------------------------------- performance tracking

def test_performance_needs_a_minimum_sample():
    """Two lucky trades are not evidence, and treating them as such overfits."""
    tracker = RegimePerformanceTracker(min_samples=10)
    for _ in range(3):
        tracker.record_trade("momentum", "TRENDING_UP", 2.0)

    assert tracker.expectancy("momentum", "TRENDING_UP") is None
    assert tracker.sample_count("momentum", "TRENDING_UP") == 3


def test_performance_is_used_once_the_sample_is_large_enough():
    tracker = RegimePerformanceTracker(min_samples=10)
    for _ in range(12):
        tracker.record_trade("momentum", "TRENDING_UP", 1.0)

    assert tracker.expectancy("momentum", "TRENDING_UP") == pytest.approx(1.0)


def test_losing_strategy_is_down_weighted_in_that_regime():
    tracker = RegimePerformanceTracker(min_samples=5)
    for _ in range(10):
        tracker.record_trade("momentum", "TRENDING_UP", -1.0)

    selector = StrategySelector(tracker)
    result = selector.select(
        [MomentumStrategy(), TrendFollowingStrategy()], regime(RegimeType.TRENDING_UP)
    )
    assert result["momentum"].performance_factor < result["trend_following"].performance_factor


def test_tracker_ignores_missing_r_multiples():
    tracker = RegimePerformanceTracker()
    tracker.record_trade("momentum", "RANGING", None)
    assert tracker.sample_count("momentum", "RANGING") == 0


def test_tracker_produces_the_regime_performance_table():
    tracker = RegimePerformanceTracker()
    for r in (1.0, -0.5, 2.0):
        tracker.record_trade("momentum", "TRENDING_UP", r)

    frame = tracker.to_frame()
    row = frame.iloc[0]
    assert row["strategy"] == "momentum"
    assert row["trades"] == 3
    assert row["mean_r"] == pytest.approx((1.0 - 0.5 + 2.0) / 3)


def test_strategy_config_min_confidence_is_independent_of_the_aggregator():
    strategy = TrendFollowingStrategy(StrategyConfig(min_confidence=0.9))
    assert strategy.config.min_confidence == 0.9
