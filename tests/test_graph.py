"""The decision graph: structure, routing, error isolation and explainability."""

from __future__ import annotations

import pandas as pd
import pytest

from app.config.loader import get_config
from app.data.schema import MarketData, empty_frame
from app.features.engine import FeatureEngine
from app.graph.routing import (
    NODE_FINALISE,
    NODE_ORDER,
    NODE_REGIME,
    NODE_RISK,
    route_after_aggregation,
    route_after_features,
    route_after_risk,
)
from app.graph.state import explain, new_state, state_summary
from app.graph.workflow import TradingGraph
from app.portfolio.portfolio import Portfolio
from app.regimes.models import MarketRegime, RegimeType, VolatilityState
from app.risk.models import RiskBlockReason, RiskDecision, RiskVerdict
from app.signals.aggregator import AggregatedDecision
from app.signals.models import Signal, SignalDirection
from tests.conftest import make_ohlcv


@pytest.fixture
def config(config_dir):
    return get_config(config_dir)


@pytest.fixture
def market_data() -> MarketData:
    return MarketData(
        symbol="XAUUSD", timeframe="1D", df=make_ohlcv(periods=600, seed=77),
        provider="synthetic",
    )


def build(config, portfolio=None, **kwargs) -> TradingGraph:
    return TradingGraph(
        config=config,
        portfolio=portfolio or Portfolio(10_000.0),
        asset=config.assets.get("XAUUSD"),
        **kwargs,
    )


# ------------------------------------------------------------------ structure

def test_graph_contains_every_stage(config):
    names = build(config).node_names()
    for expected in ("features", "regime", "selection", "aggregation", "risk", "order", "finalise"):
        assert expected in names


def test_every_enabled_strategy_gets_its_own_node(config):
    names = build(config).node_names()
    strategy_nodes = [n for n in names if n.startswith("strategy_")]
    assert len(strategy_nodes) == len(config.strategies.enabled_names())


def test_graph_refuses_to_build_with_no_strategies(config):
    from app.config.models import StrategiesConfig

    empty_config = config.__class__(
        **{
            **{f: getattr(config, f) for f in config.__dataclass_fields__},
            "strategies": StrategiesConfig(strategies={}),
        }
    )
    with pytest.raises(ValueError, match="No strategies are enabled"):
        build(empty_config)


def test_graph_renders_as_mermaid(config):
    mermaid = build(config).to_mermaid()
    assert "features" in mermaid and "aggregation" in mermaid


# -------------------------------------------------------------------- routing

def test_route_stops_when_features_are_missing():
    assert route_after_features({}) == NODE_FINALISE


def test_route_stops_when_features_are_empty():
    features = FeatureEngine().compute(
        MarketData(symbol="X", timeframe="1D", df=empty_frame(), provider="test")
    )
    assert route_after_features({"features": features}) == NODE_FINALISE


def test_route_continues_with_usable_features(market_data):
    features = FeatureEngine().compute(market_data)
    assert route_after_features({"features": features}) == NODE_REGIME


def test_route_skips_risk_when_the_ensemble_waits():
    """Nothing to size means the risk engine has nothing to say."""
    decision = AggregatedDecision(
        direction=SignalDirection.WAIT, confidence=0.0, symbol="X", timeframe="1D"
    )
    assert route_after_aggregation({"aggregated": decision}) == NODE_FINALISE


def test_route_reaches_risk_for_an_actionable_decision():
    decision = AggregatedDecision(
        direction=SignalDirection.LONG, confidence=0.8, symbol="X", timeframe="1D",
        entry_price=100.0, stop_loss=98.0,
    )
    assert route_after_aggregation({"aggregated": decision}) == NODE_RISK


def test_route_skips_order_when_risk_refuses():
    decision = RiskDecision.reject(RiskBlockReason.MAX_DRAWDOWN, "too deep")
    assert route_after_risk({"risk_decision": decision}) == NODE_FINALISE


def test_route_creates_an_order_when_risk_approves():
    decision = RiskDecision(verdict=RiskVerdict.APPROVED, quantity=5.0)
    assert route_after_risk({"risk_decision": decision}) == NODE_ORDER


def test_route_handles_missing_state_gracefully():
    assert route_after_aggregation({}) == NODE_FINALISE
    assert route_after_risk({}) == NODE_FINALISE


# ----------------------------------------------------------------- execution

def test_graph_runs_end_to_end(config, market_data):
    graph = build(config)
    features = FeatureEngine(config.features).compute(market_data)
    timestamp = features.df.index[-1]

    state = graph.run(
        symbol="XAUUSD", timeframe="1D", timestamp=timestamp,
        market_data=market_data, equity=10_000.0, features=features.upto(timestamp),
    )

    assert state.get("regime") is not None
    assert state.get("decision")
    assert len(state["strategy_signals"]) == len(config.strategies.enabled_names())


def test_all_strategies_contribute_a_signal(config, market_data):
    graph = build(config)
    features = FeatureEngine(config.features).compute(market_data)
    timestamp = features.df.index[-1]

    state = graph.run(
        "XAUUSD", "1D", timestamp, market_data, 10_000.0, features.upto(timestamp)
    )

    assert set(state["strategy_signals"]) == set(config.strategies.enabled_names())


def test_kill_switch_prevents_orders(config, market_data):
    """The graph still analyses; nothing is ordered."""
    graph = build(config, trading_enabled=False)
    features = FeatureEngine(config.features).compute(market_data)
    timestamp = features.df.index[-1]

    state = graph.run(
        "XAUUSD", "1D", timestamp, market_data, 10_000.0, features.upto(timestamp)
    )

    assert state.get("order") is None
    assert state.get("regime") is not None   # analysis still happened


def test_precomputed_features_are_used_unchanged(config, market_data):
    graph = build(config)
    features = FeatureEngine(config.features).compute(market_data)
    timestamp = features.df.index[-1]

    state = graph.run(
        "XAUUSD", "1D", timestamp, market_data, 10_000.0, features.upto(timestamp)
    )
    assert any("supplied by caller" in entry for entry in state["trace"])


def test_graph_records_a_trace(config, market_data):
    graph = build(config)
    features = FeatureEngine(config.features).compute(market_data)
    timestamp = features.df.index[-1]

    state = graph.run(
        "XAUUSD", "1D", timestamp, market_data, 10_000.0, features.upto(timestamp)
    )
    assert len(state["trace"]) >= 4


# ------------------------------------------------------------ error isolation

def test_one_failing_strategy_does_not_stop_the_others(config, market_data):
    """Section 48: the graph continues and the failure is recorded."""
    from app.strategies import MomentumStrategy, TrendFollowingStrategy

    class ExplodingStrategy(TrendFollowingStrategy):
        metadata = TrendFollowingStrategy.metadata

        def _generate(self, ctx):
            raise RuntimeError("deliberate failure")

    strategies = [ExplodingStrategy(), MomentumStrategy()]
    graph = build(config, strategies=strategies)

    features = FeatureEngine(config.features).compute(market_data)
    timestamp = features.df.index[-1]
    state = graph.run(
        "XAUUSD", "1D", timestamp, market_data, 10_000.0, features.upto(timestamp)
    )

    assert state["errors"], "the failure should be recorded"
    assert any("deliberate failure" in e["error"] for e in state["errors"])
    # The surviving strategy still produced a signal and the run completed.
    assert "momentum" in state["strategy_signals"]
    assert state.get("decision")


def test_failed_strategy_is_absent_rather_than_guessed(config, market_data):
    from app.strategies import MomentumStrategy, TrendFollowingStrategy

    class ExplodingStrategy(TrendFollowingStrategy):
        metadata = TrendFollowingStrategy.metadata

        def _generate(self, ctx):
            raise ValueError("boom")

    graph = build(config, strategies=[ExplodingStrategy(), MomentumStrategy()])
    features = FeatureEngine(config.features).compute(market_data)
    timestamp = features.df.index[-1]
    state = graph.run(
        "XAUUSD", "1D", timestamp, market_data, 10_000.0, features.upto(timestamp)
    )

    assert "trend_following" not in state["strategy_signals"]


# ------------------------------------------------------------- explainability

def test_explain_reports_actual_state():
    """Section 66: the explanation must come from state, not from fabricated text."""
    state = new_state(
        "XAUUSD", "1D", pd.Timestamp("2024-01-01", tz="UTC"),
        MarketData(symbol="XAUUSD", timeframe="1D", df=make_ohlcv(50), provider="test"),
        10_000.0,
    )
    state["regime"] = MarketRegime(
        regime=RegimeType.TRENDING_UP, confidence=0.81,
        volatility=VolatilityState.MEDIUM, trend_strength=0.7,
    )
    state["strategy_signals"] = {
        "trend_following": Signal(
            strategy="trend_following", symbol="XAUUSD", timeframe="1D",
            direction=SignalDirection.LONG, confidence=0.76,
            entry_price=100.0, stop_loss=98.0,
        ),
        "mean_reversion": Signal.wait("mean_reversion", "XAUUSD", "1D"),
    }
    state["decision"] = "ORDER_LONG"

    text = explain(state)
    assert "TRENDING_UP" in text
    assert "trend_following: LONG" in text
    assert "mean_reversion: WAIT" in text
    assert "ORDER_LONG" in text


def test_state_summary_is_serialisable(config, market_data):
    graph = build(config)
    features = FeatureEngine(config.features).compute(market_data)
    timestamp = features.df.index[-1]
    state = graph.run(
        "XAUUSD", "1D", timestamp, market_data, 10_000.0, features.upto(timestamp)
    )

    summary = state_summary(state)
    assert summary["symbol"] == "XAUUSD"
    assert isinstance(summary["signals"], dict)
    assert summary["decision"]
