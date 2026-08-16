"""Graph nodes.

Each node takes the shared state, does one job, and returns only the keys it
changed. Nodes never raise: a failure is caught, recorded in ``errors``, and
the graph continues. Section 48 of the brief requires that one strategy
failing cannot bring down the run, and the same principle is applied to every
node rather than only to strategies.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import pandas as pd

from app.config.models import AssetConfig
from app.execution.models import Order, OrderSide
from app.execution.simulator import ExecutionSimulator
from app.features.engine import FeatureEngine
from app.graph.state import TradingState
from app.portfolio.portfolio import Portfolio
from app.regimes.detector import RegimeDetector
from app.regimes.models import MarketRegime, RegimeType, VolatilityState
from app.risk.engine import RiskEngine
from app.risk.models import RiskBlockReason, RiskDecision
from app.signals.aggregator import SignalAggregator
from app.signals.models import Signal, SignalDirection
from app.signals.selector import StrategySelector
from app.strategies.base import Strategy
from app.utils.logging import get_logger

logger = get_logger(__name__)


def isolated(node_name: str) -> Callable:
    """Decorator: never let a node's failure escape.

    A raised exception is recorded in state and the graph proceeds. The
    alternative - one bad indicator killing an entire backtest at bar 4,000 -
    loses the run and tells you very little.
    """

    def decorator(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        @wraps(func)
        def wrapper(state: TradingState, *args: Any, **kwargs: Any) -> dict[str, Any]:
            try:
                return func(state, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                logger.exception(
                    "Node failed", extra={"node": node_name, "symbol": state.get("symbol")}
                )
                return {
                    "errors": [
                        {"node": node_name, "error": str(exc), "error_type": type(exc).__name__}
                    ],
                    "trace": [f"{node_name}: ERROR ({type(exc).__name__})"],
                }

        return wrapper

    return decorator


# --------------------------------------------------------------------- nodes


def make_feature_node(engine: FeatureEngine, asset: AssetConfig | None) -> Callable:
    """Compute the feature panel for the bar."""

    @isolated("features")
    def node(state: TradingState) -> dict[str, Any]:
        # The backtester precomputes features once over the whole series and
        # injects a per-bar truncated view, so recomputing here would turn an
        # O(n) run into O(n^2) for an identical result.
        existing = state.get("features")
        if existing is not None:
            return {"trace": ["features: supplied by caller"]}

        features = engine.compute(state["market_data"], asset)
        return {"features": features, "trace": ["features: computed"]}

    return node


def make_regime_node(detector: RegimeDetector) -> Callable:
    """Classify the market regime at the bar."""

    @isolated("regime")
    def node(state: TradingState) -> dict[str, Any]:
        features = state.get("features")
        if features is None:
            return {
                "regime": _unknown_regime(state.get("timestamp")),
                "trace": ["regime: skipped (no features)"],
            }
        regime = detector.detect(features, state["timestamp"])
        return {"regime": regime, "trace": [f"regime: {regime.regime}"]}

    return node


def make_strategy_node(strategy: Strategy) -> Callable:
    """Run one strategy. Registered once per strategy so they can run in parallel."""
    node_name = f"strategy_{strategy.name}"

    @isolated(node_name)
    def node(state: TradingState) -> dict[str, Any]:
        features = state.get("features")
        regime = state.get("regime")

        if features is None or regime is None:
            signal = Signal.wait(
                strategy.name,
                state["symbol"],
                state["timeframe"],
                state.get("timestamp"),
                ("Features or regime unavailable; strategy could not run",),
            )
        else:
            signal = strategy.generate_signal(
                state["market_data"], features, regime, state["timestamp"]
            )

        return {
            "strategy_signals": {strategy.name: signal},
            "trace": [f"{node_name}: {signal.direction}"],
        }

    return node


def make_selection_node(selector: StrategySelector, strategies: list[Strategy]) -> Callable:
    """Weight the strategies for the current regime."""

    @isolated("selection")
    def node(state: TradingState) -> dict[str, Any]:
        regime = state.get("regime") or _unknown_regime(state.get("timestamp"))
        weights = selector.select(strategies, regime)
        active = [name for name, w in weights.items() if w.weight > 0]
        return {
            "strategy_weights": weights,
            "trace": [f"selection: {len(active)} strategies active"],
        }

    return node


def make_aggregation_node(aggregator: SignalAggregator) -> Callable:
    """Combine the strategy signals into one decision."""

    @isolated("aggregation")
    def node(state: TradingState) -> dict[str, Any]:
        signals = list((state.get("strategy_signals") or {}).values())
        decision = aggregator.aggregate(
            signals,
            weights=state.get("strategy_weights"),
            regime=state.get("regime"),
            symbol=state["symbol"],
            timeframe=state["timeframe"],
            timestamp=state.get("timestamp"),
        )
        return {
            "aggregated": decision,
            "final_signal": decision.to_signal(),
            "trace": [f"aggregation: {decision.direction}"],
        }

    return node


def make_risk_node(
    engine: RiskEngine,
    portfolio: Portfolio,
    asset: AssetConfig | None,
    correlations_provider: Callable[[], pd.DataFrame | None] | None = None,
) -> Callable:
    """Size the position, or refuse it."""

    @isolated("risk")
    def node(state: TradingState) -> dict[str, Any]:
        signal = state.get("final_signal")
        if signal is None:
            decision = RiskDecision.reject(
                RiskBlockReason.NO_SIGNAL, "Aggregation produced no signal"
            )
            return {"risk_decision": decision, "trace": ["risk: no signal"]}

        correlations = correlations_provider() if correlations_provider else None
        decision = engine.evaluate(
            signal,
            portfolio,
            equity=state.get("equity", portfolio.initial_balance),
            asset=asset,
            correlations=correlations,
        )
        return {
            "risk_decision": decision,
            "trace": [f"risk: {decision.verdict}"],
        }

    return node


def make_order_node(asset: AssetConfig | None = None) -> Callable:  # noqa: ARG001
    """Build the order the risk engine approved.

    Building an order is all this node does. The fill happens against the
    *next* bar, which only the backtester or paper-trading loop can supply -
    the graph deliberately cannot fill an order using the bar it is analysing.
    """

    @isolated("order")
    def node(state: TradingState) -> dict[str, Any]:
        signal = state.get("final_signal")
        decision = state.get("risk_decision")

        if signal is None or decision is None or not decision.approved:
            return {"decision": "NO_TRADE", "trace": ["order: not created"]}

        order = Order(
            symbol=signal.symbol,
            side=OrderSide.from_direction(signal.direction),
            quantity=decision.quantity,
            created_at=state.get("timestamp"),
            strategy=signal.strategy,
            reference_price=signal.entry_price,
            metadata={
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "risk_amount": decision.risk_amount,
                "regime": str(state["regime"].regime) if state.get("regime") else "UNKNOWN",
            },
        )
        return {
            "order": order,
            "decision": f"ORDER_{signal.direction}",
            "trace": [f"order: {order.side} {order.quantity:.6g}"],
        }

    return node


@isolated("finalise")
def finalise_node(state: TradingState) -> dict[str, Any]:
    """Record the outcome when no order was created."""
    if state.get("decision"):
        return {"trace": ["finalise: decision already recorded"]}

    risk = state.get("risk_decision")
    aggregated = state.get("aggregated")

    if risk is not None and not risk.approved:
        outcome = f"BLOCKED_{risk.block_reason}"
    elif aggregated is not None and not aggregated.is_actionable:
        outcome = "WAIT"
    else:
        outcome = "NO_TRADE"

    return {"decision": outcome, "trace": [f"finalise: {outcome}"]}


# ------------------------------------------------------------------- helpers


def _unknown_regime(timestamp: pd.Timestamp | None) -> MarketRegime:
    return MarketRegime(
        regime=RegimeType.UNCERTAIN,
        confidence=0.0,
        volatility=VolatilityState.UNKNOWN,
        trend_strength=0.0,
        timestamp=timestamp,
        reasoning=("Regime detection did not run",),
    )


def build_execution_simulator(
    asset: AssetConfig | None, execution_config: Any
) -> ExecutionSimulator:
    """Convenience constructor used by the backtester and paper loop."""
    return ExecutionSimulator(config=execution_config, asset=asset)


__all__ = [
    "build_execution_simulator",
    "finalise_node",
    "isolated",
    "make_aggregation_node",
    "make_feature_node",
    "make_order_node",
    "make_regime_node",
    "make_risk_node",
    "make_selection_node",
    "make_strategy_node",
    "SignalDirection",
]
