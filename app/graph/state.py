"""The graph's shared state.

A single typed structure passed between nodes, rather than an untyped dict that
every node mutates in its own way. Section 65 of the brief asks for exactly
this, and the reason is practical: with eight nodes and five strategies, an
untyped state becomes unauditable within a week.

The state is also the decision record. Everything needed to explain *why* the
graph reached its conclusion - regime, every signal, weights, the aggregate,
the risk verdict, and any node errors - is present at the end of a run.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pandas as pd

from app.data.schema import MarketData
from app.execution.models import Fill, Order
from app.features.engine import FeatureSet
from app.regimes.models import MarketRegime
from app.risk.models import RiskDecision
from app.signals.aggregator import AggregatedDecision
from app.signals.models import Signal
from app.signals.selector import StrategyWeight


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Reducer for state keys written by concurrently-running nodes.

    Strategy nodes run in parallel and each writes one entry; without a reducer
    LangGraph rejects the concurrent update.
    """
    return {**(left or {}), **(right or {})}


def append_lists(left: list | None, right: list | None) -> list:
    """Reducer that concatenates, used for errors and log entries."""
    return [*(left or []), *(right or [])]


class NodeError(TypedDict):
    """A recorded failure in one node. The graph continues around it."""

    node: str
    error: str
    error_type: str


class TradingState(TypedDict, total=False):
    """State shared across the decision graph.

    ``total=False`` because nodes populate it progressively; only ``symbol``,
    ``timeframe`` and ``timestamp`` are present from the start.
    """

    # --- inputs -----------------------------------------------------------
    symbol: str
    timeframe: str
    timestamp: pd.Timestamp
    market_data: MarketData
    equity: float

    # --- computed by nodes -------------------------------------------------
    features: FeatureSet
    regime: MarketRegime
    strategy_signals: Annotated[dict[str, Signal], merge_dicts]
    strategy_weights: dict[str, StrategyWeight]
    aggregated: AggregatedDecision
    final_signal: Signal
    risk_decision: RiskDecision
    order: Order
    fill: Fill

    # --- bookkeeping --------------------------------------------------------
    errors: Annotated[list[NodeError], append_lists]
    trace: Annotated[list[str], append_lists]
    decision: str
    halted_at: str


def new_state(
    symbol: str,
    timeframe: str,
    timestamp: pd.Timestamp,
    market_data: MarketData,
    equity: float,
    features: FeatureSet | None = None,
) -> TradingState:
    """Fresh state for one bar.

    ``features`` may be supplied by a caller that has already computed them -
    the backtester does, once for the whole series - in which case the feature
    node passes them through untouched.
    """
    state = TradingState(
        symbol=symbol.upper(),
        timeframe=timeframe,
        timestamp=timestamp,
        market_data=market_data,
        equity=equity,
        strategy_signals={},
        errors=[],
        trace=[],
    )
    if features is not None:
        state["features"] = features
    return state


def explain(state: TradingState) -> str:
    """Human-readable account of how the graph reached its decision.

    Built entirely from recorded state. Nothing here is generated text about
    what the system might have been thinking - section 66 of the brief requires
    the explanation to come from actual state, and this is that requirement
    implemented rather than asserted.
    """
    lines: list[str] = []
    symbol = state.get("symbol", "?")
    timestamp = state.get("timestamp")
    lines.append(f"{symbol} {state.get('timeframe', '')} at {timestamp}")

    regime = state.get("regime")
    if regime is not None:
        lines.append(f"Regime: {regime.regime} (confidence {regime.confidence:.2f})")

    signals = state.get("strategy_signals") or {}
    if signals:
        lines.append("Strategies:")
        weights = state.get("strategy_weights") or {}
        for name in sorted(signals):
            signal = signals[name]
            weight = weights.get(name)
            weight_note = f", weight {weight.weight:.2f}" if weight else ""
            detail = (
                f"{signal.direction} at {signal.confidence:.2f}"
                if signal.is_actionable
                else "WAIT"
            )
            lines.append(f"  {name}: {detail}{weight_note}")

    aggregated = state.get("aggregated")
    if aggregated is not None:
        lines.append(f"Aggregate: {aggregated.direction}")
        for reason in aggregated.reasoning:
            lines.append(f"  - {reason}")

    risk = state.get("risk_decision")
    if risk is not None:
        lines.append(f"Risk: {risk.describe()}")

    fill = state.get("fill")
    if fill is not None:
        lines.append(
            f"Fill: {fill.filled_quantity:.6g} @ {fill.price:.6g} "
            f"(costs {fill.total_cost:.2f})"
        )

    errors = state.get("errors") or []
    if errors:
        lines.append("Errors:")
        lines.extend(f"  {e['node']}: {e['error']}" for e in errors)

    lines.append(f"Decision: {state.get('decision', 'UNKNOWN')}")
    return "\n".join(lines)


def state_summary(state: TradingState) -> dict[str, Any]:
    """Serialisable summary, for logging and experiment records."""
    regime = state.get("regime")
    aggregated = state.get("aggregated")
    risk = state.get("risk_decision")
    return {
        "symbol": state.get("symbol"),
        "timeframe": state.get("timeframe"),
        "timestamp": str(state.get("timestamp")),
        "regime": str(regime.regime) if regime else None,
        "regime_confidence": regime.confidence if regime else None,
        "signals": {
            name: str(signal.direction)
            for name, signal in (state.get("strategy_signals") or {}).items()
        },
        "aggregated": str(aggregated.direction) if aggregated else None,
        "risk_verdict": str(risk.verdict) if risk else None,
        "decision": state.get("decision"),
        "errors": state.get("errors") or [],
    }
