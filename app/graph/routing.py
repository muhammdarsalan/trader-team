"""Conditional routing.

Each function inspects the state and names the next node. Routing is what lets
the graph stop early - and stopping early matters: once the aggregator has said
WAIT there is nothing for the risk engine to size, and running it anyway would
produce a meaningless decision record.
"""

from __future__ import annotations

from app.graph.state import TradingState

# Node names, shared between the routers and the workflow builder.
NODE_FEATURES = "features"
NODE_REGIME = "regime"
NODE_SELECTION = "selection"
NODE_AGGREGATION = "aggregation"
NODE_RISK = "risk"
NODE_ORDER = "order"
NODE_FINALISE = "finalise"


def route_after_features(state: TradingState) -> str:
    """Stop if features could not be computed.

    Every downstream node reads features, so continuing produces a cascade of
    identical failures and an unreadable error list.
    """
    features = state.get("features")
    if features is None or features.is_empty:
        return NODE_FINALISE
    return NODE_REGIME


def route_after_aggregation(state: TradingState) -> str:
    """Only consult the risk engine when there is a trade to size."""
    aggregated = state.get("aggregated")
    if aggregated is None or not aggregated.is_actionable:
        return NODE_FINALISE
    return NODE_RISK


def route_after_risk(state: TradingState) -> str:
    """Create an order only if risk approved one."""
    decision = state.get("risk_decision")
    if decision is None or not decision.approved:
        return NODE_FINALISE
    return NODE_ORDER
