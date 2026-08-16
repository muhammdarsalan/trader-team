"""Signals: the standard contract, aggregation and strategy selection."""

from app.signals.aggregator import (
    AggregatedDecision,
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

__all__ = [
    "AggregatedDecision",
    "AggregationMethod",
    "ConflictPolicy",
    "RegimePerformanceTracker",
    "Signal",
    "SignalAggregator",
    "SignalDirection",
    "StrategySelector",
    "StrategyWeight",
    "signals_conflict",
]
