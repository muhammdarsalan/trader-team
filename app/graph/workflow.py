"""The decision graph.

Assembles the nodes into a LangGraph ``StateGraph``:

    START -> features -> regime -> [strategy fan-out] -> selection
          -> aggregation -> risk -> order -> END

Strategies fan out in parallel from the regime node and fan back in at
selection. Each writes one entry into ``strategy_signals``, merged by the
reducer on that key.

The graph deliberately stops at *order creation*. It never fills anything,
because a fill requires the next bar's prices and the graph only ever sees the
bar it is analysing. Letting it fill would put look-ahead bias into the
architecture itself, where no test could later remove it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from langgraph.graph import END, START, StateGraph

from app.config.loader import AppConfig
from app.config.models import AssetConfig
from app.features.engine import FeatureEngine
from app.graph import nodes as node_factory
from app.graph.routing import (
    NODE_AGGREGATION,
    NODE_FEATURES,
    NODE_FINALISE,
    NODE_ORDER,
    NODE_REGIME,
    NODE_RISK,
    NODE_SELECTION,
    route_after_aggregation,
    route_after_features,
    route_after_risk,
)
from app.graph.state import TradingState, new_state
from app.portfolio.portfolio import Portfolio
from app.regimes.detector import RegimeDetector
from app.risk.engine import RiskEngine
from app.signals.aggregator import SignalAggregator
from app.signals.selector import RegimePerformanceTracker, StrategySelector
from app.strategies.base import Strategy
from app.strategies.registry import build_enabled_strategies
from app.utils.logging import get_logger

logger = get_logger(__name__)


class TradingGraph:
    """The compiled decision graph plus the components it drives."""

    def __init__(
        self,
        config: AppConfig,
        portfolio: Portfolio,
        asset: AssetConfig | None = None,
        strategies: list[Strategy] | None = None,
        performance: RegimePerformanceTracker | None = None,
        correlations_provider: Callable[[], pd.DataFrame | None] | None = None,
        trading_enabled: bool | None = None,
        data_quality: str | None = None,
    ) -> None:
        self.config = config
        self.portfolio = portfolio
        self.asset = asset
        self.data_quality = data_quality

        self.strategies = strategies or build_enabled_strategies(config.strategies)
        if not self.strategies:
            raise ValueError(
                "No strategies are enabled. Enable at least one in configs/strategies.yaml."
            )

        self.feature_engine = FeatureEngine(config.features)
        self.regime_detector = RegimeDetector(config.regimes)
        self.performance = performance or RegimePerformanceTracker()
        self.selector = StrategySelector(self.performance)
        self.aggregator = SignalAggregator()
        self.risk_engine = RiskEngine(
            config.risk,
            trading_enabled=(
                config.platform.trading_enabled if trading_enabled is None else trading_enabled
            ),
            data_quality=data_quality,
        )
        self.correlations_provider = correlations_provider

        self._graph = self._build()

    # ------------------------------------------------------------------ build

    def _build(self):
        graph = StateGraph(TradingState)

        graph.add_node(NODE_FEATURES, node_factory.make_feature_node(self.feature_engine, self.asset))
        graph.add_node(NODE_REGIME, node_factory.make_regime_node(self.regime_detector))

        strategy_nodes = []
        for strategy in self.strategies:
            name = f"strategy_{strategy.name}"
            graph.add_node(name, node_factory.make_strategy_node(strategy))
            strategy_nodes.append(name)

        graph.add_node(
            NODE_SELECTION, node_factory.make_selection_node(self.selector, self.strategies)
        )
        graph.add_node(NODE_AGGREGATION, node_factory.make_aggregation_node(self.aggregator))
        graph.add_node(
            NODE_RISK,
            node_factory.make_risk_node(
                self.risk_engine, self.portfolio, self.asset, self.correlations_provider
            ),
        )
        graph.add_node(NODE_ORDER, node_factory.make_order_node(self.asset))
        graph.add_node(NODE_FINALISE, node_factory.finalise_node)

        graph.add_edge(START, NODE_FEATURES)

        # Bail out early if features failed; every later node depends on them.
        graph.add_conditional_edges(
            NODE_FEATURES, route_after_features, {NODE_REGIME: NODE_REGIME, NODE_FINALISE: NODE_FINALISE}
        )

        # Fan out to every strategy, then fan back in at selection. LangGraph
        # runs them concurrently and merges their writes via the state reducer.
        for name in strategy_nodes:
            graph.add_edge(NODE_REGIME, name)
            graph.add_edge(name, NODE_SELECTION)

        graph.add_edge(NODE_SELECTION, NODE_AGGREGATION)

        graph.add_conditional_edges(
            NODE_AGGREGATION,
            route_after_aggregation,
            {NODE_RISK: NODE_RISK, NODE_FINALISE: NODE_FINALISE},
        )
        graph.add_conditional_edges(
            NODE_RISK, route_after_risk, {NODE_ORDER: NODE_ORDER, NODE_FINALISE: NODE_FINALISE}
        )

        graph.add_edge(NODE_ORDER, END)
        graph.add_edge(NODE_FINALISE, END)

        return graph.compile()

    # -------------------------------------------------------------------- run

    def run(
        self,
        symbol: str,
        timeframe: str,
        timestamp: pd.Timestamp,
        market_data: Any,
        equity: float,
        features: Any = None,
        data_quality: str | None = None,
    ) -> TradingState:
        """Execute the graph for one bar and return the final state."""
        state = new_state(
            symbol, timeframe, timestamp, market_data, equity, features,
            data_quality if data_quality is not None else self.data_quality,
        )
        result = self._graph.invoke(state)
        return result  # type: ignore[return-value]

    def node_names(self) -> list[str]:
        """Every node in the compiled graph, for visualisation and tests."""
        return list(self._graph.get_graph().nodes)

    def to_mermaid(self) -> str:
        """Mermaid rendering of the graph structure."""
        return self._graph.get_graph().draw_mermaid()


def build_graph(
    config: AppConfig,
    portfolio: Portfolio,
    asset: AssetConfig | None = None,
    **kwargs: Any,
) -> TradingGraph:
    """Convenience constructor."""
    return TradingGraph(config=config, portfolio=portfolio, asset=asset, **kwargs)
