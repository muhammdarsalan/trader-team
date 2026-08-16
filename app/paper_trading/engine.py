"""Paper trading execution loop and state persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.config.loader import AppConfig, get_config
from app.config.models import AssetConfig
from app.data.schema import MarketData
from app.data.service import MarketDataService
from app.execution.models import Order, OrderSide, Position, Trade
from app.execution.simulator import ExecutionSimulator
from app.features.engine import FeatureEngine
from app.graph.state import TradingState, state_summary
from app.graph.workflow import TradingGraph
from app.portfolio.portfolio import Portfolio
from app.signals.models import SignalDirection
from app.signals.selector import RegimePerformanceTracker
from app.utils.logging import get_logger
from app.utils.timeutils import normalize_timeframe, utcnow

logger = get_logger(__name__)


def _serialise_timestamp(value: pd.Timestamp | None) -> str | None:
    return None if value is None else value.isoformat()


def _deserialise_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    return pd.Timestamp(value).tz_localize("UTC") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value)


class PaperTradingEngine:
    """Paper trading loop with JSON-backed state persistence.

    The engine mirrors the backtester sequencing but executes only on a live or
    near-live feed. Orders produced on the current bar are queued for the next
    bar's open, and each bar is tracked with a persistent processed-bar set so a
    restart does not double-fill anything that had already completed.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        symbol: str | None = None,
        timeframe: str = "1D",
        state_path: str | Path | None = None,
        market_data_service: MarketDataService | None = None,
        portfolio: Portfolio | None = None,
        asset: AssetConfig | None = None,
    ) -> None:
        self.config = config or get_config()
        self.symbol = (symbol or self.config.assets.enabled_symbols()[0]).upper()
        self.timeframe = normalize_timeframe(timeframe).code
        self.asset = asset or self.config.assets.get(self.symbol)
        self.state_path = Path(state_path) if state_path is not None else None
        self.market_data_service = market_data_service or MarketDataService()
        self.portfolio = portfolio or Portfolio(
            self.config.platform.starting_balance, self.config.platform.base_currency
        )
        self.performance = RegimePerformanceTracker()
        self.graph = TradingGraph(
            config=self.config,
            portfolio=self.portfolio,
            asset=self.asset,
            performance=self.performance,
            trading_enabled=self.config.platform.trading_enabled,
        )
        self.simulator = ExecutionSimulator(self.config.execution, self.asset)
        self.feature_engine = FeatureEngine(self.config.features)

        self._state: dict[str, Any] = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "starting_balance": self.config.platform.starting_balance,
            "currency": self.config.platform.base_currency,
            "cash": float(self.portfolio.cash),
            "realised_pnl": float(self.portfolio.realised_pnl),
            "total_costs": float(self.portfolio.total_costs),
            "lanes": {"market": {}, "regime": {}, "strategy_signals": {}, "risk": {}, "final": {}},
            "processed_bars": [],
            "pending_orders": [],
            "fills": [],
            "orders": [],
            "positions": [],
            "closed_trades": [],
            "recent_events": [],
            "recent_errors": [],
            "latest_decision": None,
            "latest_state": None,
            "last_bar": None,
            "last_quality": None,
        }
        self._last_market: dict[str, Any] | None = None
        self._last_state: dict[str, Any] | None = None
        self._last_quality_status: str | None = None

        if self.state_path is not None and self.state_path.exists():
            self.load_state(self.state_path)

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def _persist(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._serialise_state()
        self.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _serialise_state(self) -> dict[str, Any]:
        positions = [
            {
                "symbol": p.symbol,
                "direction": str(p.direction),
                "quantity": p.quantity,
                "entry_price": p.entry_price,
                "entry_time": _serialise_timestamp(p.entry_time),
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
                "strategy": p.strategy,
                "entry_regime": p.entry_regime,
                "entry_costs": p.entry_costs,
                "risk_amount": p.risk_amount,
                "metadata": p.metadata,
            }
            for p in self.portfolio.positions
        ]
        trades = [t.to_dict() for t in self.portfolio.closed_trades]
        snapshot = {
            **self._state,
            "processed_bars": list(self._state.get("processed_bars", [])),
            "pending_orders": list(self._state.get("pending_orders", [])),
            "fills": list(self._state.get("fills", [])),
            "orders": list(self._state.get("orders", [])),
            "positions": positions,
            "closed_trades": trades,
            "cash": float(self.portfolio.cash),
            "realised_pnl": float(self.portfolio.realised_pnl),
            "total_costs": float(self.portfolio.total_costs),
            "last_bar": _serialise_timestamp(self._last_bar_timestamp()),
            "latest_state": self._last_state,
            "latest_decision": self._state.get("latest_decision"),
        }
        return snapshot

    def _last_bar_timestamp(self) -> pd.Timestamp | None:
        ts = self._state.get("last_bar")
        if ts is None:
            return None
        if isinstance(ts, str):
            return _deserialise_timestamp(ts)
        return pd.Timestamp(ts)

    def _load_from_snapshot(self, payload: dict[str, Any]) -> None:
        self._state = {
            **self._state,
            **payload,
            "positions": list(payload.get("positions", [])),
            "closed_trades": list(payload.get("closed_trades", [])),
            "fills": list(payload.get("fills", [])),
            "orders": list(payload.get("orders", [])),
            "pending_orders": list(payload.get("pending_orders", [])),
            "processed_bars": list(payload.get("processed_bars", [])),
            "recent_events": list(payload.get("recent_events", [])),
            "recent_errors": list(payload.get("recent_errors", [])),
            "latest_decision": payload.get("latest_decision"),
            "latest_state": payload.get("latest_state"),
        }
        self.portfolio = Portfolio(
            float(payload.get("starting_balance", self.config.platform.starting_balance)),
            payload.get("currency", self.config.platform.base_currency),
        )
        self.portfolio.cash = float(payload.get("cash", self.portfolio.initial_balance))
        self.portfolio.realised_pnl = float(payload.get("realised_pnl", 0.0))
        self.portfolio.total_costs = float(payload.get("total_costs", 0.0))
        for item in payload.get("positions", []):
            pos = Position(
                symbol=item["symbol"],
                direction=SignalDirection[item["direction"].upper()],
                quantity=float(item["quantity"]),
                entry_price=float(item["entry_price"]),
                entry_time=_deserialise_timestamp(item.get("entry_time")) or utcnow(),
                stop_loss=float(item["stop_loss"]),
                take_profit=item.get("take_profit"),
                strategy=item.get("strategy", "unknown"),
                entry_regime=item.get("entry_regime", "UNKNOWN"),
                entry_costs=float(item.get("entry_costs", 0.0)),
                risk_amount=float(item.get("risk_amount", 0.0)),
                metadata=item.get("metadata", {}),
            )
            self.portfolio.positions.append(pos)
        for raw in payload.get("closed_trades", []):
            try:
                trade = Trade(
                    symbol=raw["symbol"],
                    strategy=raw["strategy"],
                    direction=SignalDirection[raw["direction"].upper()],
                    quantity=float(raw["quantity"]),
                    entry_time=_deserialise_timestamp(raw["entry_time"]) or utcnow(),
                    entry_price=float(raw["entry_price"]),
                    exit_time=_deserialise_timestamp(raw["exit_time"]) or utcnow(),
                    exit_price=float(raw["exit_price"]),
                    stop_loss=float(raw["stop_loss"]),
                    take_profit=raw.get("take_profit"),
                    exit_reason=raw.get("exit_reason"),
                    gross_pnl=float(raw.get("gross_pnl", 0.0)),
                    costs=float(raw.get("costs", 0.0)),
                    net_pnl=float(raw.get("net_pnl", 0.0)),
                    r_multiple=raw.get("r_multiple"),
                    entry_regime=raw.get("entry_regime", "UNKNOWN"),
                    exit_regime=raw.get("exit_regime", "UNKNOWN"),
                    bars_held=int(raw.get("bars_held", 0)),
                    mae=float(raw.get("mae", 0.0)),
                    mfe=float(raw.get("mfe", 0.0)),
                    metadata={k: v for k, v in raw.items() if k.startswith("meta_")},
                )
                self.portfolio.closed_trades.append(trade)
            except Exception:
                self._record_error("Failed to restore a saved trade", node="paper_trading")

    def save_state(self, path: str | Path | None = None) -> dict[str, Any]:
        if path is not None:
            self.state_path = Path(path)
        self._persist()
        return self._serialise_state()

    def load_state(self, path: str | Path | None = None) -> dict[str, Any]:
        target = Path(path) if path is not None else self.state_path
        if target is None or not target.exists():
            self._persist()
            return self._serialise_state()
        with target.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self._load_from_snapshot(payload)
        self._state["symbol"] = payload.get("symbol", self.symbol)
        self._state["timeframe"] = payload.get("timeframe", self.timeframe)
        self._last_state = payload.get("latest_state")
        self._last_quality_status = payload.get("last_quality")
        return payload

    # ------------------------------------------------------------------ monitoring

    def refresh_market_data(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        bars: int = 200,
        *,
        validate: bool = True,
    ) -> MarketData:
        symbol = (symbol or self.symbol).upper()
        timeframe = normalize_timeframe(timeframe or self.timeframe).code
        result = self.market_data_service.get_latest_data(
            symbol,
            timeframe,
            bars=bars,
            validate=validate,
        )
        self._last_market = {
            "symbol": symbol,
            "timeframe": timeframe,
            "rows": len(result.df),
            "last_bar": _serialise_timestamp(result.df.index[-1]) if not result.df.empty else None,
            "quality_status": str(result.quality.status),
            "quality": result.quality.to_dict(),
            "provider": result.data.provider,
            "rows_seen": len(result.df),
        }
        self._state["market"] = self._last_market
        self._state["last_quality"] = str(result.quality.status)
        self._last_quality_status = self._state["last_quality"]
        self._record_event(
            "market_data_refreshed",
            f"Refreshed {symbol} {timeframe} with {len(result.df)} bars",
            **self._last_market,
        )
        return result.data

    def _record_event(self, event: str, message: str, **payload: Any) -> None:
        entry = {"event": event, "message": message, "timestamp": utcnow().isoformat(), **payload}
        recent = list(self._state.get("recent_events", []))
        recent.append(entry)
        self._state["recent_events"] = recent[-50:]

    def _record_error(self, message: str, *, node: str = "paper_trading", **payload: Any) -> None:
        entry = {"node": node, "error": message, "timestamp": utcnow().isoformat(), **payload}
        recent = list(self._state.get("recent_errors", []))
        recent.append(entry)
        self._state["recent_errors"] = recent[-25:]

    def _current_market_summary(self) -> dict[str, Any]:
        market = self._state.get("market") or self._last_market or {}
        if not market:
            return {"symbol": self.symbol, "timeframe": self.timeframe, "status": "NO_DATA"}
        return {
            "symbol": market.get("symbol", self.symbol),
            "timeframe": market.get("timeframe", self.timeframe),
            "rows": market.get("rows", 0),
            "last_bar": market.get("last_bar"),
            "quality_status": market.get("quality_status", "UNKNOWN"),
            "provider": market.get("provider", "unknown"),
        }

    def _decide_on_bar(
        self,
        market_data: MarketData,
        timestamp: pd.Timestamp,
        *,
        features: Any = None,
        data_quality: str | None = None,
    ) -> TradingState:
        if features is None:
            features = self.feature_engine.compute(market_data, self.asset)
        state = self.graph.run(
            symbol=market_data.symbol,
            timeframe=market_data.timeframe.code,
            timestamp=timestamp,
            market_data=market_data,
            equity=self.portfolio.equity({market_data.symbol: float(market_data.df["close"].iloc[-1])}),
            features=features.upto(timestamp) if hasattr(features, "upto") else features,
            data_quality=data_quality or self._last_quality_status,
        )
        self._last_state = state_summary(state)
        self._state["latest_state"] = self._last_state
        self._state["latest_decision"] = state.get("decision")
        self._state["last_bar"] = _serialise_timestamp(timestamp)
        return state

    def _fill_pending_orders(self, bar: pd.Series, timestamp: pd.Timestamp) -> None:
        for pending in list(self._state.get("pending_orders", [])):
            if pending.get("fill_key") in {item.get("fill_key") for item in self._state.get("fills", [])}:
                continue
            order = Order(
                symbol=pending["symbol"],
                side=OrderSide[pending["side"]],
                quantity=float(pending["quantity"]),
                created_at=_deserialise_timestamp(pending.get("created_at")),
                strategy=pending.get("strategy", "unknown"),
                reference_price=float(pending.get("reference_price", 0.0)),
                metadata=pending.get("metadata", {}),
            )
            fill = self.simulator.execute(order, bar, timestamp)
            if fill.is_filled:
                position = Position(
                    symbol=order.symbol,
                    direction=SignalDirection.LONG if order.side is OrderSide.BUY else SignalDirection.SHORT,
                    quantity=fill.filled_quantity,
                    entry_price=fill.price,
                    entry_time=timestamp,
                    stop_loss=float(pending.get("stop_loss", fill.price)),
                    take_profit=pending.get("take_profit"),
                    strategy=order.strategy,
                    entry_regime=pending.get("regime", "UNKNOWN"),
                    entry_costs=fill.total_cost,
                    risk_amount=float(pending.get("risk_amount", 0.0)),
                    metadata={"fill_order_id": pending.get("fill_key")},
                )
                self.portfolio.open_position(position, fill)
                self._state["positions"] = [
                    {
                        "symbol": p.symbol,
                        "direction": str(p.direction),
                        "quantity": p.quantity,
                        "entry_price": p.entry_price,
                        "entry_time": _serialise_timestamp(p.entry_time),
                        "stop_loss": p.stop_loss,
                        "take_profit": p.take_profit,
                        "strategy": p.strategy,
                        "entry_regime": p.entry_regime,
                        "entry_costs": p.entry_costs,
                        "risk_amount": p.risk_amount,
                        "metadata": p.metadata,
                    }
                    for p in self.portfolio.positions
                ]
                fill_record = {
                    "fill_key": pending.get("fill_key"),
                    "symbol": order.symbol,
                    "side": str(order.side),
                    "quantity": fill.filled_quantity,
                    "price": fill.price,
                    "reference_price": fill.reference_price,
                    "timestamp": _serialise_timestamp(timestamp),
                    "strategy": order.strategy,
                    "total_cost": fill.total_cost,
                }
                self._state.setdefault("fills", []).append(fill_record)
                self._state["pending_orders"] = [
                    item for item in self._state.get("pending_orders", [])
                    if item.get("fill_key") != pending.get("fill_key")
                ]
                self._record_event(
                    "paper_fill",
                    f"Filled {order.symbol} {order.side} at {fill.price:.6g}",
                    order=order,
                    fill=fill_record,
                )
            else:
                self._record_error(f"Pending order rejected for {order.symbol}", node="execution")

    def _queue_order(self, state: TradingState) -> None:
        order = state.get("order")
        if order is None:
            return
        order_key = f"{order.symbol}:{order.side}:{order.created_at}:{order.quantity}:{order.strategy}"
        pending = {
            "fill_key": order_key,
            "symbol": order.symbol,
            "side": str(order.side),
            "quantity": float(order.quantity),
            "created_at": _serialise_timestamp(order.created_at),
            "strategy": order.strategy,
            "reference_price": order.reference_price,
            "stop_loss": order.metadata.get("stop_loss"),
            "take_profit": order.metadata.get("take_profit"),
            "risk_amount": order.metadata.get("risk_amount"),
            "regime": order.metadata.get("regime", "UNKNOWN"),
            "metadata": order.metadata,
        }
        self._state.setdefault("orders", []).append({
            "order_key": order_key,
            "symbol": order.symbol,
            "side": str(order.side),
            "quantity": float(order.quantity),
            "created_at": _serialise_timestamp(order.created_at),
            "strategy": order.strategy,
            "reference_price": order.reference_price,
        })
        self._state.setdefault("pending_orders", []).append(pending)
        self._record_event(
            "paper_order",
            f"Queued {order.side} order for {order.symbol} quantity {order.quantity}",
            order_key=order_key,
            quantity=order.quantity,
            symbol=order.symbol,
        )

    def process_bar(
        self,
        market_data: MarketData,
        *,
        timestamp: pd.Timestamp | None = None,
        features: Any = None,
        data_quality: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        if market_data.df.empty:
            raise ValueError("Cannot process an empty market data frame")
        if timestamp is None:
            timestamp = market_data.df.index[-1]
        bar_key = timestamp.isoformat()
        if not force and bar_key in self._state.get("processed_bars", []):
            if self._last_state is not None:
                return {"status": "skipped", "bar": bar_key, "latest_state": self._last_state}
            return {"status": "skipped", "bar": bar_key}

        bar = market_data.df.loc[timestamp]
        self._fill_pending_orders(bar, timestamp)

        state = self._decide_on_bar(market_data, timestamp, features=features, data_quality=data_quality)
        self._queue_order(state)
        self._state.setdefault("processed_bars", []).append(bar_key)
        self._state["last_bar"] = _serialise_timestamp(timestamp)
        self._state["last_quality"] = data_quality or self._last_quality_status or "UNKNOWN"
        self._persist()
        return self.snapshot()

    def catch_up(
        self,
        market_data: MarketData,
        *,
        features: Any = None,
        data_quality: str | None = None,
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for timestamp in market_data.df.index:
            if timestamp.isoformat() in self._state.get("processed_bars", []):
                continue
            output = self.process_bar(
                market_data,
                timestamp=timestamp,
                features=features,
                data_quality=data_quality,
                force=False,
            )
            outputs.append(output)
        return outputs

    def portfolio_snapshot(self, prices: dict[str, float] | None = None) -> dict[str, Any]:
        if prices is None:
            prices = {}
        equity = self.portfolio.equity(prices)
        drawdown = self.portfolio.drawdown(equity)
        return {
            "cash": float(self.portfolio.cash),
            "balance": float(self.portfolio.initial_balance),
            "equity": float(equity),
            "unrealised_pnl": float(self.portfolio.unrealised_pnl(prices)),
            "realised_pnl": float(self.portfolio.realised_pnl),
            "open_positions": len(self.portfolio.positions),
            "total_costs": float(self.portfolio.total_costs),
            "drawdown": float(drawdown),
            "open_risk": float(self.portfolio.open_risk()),
            "exposure": float(self.portfolio.exposure(prices)),
        }

    def snapshot(self) -> dict[str, Any]:
        prices = {
            pos.symbol: pos.entry_price for pos in self.portfolio.positions
        }
        metrics = self.portfolio_snapshot(prices)
        chart = {
            "market": self._current_market_summary(),
            "portfolio": metrics,
            "decision": self._state.get("latest_decision"),
            "state": self._state.get("latest_state"),
            "events": list(self._state.get("recent_events", []))[-10:],
            "errors": list(self._state.get("recent_errors", []))[-10:],
            "orders": list(self._state.get("orders", []))[-10:],
            "fills": list(self._state.get("fills", []))[-10:],
            "positions": list(self._state.get("positions", [])),
        }
        self._state["snapshot"] = chart
        return chart

    def dashboard_data(self) -> dict[str, Any]:
        market = self._current_market_summary()
        prices = {pos.symbol: pos.entry_price for pos in self.portfolio.positions}
        portfolio = self.portfolio_snapshot(prices)
        latest = self._state.get("latest_state") or self._last_state or {}
        decision = self._state.get("latest_decision") or (self._last_state.get("decision") if self._last_state else None)
        risk = None
        if isinstance(latest, dict):
            risk = latest.get("risk_verdict")
        graph = build_graph_visualization(self._state.get("latest_state") or self._last_state or self._state)
        payload = {
            "symbol": market.get("symbol", self.symbol),
            "timeframe": market.get("timeframe", self.timeframe),
            "market": market,
            "portfolio": portfolio,
            "current_regime": latest.get("regime") if isinstance(latest, dict) else None,
            "strategy_signals": latest.get("signals") if isinstance(latest, dict) else {},
            "selection_reasoning": latest.get("decision") if isinstance(latest, dict) else None,
            "risk_decision": risk,
            "final_decision": decision,
            "graph": graph,
            "system_health": {
                "status": "HEALTHY" if not self._state.get("recent_errors") else "DEGRADED",
                "data_freshness": "OK" if market.get("last_bar") else "STALE",
                "quality_status": market.get("quality_status", "UNKNOWN"),
            },
            "events": list(self._state.get("recent_events", []))[-10:],
            "errors": list(self._state.get("recent_errors", []))[-10:],
            "performance": {
                "realised_pnl": portfolio["realised_pnl"],
                "drawdown": portfolio["drawdown"],
                "open_positions": portfolio["open_positions"],
            },
        }
        return payload


def build_graph_visualization(state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Simple graph metadata for the Streamlit panel and tests."""
    nodes = [
        {"id": "market_data", "label": "Market Data", "status": "ok"},
        {"id": "features", "label": "Features", "status": "ok"},
        {"id": "regime", "label": "Regime", "status": "ok"},
        {"id": "strategy_aggregation", "label": "Strategies", "status": "ok"},
        {"id": "selection", "label": "Aggregation / Selection", "status": "ok"},
        {"id": "risk", "label": "Risk", "status": "ok"},
        {"id": "execution", "label": "Execution", "status": "ok"},
        {"id": "paper_position", "label": "Paper Position", "status": "ok"},
    ]
    edges = [
        ["market_data", "features"],
        ["features", "regime"],
        ["regime", "strategy_aggregation"],
        ["strategy_aggregation", "selection"],
        ["selection", "risk"],
        ["risk", "execution"],
        ["execution", "paper_position"],
    ]
    if state is None:
        return {"nodes": nodes, "edges": edges, "active_nodes": ["market_data", "features"], "suppressed": [], "rejected": [], "risk_blocks": []}

    signals = state.get("signals") or {}
    weights = state.get("strategy_weights") or {}
    risk_decision = state.get("risk_decision")
    if risk_decision is not None:
        risk_label = str(risk_decision.block_reason) if getattr(risk_decision, "block_reason", None) else str(risk_decision.verdict)
    else:
        risk_label = "UNKNOWN"

    active = ["market_data", "features", "regime", "strategy_aggregation", "selection"]
    suppressed: list[str] = []
    for name, weight in weights.items():
        if getattr(weight, "weight", 0.0) <= 0:
            suppressed.append(name)
    for name, signal in (signals or {}).items():
        if getattr(signal, "is_actionable", False):
            active.append(f"strategy_{name}")
        else:
            active.append(f"strategy_{name}_rejected")
    if risk_decision is not None and not getattr(risk_decision, "approved", True):
        active.append("risk_blocked")
    if state.get("order") is not None:
        active.append("execution")
        active.append("paper_position")
    return {
        "nodes": nodes,
        "edges": edges,
        "active_nodes": sorted(set(active)),
        "suppressed": sorted(suppressed),
        "rejected": sorted(name for name, signal in (signals or {}).items() if not getattr(signal, "is_actionable", False)),
        "risk_blocks": [risk_label] if risk_decision is not None and not getattr(risk_decision, "approved", True) else [],
    }


__all__ = ["PaperTradingEngine", "build_graph_visualization"]
