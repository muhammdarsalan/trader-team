"""Portfolio state.

Tracks cash, open positions, realised and unrealised P&L, and the equity curve.

Equity is marked to market on every bar, not only when a trade closes. A
drawdown that exists only in open positions is a real drawdown - it is what a
risk limit must react to, and what would actually have been felt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.execution.models import ExitReason, Fill, Position, Trade
from app.signals.models import SignalDirection
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Portfolio state at one bar, recorded for the equity curve."""

    timestamp: pd.Timestamp
    cash: float
    equity: float
    unrealised_pnl: float
    realised_pnl: float
    open_positions: int
    exposure: float
    drawdown: float


class Portfolio:
    """Cash, positions and the equity history."""

    def __init__(self, initial_balance: float, currency: str = "USD") -> None:
        if initial_balance <= 0:
            raise ValueError(f"initial_balance must be positive, got {initial_balance}")

        self.initial_balance = float(initial_balance)
        self.currency = currency
        self.cash = float(initial_balance)

        self.positions: list[Position] = []
        self.closed_trades: list[Trade] = []
        self.snapshots: list[PortfolioSnapshot] = []

        self.realised_pnl = 0.0
        self.total_costs = 0.0

        self._peak_equity = float(initial_balance)
        self._day_start_equity = float(initial_balance)
        self._current_day: pd.Timestamp | None = None
        # Highest and lowest excursion seen while each position is open, keyed
        # by id(position), for MAE/MFE reporting.
        self._excursions: dict[int, tuple[float, float]] = {}

    # ------------------------------------------------------------------ state

    @property
    def open_count(self) -> int:
        return len(self.positions)

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus the mark-to-market value of open positions."""
        return self.cash + self.unrealised_pnl(prices)

    def unrealised_pnl(self, prices: dict[str, float]) -> float:
        total = 0.0
        for position in self.positions:
            price = prices.get(position.symbol)
            if price is not None:
                total += position.unrealised_pnl(price)
        return total

    def exposure(self, prices: dict[str, float]) -> float:
        """Total notional of open positions at current prices."""
        total = 0.0
        for position in self.positions:
            price = prices.get(position.symbol, position.entry_price)
            total += abs(position.quantity * price)
        return total

    def open_risk(self) -> float:
        """Sum of the money still at risk across open positions.

        Measured to each position's stop, which is the loss that would occur if
        every open trade hit its stop at once - the scenario a portfolio risk
        limit exists to bound.
        """
        return sum(position.risk_amount for position in self.positions)

    def positions_for(self, symbol: str) -> list[Position]:
        return [p for p in self.positions if p.symbol == symbol.upper()]

    def risk_for_strategy(self, strategy: str) -> float:
        return sum(p.risk_amount for p in self.positions if p.strategy == strategy)

    def drawdown(self, equity: float) -> float:
        """Current drawdown from the equity peak, as a positive fraction."""
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - equity) / self._peak_equity)

    def daily_loss(self, equity: float) -> float:
        """Loss so far today as a positive fraction of the day's starting equity."""
        if self._day_start_equity <= 0:
            return 0.0
        return max(0.0, (self._day_start_equity - equity) / self._day_start_equity)

    # ----------------------------------------------------------------- trading

    def open_position(self, position: Position, fill: Fill) -> None:
        """Record a new position and deduct its entry costs."""
        self.positions.append(position)
        self.cash -= fill.total_cost
        self.total_costs += fill.total_cost
        self._excursions[id(position)] = (position.entry_price, position.entry_price)

        logger.info(
            "Position opened",
            extra={
                "symbol": position.symbol, "strategy": position.strategy,
                "direction": str(position.direction), "quantity": position.quantity,
                "entry": position.entry_price, "stop": position.stop_loss,
                "risk": position.risk_amount,
            },
        )

    def close_position(
        self,
        position: Position,
        fill: Fill,
        timestamp: pd.Timestamp,
        reason: ExitReason,
        exit_regime: str = "UNKNOWN",
        bars_held: int = 0,
    ) -> Trade:
        """Close a position and record the completed trade."""
        exit_price = fill.price
        gross = (exit_price - position.entry_price) * position.quantity * position.sign
        costs = position.entry_costs + fill.total_cost
        net = gross - fill.total_cost  # entry costs were charged when opened

        self.cash += gross - fill.total_cost
        self.realised_pnl += net
        self.total_costs += fill.total_cost

        risk_per_unit = position.risk_per_unit
        r_multiple = None
        if risk_per_unit > 0 and position.quantity > 0:
            r_multiple = net / (risk_per_unit * position.quantity)

        low, high = self._excursions.pop(id(position), (position.entry_price, position.entry_price))
        if position.direction is SignalDirection.LONG:
            mae = position.entry_price - low
            mfe = high - position.entry_price
        else:
            mae = high - position.entry_price
            mfe = position.entry_price - low

        trade = Trade(
            symbol=position.symbol,
            strategy=position.strategy,
            direction=position.direction,
            quantity=position.quantity,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_time=timestamp,
            exit_price=exit_price,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            exit_reason=reason,
            gross_pnl=gross,
            costs=costs,
            net_pnl=net,
            r_multiple=r_multiple,
            entry_regime=position.entry_regime,
            exit_regime=exit_regime,
            bars_held=bars_held,
            mae=max(0.0, mae),
            mfe=max(0.0, mfe),
            metadata=dict(position.metadata),
        )

        self.closed_trades.append(trade)
        if position in self.positions:
            self.positions.remove(position)

        logger.info(
            "Position closed",
            extra={
                "symbol": trade.symbol, "strategy": trade.strategy,
                "reason": str(reason), "net_pnl": round(net, 2),
                "r_multiple": None if r_multiple is None else round(r_multiple, 3),
            },
        )
        return trade

    # ---------------------------------------------------------------- tracking

    def track_excursions(self, prices: dict[str, float]) -> None:
        """Update the best and worst price seen while each position is open."""
        for position in self.positions:
            price = prices.get(position.symbol)
            if price is None:
                continue
            low, high = self._excursions.get(id(position), (price, price))
            self._excursions[id(position)] = (min(low, price), max(high, price))

    def record_snapshot(
        self, timestamp: pd.Timestamp, prices: dict[str, float]
    ) -> PortfolioSnapshot:
        """Mark to market and append to the equity curve."""
        self._roll_day(timestamp)

        equity = self.equity(prices)
        self._peak_equity = max(self._peak_equity, equity)

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            equity=equity,
            unrealised_pnl=self.unrealised_pnl(prices),
            realised_pnl=self.realised_pnl,
            open_positions=len(self.positions),
            exposure=self.exposure(prices),
            drawdown=self.drawdown(equity),
        )
        self.snapshots.append(snapshot)
        return snapshot

    def _roll_day(self, timestamp: pd.Timestamp) -> None:
        """Reset the daily loss baseline when the calendar day changes."""
        day = timestamp.normalize()
        if self._current_day is None:
            self._current_day = day
            return
        if day != self._current_day:
            self._current_day = day
            self._day_start_equity = self.snapshots[-1].equity if self.snapshots else self.cash

    # ----------------------------------------------------------------- outputs

    def equity_curve(self) -> pd.Series:
        """Equity over time, indexed by bar timestamp."""
        if not self.snapshots:
            return pd.Series(dtype="float64", name="equity")
        return pd.Series(
            [s.equity for s in self.snapshots],
            index=pd.DatetimeIndex([s.timestamp for s in self.snapshots], name="timestamp"),
            name="equity",
        )

    def trades_frame(self) -> pd.DataFrame:
        """Every completed trade as a DataFrame."""
        if not self.closed_trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.closed_trades])

    def snapshots_frame(self) -> pd.DataFrame:
        if not self.snapshots:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "timestamp": s.timestamp, "cash": s.cash, "equity": s.equity,
                    "unrealised_pnl": s.unrealised_pnl, "realised_pnl": s.realised_pnl,
                    "open_positions": s.open_positions, "exposure": s.exposure,
                    "drawdown": s.drawdown,
                }
                for s in self.snapshots
            ]
        ).set_index("timestamp")

    def summary(self) -> dict[str, Any]:
        final_equity = self.snapshots[-1].equity if self.snapshots else self.cash
        return {
            "initial_balance": self.initial_balance,
            "final_equity": final_equity,
            "realised_pnl": self.realised_pnl,
            "total_costs": self.total_costs,
            "closed_trades": len(self.closed_trades),
            "open_positions": len(self.positions),
        }
