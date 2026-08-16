"""Orders, fills and positions.

Every cost that separates a signal price from a realised price is recorded
separately - spread, slippage, commission - so a backtest can answer "how much
of this result was eaten by execution?" That question is usually the difference
between a strategy that works and one that only appeared to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from app.signals.models import SignalDirection


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is OrderSide.BUY else -1

    @classmethod
    def from_direction(cls, direction: SignalDirection) -> OrderSide:
        if direction is SignalDirection.LONG:
            return cls.BUY
        if direction is SignalDirection.SHORT:
            return cls.SELL
        raise ValueError(f"Cannot build an order side from {direction}")


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class ExitReason(StrEnum):
    """Why a position closed. Every close records one."""

    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    END_OF_BACKTEST = "END_OF_BACKTEST"
    RISK_LIMIT = "RISK_LIMIT"
    KILL_SWITCH = "KILL_SWITCH"
    AMBIGUOUS_BAR_SKIPPED = "AMBIGUOUS_BAR_SKIPPED"


@dataclass(frozen=True)
class Order:
    """An instruction to trade, before it has been filled."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    created_at: pd.Timestamp | None = None
    strategy: str = "unknown"
    reference_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("A LIMIT order requires limit_price")
        if self.order_type is OrderType.STOP and self.stop_price is None:
            raise ValueError("A STOP order requires stop_price")


@dataclass(frozen=True)
class Fill:
    """The result of attempting an order.

    ``price`` is what was actually paid or received, including spread and
    slippage. ``reference_price`` is the untouched market price the fill was
    derived from, so the two can be compared.
    """

    order: Order
    status: OrderStatus
    filled_quantity: float = 0.0
    price: float = 0.0
    reference_price: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    commission: float = 0.0
    timestamp: pd.Timestamp | None = None
    rejection_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}

    @property
    def total_cost(self) -> float:
        """Every execution cost, in account currency."""
        return self.spread_cost + self.slippage_cost + self.commission

    @property
    def notional(self) -> float:
        return self.filled_quantity * self.price

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.order.symbol,
            "side": str(self.order.side),
            "status": str(self.status),
            "quantity": self.filled_quantity,
            "price": self.price,
            "reference_price": self.reference_price,
            "spread_cost": self.spread_cost,
            "slippage_cost": self.slippage_cost,
            "commission": self.commission,
            "total_cost": self.total_cost,
            "timestamp": self.timestamp.isoformat() if self.timestamp is not None else None,
            "strategy": self.order.strategy,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class Position:
    """An open position."""

    symbol: str
    direction: SignalDirection
    quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: float
    take_profit: float | None = None
    strategy: str = "unknown"
    entry_regime: str = "UNKNOWN"
    entry_costs: float = 0.0
    risk_amount: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sign(self) -> int:
        return self.direction.sign

    @property
    def notional(self) -> float:
        return abs(self.quantity * self.entry_price)

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    def unrealised_pnl(self, price: float) -> float:
        """Mark-to-market profit, before exit costs."""
        return (price - self.entry_price) * self.quantity * self.sign

    def r_multiple(self, price: float) -> float | None:
        """Profit measured in units of the risk taken.

        R is the only cross-instrument, cross-size comparison that means
        anything: +2R is the same achievement on gold and on EURUSD.
        """
        risk = self.risk_per_unit
        if risk <= 0:
            return None
        return (price - self.entry_price) * self.sign / risk


@dataclass(frozen=True)
class Trade:
    """A completed round trip. The unit of record for all performance analysis."""

    symbol: str
    strategy: str
    direction: SignalDirection
    quantity: float

    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float

    stop_loss: float
    take_profit: float | None
    exit_reason: ExitReason

    gross_pnl: float
    costs: float
    net_pnl: float
    r_multiple: float | None

    entry_regime: str = "UNKNOWN"
    exit_regime: str = "UNKNOWN"
    bars_held: int = 0
    mae: float = 0.0  # maximum adverse excursion, price units
    mfe: float = 0.0  # maximum favourable excursion
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    @property
    def return_pct(self) -> float:
        """Profit relative to the notional committed."""
        notional = abs(self.quantity * self.entry_price)
        return self.net_pnl / notional if notional else 0.0

    def to_dict(self) -> dict[str, Any]:
        """The full trade record required for reproducibility."""
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "direction": str(self.direction),
            "quantity": self.quantity,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat(),
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_reason": str(self.exit_reason),
            "gross_pnl": self.gross_pnl,
            "costs": self.costs,
            "net_pnl": self.net_pnl,
            "r_multiple": self.r_multiple,
            "entry_regime": self.entry_regime,
            "exit_regime": self.exit_regime,
            "bars_held": self.bars_held,
            "mae": self.mae,
            "mfe": self.mfe,
            "return_pct": self.return_pct,
            **{f"meta_{k}": v for k, v in self.metadata.items()},
        }
