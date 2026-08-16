"""Realistic fill simulation.

The rules below are the ones that decide whether a backtest is informative or
fiction. Each is deliberately pessimistic, because every optimistic execution
assumption manufactures profit that does not exist:

1. **Fills happen on the next bar's open.** A signal derived from bar t's close
   cannot be filled at that close - the price had already passed by the time
   the decision existed.
2. **Buys pay the ask, sells receive the bid.** The spread is a real cost paid
   on entry *and* exit.
3. **Slippage scales with volatility.** Fills are worst precisely when the
   market is moving fastest, so a fixed figure understates the cost of the
   trades that matter most.
4. **Gaps are honoured.** If a bar opens beyond the stop, the fill is at the
   open, not at the stop price. Pretending otherwise deletes exactly the losses
   that hurt most.
5. **Ambiguous bars resolve against you.** When one bar's range contains both
   the stop and the target, OHLC data cannot say which came first. Assuming the
   target believes the flattering possibility every single time.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.config.models import AssetConfig, ExecutionConfig
from app.execution.models import (
    ExitReason,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)
from app.signals.models import SignalDirection
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExitEvent:
    """A position exit detected within a bar."""

    reason: ExitReason
    price: float
    ambiguous: bool = False


class ExecutionSimulator:
    """Turns orders into fills using a bar's OHLC and configured costs."""

    def __init__(
        self,
        config: ExecutionConfig | None = None,
        asset: AssetConfig | None = None,
    ) -> None:
        self.config = config or ExecutionConfig()
        self.asset = asset

    # ------------------------------------------------------------------ entry

    def execute(
        self,
        order: Order,
        bar: pd.Series,
        timestamp: pd.Timestamp,
        atr: float | None = None,
    ) -> Fill:
        """Fill ``order`` against ``bar``.

        Args:
            order: the instruction. Market orders fill at the bar's open.
            bar: the OHLCV row to fill against. **This must be the bar *after*
                the one that produced the signal** - the caller is responsible
                for that, and the backtester enforces it.
            timestamp: the fill bar's timestamp.
            atr: current ATR, required by the ``atr_fraction`` slippage model.

        Returns:
            A :class:`Fill`, which may be a rejection.
        """
        reference = float(bar["open"])

        if reference <= 0:
            return self._reject(order, timestamp, "Non-positive reference price")

        rejection = self._check_volume_limit(order, bar)
        if rejection is not None:
            return self._reject(order, timestamp, rejection)

        spread_per_unit = self._spread_per_unit()
        slippage_per_unit = self._slippage_per_unit(reference, atr)

        # Both costs always work against the trader: a buy pays more, a sell
        # receives less.
        adjustment = order.side.sign * (spread_per_unit / 2.0 + slippage_per_unit)
        fill_price = reference + adjustment

        if fill_price <= 0:
            return self._reject(order, timestamp, "Costs drove the fill price non-positive")

        commission = self._commission(order.quantity, fill_price)

        return Fill(
            order=order,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            price=fill_price,
            reference_price=reference,
            spread_cost=abs(spread_per_unit / 2.0) * order.quantity,
            slippage_cost=abs(slippage_per_unit) * order.quantity,
            commission=commission,
            timestamp=timestamp,
            metadata={"bar_open": reference},
        )

    # ------------------------------------------------------------------- exit

    def check_exit(self, position: Position, bar: pd.Series) -> ExitEvent | None:
        """Whether ``position`` would have closed during ``bar``.

        Checks the bar's high and low against the stop and target. Returns None
        when the position survives the bar.

        The hard case is a bar whose range contains both levels. Daily OHLC
        cannot say which was touched first - resolving it needs intrabar data
        the platform does not have. ``same_bar_resolution`` decides, and
        defaults to assuming the stop.
        """
        high, low = float(bar["high"]), float(bar["low"])
        open_price = float(bar["open"])
        is_long = position.direction is SignalDirection.LONG

        stop = position.stop_loss
        target = position.take_profit

        stop_hit = low <= stop if is_long else high >= stop
        target_hit = (
            target is not None and (high >= target if is_long else low <= target)
        )

        if not stop_hit and not target_hit:
            return None

        # --- gaps: the open is already through the level ---------------------
        if self.config.honour_gaps:
            gapped_through_stop = open_price <= stop if is_long else open_price >= stop
            if stop_hit and gapped_through_stop:
                # The market opened beyond the stop; that open is the first
                # available price, and it is worse than the stop.
                return ExitEvent(ExitReason.STOP_LOSS, open_price)

            if target is not None:
                gapped_through_target = (
                    open_price >= target if is_long else open_price <= target
                )
                if target_hit and gapped_through_target and not stop_hit:
                    return ExitEvent(ExitReason.TAKE_PROFIT, open_price)

        # --- both levels inside one bar --------------------------------------
        if stop_hit and target_hit:
            resolution = self.config.same_bar_resolution
            if resolution == "target_first":
                return ExitEvent(ExitReason.TAKE_PROFIT, target, ambiguous=True)
            if resolution == "skip":
                return ExitEvent(ExitReason.AMBIGUOUS_BAR_SKIPPED, stop, ambiguous=True)
            return ExitEvent(ExitReason.STOP_LOSS, stop, ambiguous=True)

        if stop_hit:
            return ExitEvent(ExitReason.STOP_LOSS, stop)
        return ExitEvent(ExitReason.TAKE_PROFIT, float(target))

    def exit_fill(
        self,
        position: Position,
        exit_price: float,
        timestamp: pd.Timestamp,
        reason: ExitReason,
        atr: float | None = None,
        apply_slippage: bool = True,
    ) -> Fill:
        """Costs on closing a position.

        Exit costs are as real as entry costs and are frequently ignored, which
        roughly halves the modelled cost of every round trip.
        """
        side = OrderSide.SELL if position.direction is SignalDirection.LONG else OrderSide.BUY
        order = Order(
            symbol=position.symbol,
            side=side,
            quantity=position.quantity,
            created_at=timestamp,
            strategy=position.strategy,
            reference_price=exit_price,
            metadata={"exit_reason": str(reason)},
        )

        spread_per_unit = self._spread_per_unit()
        # A stop or target fills *at* its level by construction; the extra cost
        # is the spread and any slippage beyond it.
        slippage_per_unit = self._slippage_per_unit(exit_price, atr) if apply_slippage else 0.0

        adjustment = side.sign * (spread_per_unit / 2.0 + slippage_per_unit)
        fill_price = max(exit_price + adjustment, 1e-9)
        commission = self._commission(position.quantity, fill_price)

        return Fill(
            order=order,
            status=OrderStatus.FILLED,
            filled_quantity=position.quantity,
            price=fill_price,
            reference_price=exit_price,
            spread_cost=abs(spread_per_unit / 2.0) * position.quantity,
            slippage_cost=abs(slippage_per_unit) * position.quantity,
            commission=commission,
            timestamp=timestamp,
            metadata={"exit_reason": str(reason)},
        )

    # -------------------------------------------------------------- internals

    def _spread_per_unit(self) -> float:
        if not self.config.apply_spread or self.asset is None:
            return 0.0
        return float(self.asset.typical_spread) * self.config.spread_multiplier

    def _slippage_per_unit(self, price: float, atr: float | None) -> float:
        model = self.config.slippage_model
        value = self.config.slippage_value

        if model == "none" or value <= 0:
            return 0.0
        if model == "fixed_ticks":
            tick = float(self.asset.tick_size) if self.asset else 0.01
            return value * tick
        if model == "percent":
            return price * value
        if model == "atr_fraction":
            if atr is None or not pd.notna(atr) or atr <= 0:
                # No volatility estimate: fall back to a percentage rather than
                # silently assuming zero slippage.
                return price * 0.0005
            return float(atr) * value
        raise ValueError(f"Unknown slippage model {model!r}")

    def _commission(self, quantity: float, price: float) -> float:
        commission = quantity * self.config.commission_per_unit
        commission += abs(quantity * price) * self.config.commission_pct
        return max(commission, self.config.min_commission) if commission or self.config.min_commission else 0.0

    def _check_volume_limit(self, order: Order, bar: pd.Series) -> str | None:
        """Reject orders too large for the bar, where volume is meaningful."""
        if not self.config.enforce_volume_limit:
            return None
        if self.asset is not None and not self.asset.has_reliable_volume:
            return None

        volume = bar.get("volume")
        if volume is None or pd.isna(volume) or volume <= 0:
            return None

        participation = order.quantity / float(volume)
        if participation > self.config.max_volume_participation:
            return (
                f"Order of {order.quantity:.4g} units is {participation:.1%} of the bar's "
                f"volume, above the {self.config.max_volume_participation:.1%} limit"
            )
        return None

    @staticmethod
    def _reject(order: Order, timestamp: pd.Timestamp, reason: str) -> Fill:
        logger.warning(
            "Order rejected",
            extra={"symbol": order.symbol, "strategy": order.strategy, "reason": reason},
        )
        return Fill(
            order=order,
            status=OrderStatus.REJECTED,
            timestamp=timestamp,
            rejection_reason=reason,
        )
