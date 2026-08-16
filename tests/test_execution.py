"""Execution realism: fills, spread, slippage, costs, gaps and ambiguous bars.

These tests pin the assumptions that decide whether a backtest is informative.
Each one exists because the optimistic alternative manufactures profit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.config.models import AssetConfig, ExecutionConfig
from app.execution.models import (
    ExitReason,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from app.execution.simulator import ExecutionSimulator
from app.signals.models import SignalDirection


def asset(**overrides) -> AssetConfig:
    base = {
        "symbol": "XAUUSD", "name": "Gold", "asset_class": "METAL",
        "quote_currency": "USD", "tick_size": 0.01, "typical_spread": 0.30,
    }
    return AssetConfig(**{**base, **overrides})


def bar(open_=100.0, high=102.0, low=98.0, close=101.0, volume=1000.0) -> pd.Series:
    return pd.Series(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def order(side=OrderSide.BUY, quantity=10.0) -> Order:
    return Order(symbol="XAUUSD", side=side, quantity=quantity, strategy="test")


def position(direction=SignalDirection.LONG, entry=100.0, stop=98.0, target=104.0) -> Position:
    return Position(
        symbol="XAUUSD",
        direction=direction,
        quantity=10.0,
        entry_price=entry,
        entry_time=pd.Timestamp("2024-01-01", tz="UTC"),
        stop_loss=stop,
        take_profit=target,
        strategy="test",
    )


TS = pd.Timestamp("2024-01-02", tz="UTC")


# ------------------------------------------------------------------- basic fill

def test_market_order_fills_at_the_bar_open():
    """Not at the close: the close is a price that has already traded."""
    sim = ExecutionSimulator(ExecutionConfig(apply_spread=False, slippage_model="none"))
    fill = sim.execute(order(), bar(open_=100.0, close=105.0), TS)

    assert fill.status is OrderStatus.FILLED
    assert fill.price == pytest.approx(100.0)
    assert fill.reference_price == pytest.approx(100.0)


def test_fill_records_quantity_and_timestamp():
    sim = ExecutionSimulator(ExecutionConfig(apply_spread=False, slippage_model="none"))
    fill = sim.execute(order(quantity=7.5), bar(), TS)
    assert fill.filled_quantity == 7.5
    assert fill.timestamp == TS


def test_non_positive_price_is_rejected():
    sim = ExecutionSimulator()
    fill = sim.execute(order(), bar(open_=0.0), TS)
    assert fill.status is OrderStatus.REJECTED
    assert "Non-positive" in fill.rejection_reason


def test_order_requires_positive_quantity():
    with pytest.raises(ValueError, match="quantity must be positive"):
        Order(symbol="XAUUSD", side=OrderSide.BUY, quantity=0.0)


def test_limit_order_requires_a_limit_price():
    with pytest.raises(ValueError, match="requires limit_price"):
        Order(symbol="X", side=OrderSide.BUY, quantity=1.0, order_type=OrderType.LIMIT)


# ----------------------------------------------------------------------- spread

def test_buy_pays_the_ask():
    sim = ExecutionSimulator(
        ExecutionConfig(apply_spread=True, slippage_model="none"), asset()
    )
    fill = sim.execute(order(OrderSide.BUY), bar(open_=100.0), TS)
    assert fill.price == pytest.approx(100.15)   # half the 0.30 spread
    assert fill.spread_cost == pytest.approx(0.15 * 10)


def test_sell_receives_the_bid():
    sim = ExecutionSimulator(
        ExecutionConfig(apply_spread=True, slippage_model="none"), asset()
    )
    fill = sim.execute(order(OrderSide.SELL), bar(open_=100.0), TS)
    assert fill.price == pytest.approx(99.85)


def test_spread_can_be_scaled():
    sim = ExecutionSimulator(
        ExecutionConfig(apply_spread=True, spread_multiplier=2.0, slippage_model="none"),
        asset(),
    )
    fill = sim.execute(order(), bar(open_=100.0), TS)
    assert fill.price == pytest.approx(100.30)


def test_spread_costs_the_trader_in_both_directions():
    """A round trip pays the spread twice, which is the point of modelling it."""
    sim = ExecutionSimulator(
        ExecutionConfig(apply_spread=True, slippage_model="none"), asset()
    )
    buy = sim.execute(order(OrderSide.BUY), bar(open_=100.0), TS)
    sell = sim.execute(order(OrderSide.SELL), bar(open_=100.0), TS)
    assert buy.price > sell.price


# --------------------------------------------------------------------- slippage

def test_atr_slippage_scales_with_volatility():
    """Fills are worst when the market moves fastest; a fixed figure hides that."""
    sim = ExecutionSimulator(
        ExecutionConfig(apply_spread=False, slippage_model="atr_fraction", slippage_value=0.1),
        asset(),
    )
    calm = sim.execute(order(), bar(open_=100.0), TS, atr=1.0)
    volatile = sim.execute(order(), bar(open_=100.0), TS, atr=10.0)

    assert calm.price == pytest.approx(100.1)
    assert volatile.price == pytest.approx(101.0)
    assert volatile.slippage_cost > calm.slippage_cost


def test_atr_slippage_falls_back_when_atr_is_missing():
    """Missing volatility must not silently mean zero slippage."""
    sim = ExecutionSimulator(
        ExecutionConfig(apply_spread=False, slippage_model="atr_fraction"), asset()
    )
    fill = sim.execute(order(), bar(open_=100.0), TS, atr=None)
    assert fill.price > 100.0


def test_fixed_tick_slippage():
    sim = ExecutionSimulator(
        ExecutionConfig(apply_spread=False, slippage_model="fixed_ticks", slippage_value=5),
        asset(tick_size=0.01),
    )
    fill = sim.execute(order(), bar(open_=100.0), TS)
    assert fill.price == pytest.approx(100.05)


def test_percent_slippage():
    sim = ExecutionSimulator(
        ExecutionConfig(apply_spread=False, slippage_model="percent", slippage_value=0.001),
        asset(),
    )
    fill = sim.execute(order(), bar(open_=100.0), TS)
    assert fill.price == pytest.approx(100.1)


def test_slippage_always_works_against_the_trader():
    sim = ExecutionSimulator(
        ExecutionConfig(apply_spread=False, slippage_model="percent", slippage_value=0.01),
        asset(),
    )
    buy = sim.execute(order(OrderSide.BUY), bar(open_=100.0), TS)
    sell = sim.execute(order(OrderSide.SELL), bar(open_=100.0), TS)
    assert buy.price > 100.0
    assert sell.price < 100.0


def test_no_slippage_model_costs_nothing():
    sim = ExecutionSimulator(ExecutionConfig(apply_spread=False, slippage_model="none"), asset())
    fill = sim.execute(order(), bar(open_=100.0), TS)
    assert fill.slippage_cost == 0.0


# ------------------------------------------------------------------ commission

def test_per_unit_commission():
    sim = ExecutionSimulator(
        ExecutionConfig(apply_spread=False, slippage_model="none", commission_per_unit=0.5),
        asset(),
    )
    fill = sim.execute(order(quantity=10.0), bar(), TS)
    assert fill.commission == pytest.approx(5.0)


def test_percentage_commission():
    sim = ExecutionSimulator(
        ExecutionConfig(apply_spread=False, slippage_model="none", commission_pct=0.001),
        asset(),
    )
    fill = sim.execute(order(quantity=10.0), bar(open_=100.0), TS)
    assert fill.commission == pytest.approx(1.0)


def test_minimum_commission_applies():
    sim = ExecutionSimulator(
        ExecutionConfig(
            apply_spread=False, slippage_model="none",
            commission_per_unit=0.01, min_commission=5.0,
        ),
        asset(),
    )
    fill = sim.execute(order(quantity=1.0), bar(), TS)
    assert fill.commission == pytest.approx(5.0)


def test_total_cost_sums_every_component():
    sim = ExecutionSimulator(
        ExecutionConfig(
            apply_spread=True, slippage_model="percent", slippage_value=0.001,
            commission_per_unit=0.1,
        ),
        asset(),
    )
    fill = sim.execute(order(quantity=10.0), bar(open_=100.0), TS)
    assert fill.total_cost == pytest.approx(
        fill.spread_cost + fill.slippage_cost + fill.commission
    )
    assert fill.total_cost > 0


# ------------------------------------------------------------------ volume cap

def test_oversized_order_is_rejected_when_volume_limits_are_enforced():
    sim = ExecutionSimulator(
        ExecutionConfig(enforce_volume_limit=True, max_volume_participation=0.1),
        asset(has_reliable_volume=True),
    )
    fill = sim.execute(order(quantity=500.0), bar(volume=1000.0), TS)
    assert fill.status is OrderStatus.REJECTED
    assert "volume" in fill.rejection_reason


def test_volume_limit_ignored_where_volume_is_meaningless():
    """Spot FX has no real volume; rejecting on it would be arbitrary."""
    sim = ExecutionSimulator(
        ExecutionConfig(enforce_volume_limit=True, max_volume_participation=0.1),
        asset(has_reliable_volume=False),
    )
    fill = sim.execute(order(quantity=500.0), bar(volume=1000.0), TS)
    assert fill.status is OrderStatus.FILLED


def test_volume_limit_off_by_default():
    sim = ExecutionSimulator(ExecutionConfig(), asset(has_reliable_volume=True))
    fill = sim.execute(order(quantity=500.0), bar(volume=1000.0), TS)
    assert fill.status is OrderStatus.FILLED


# --------------------------------------------------------------------- exits

def test_long_stop_is_detected():
    sim = ExecutionSimulator(ExecutionConfig())
    event = sim.check_exit(position(stop=98.0), bar(open_=100.0, high=101.0, low=97.0))
    assert event is not None
    assert event.reason is ExitReason.STOP_LOSS
    assert event.price == pytest.approx(98.0)


def test_long_target_is_detected():
    sim = ExecutionSimulator(ExecutionConfig())
    event = sim.check_exit(position(target=104.0), bar(open_=100.0, high=105.0, low=99.5))
    assert event.reason is ExitReason.TAKE_PROFIT
    assert event.price == pytest.approx(104.0)


def test_short_stop_is_detected():
    sim = ExecutionSimulator(ExecutionConfig())
    pos = position(SignalDirection.SHORT, entry=100.0, stop=102.0, target=96.0)
    event = sim.check_exit(pos, bar(open_=100.0, high=103.0, low=99.0))
    assert event.reason is ExitReason.STOP_LOSS


def test_position_survives_a_quiet_bar():
    sim = ExecutionSimulator(ExecutionConfig())
    assert sim.check_exit(position(), bar(open_=100.0, high=101.0, low=99.0)) is None


# ------------------------------------------------------------------------ gaps

def test_gap_through_the_stop_fills_at_the_open_not_the_stop():
    """The loss that hurts most is the one an optimistic model deletes."""
    sim = ExecutionSimulator(ExecutionConfig(honour_gaps=True))
    event = sim.check_exit(position(stop=98.0), bar(open_=95.0, high=96.0, low=94.0))

    assert event.reason is ExitReason.STOP_LOSS
    assert event.price == pytest.approx(95.0)
    assert event.price < 98.0


def test_gap_through_the_target_fills_at_the_open():
    sim = ExecutionSimulator(ExecutionConfig(honour_gaps=True))
    event = sim.check_exit(position(target=104.0), bar(open_=108.0, high=109.0, low=107.0))
    assert event.reason is ExitReason.TAKE_PROFIT
    assert event.price == pytest.approx(108.0)


def test_short_gap_through_the_stop_fills_at_the_open():
    sim = ExecutionSimulator(ExecutionConfig(honour_gaps=True))
    pos = position(SignalDirection.SHORT, entry=100.0, stop=102.0, target=96.0)
    event = sim.check_exit(pos, bar(open_=106.0, high=107.0, low=105.0))
    assert event.price == pytest.approx(106.0)


def test_ignoring_gaps_fills_at_the_stop_price():
    sim = ExecutionSimulator(ExecutionConfig(honour_gaps=False))
    event = sim.check_exit(position(stop=98.0), bar(open_=95.0, high=96.0, low=94.0))
    assert event.price == pytest.approx(98.0)


# --------------------------------------------------------- the ambiguous bar

def test_stop_wins_when_one_bar_contains_both_levels():
    """OHLC cannot say which came first; assuming the target flatters every time."""
    sim = ExecutionSimulator(ExecutionConfig(same_bar_resolution="stop_first"))
    event = sim.check_exit(
        position(stop=98.0, target=104.0), bar(open_=100.0, high=105.0, low=97.0)
    )
    assert event.reason is ExitReason.STOP_LOSS
    assert event.ambiguous is True


def test_optimistic_resolution_is_available_but_not_the_default():
    sim = ExecutionSimulator(ExecutionConfig(same_bar_resolution="target_first"))
    event = sim.check_exit(
        position(stop=98.0, target=104.0), bar(open_=100.0, high=105.0, low=97.0)
    )
    assert event.reason is ExitReason.TAKE_PROFIT
    assert event.ambiguous is True


def test_ambiguous_bars_can_be_skipped_and_counted():
    sim = ExecutionSimulator(ExecutionConfig(same_bar_resolution="skip"))
    event = sim.check_exit(
        position(stop=98.0, target=104.0), bar(open_=100.0, high=105.0, low=97.0)
    )
    assert event.reason is ExitReason.AMBIGUOUS_BAR_SKIPPED


def test_default_config_resolves_ambiguity_against_the_trader():
    assert ExecutionConfig().same_bar_resolution == "stop_first"


# ------------------------------------------------------------------ exit fills

def test_exit_fill_applies_costs():
    sim = ExecutionSimulator(
        ExecutionConfig(apply_spread=True, slippage_model="none"), asset()
    )
    fill = sim.exit_fill(position(), 98.0, TS, ExitReason.STOP_LOSS)

    # Closing a long is a sell, so it receives the bid.
    assert fill.price < 98.0
    assert fill.total_cost > 0


def test_exit_costs_are_charged_as_well_as_entry_costs():
    """Ignoring exit costs roughly halves the modelled cost of a round trip."""
    sim = ExecutionSimulator(
        ExecutionConfig(apply_spread=True, slippage_model="none", commission_per_unit=0.1),
        asset(),
    )
    entry = sim.execute(order(), bar(open_=100.0), TS)
    exit_ = sim.exit_fill(position(), 104.0, TS, ExitReason.TAKE_PROFIT)
    assert entry.total_cost > 0 and exit_.total_cost > 0


def test_short_exit_pays_the_ask():
    sim = ExecutionSimulator(
        ExecutionConfig(apply_spread=True, slippage_model="none"), asset()
    )
    pos = position(SignalDirection.SHORT, entry=100.0, stop=102.0, target=96.0)
    fill = sim.exit_fill(pos, 96.0, TS, ExitReason.TAKE_PROFIT)
    assert fill.price > 96.0   # buying back costs the ask


# --------------------------------------------------------------- position maths

def test_long_pnl_and_r_multiple():
    pos = position(entry=100.0, stop=98.0)
    assert pos.unrealised_pnl(105.0) == pytest.approx(50.0)   # 10 units x 5
    assert pos.r_multiple(102.0) == pytest.approx(1.0)        # 2 points risk


def test_short_pnl_and_r_multiple():
    pos = position(SignalDirection.SHORT, entry=100.0, stop=102.0, target=96.0)
    assert pos.unrealised_pnl(95.0) == pytest.approx(50.0)
    assert pos.r_multiple(98.0) == pytest.approx(1.0)


def test_r_multiple_is_none_without_risk():
    pos = position(entry=100.0, stop=100.0)
    assert pos.r_multiple(105.0) is None
