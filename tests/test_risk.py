"""Position sizing and risk limits.

Section 15 of the brief requires every calculation to be explicit and testable.
The arithmetic tests below are deliberately hand-checkable: $10,000 at 1% risk
with a $10 stop distance is 10 units, and if that ever stops being true the
whole risk framework is unreliable.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.config.models import AssetConfig, RiskConfig
from app.data.validators.quality import QualityStatus
from app.portfolio.portfolio import Portfolio
from app.risk.engine import RiskEngine, rolling_correlations
from app.risk.models import RiskBlockReason, RiskVerdict
from app.signals.models import Signal, SignalDirection


def asset(**overrides) -> AssetConfig:
    base = {
        "symbol": "XAUUSD", "name": "Gold", "asset_class": "METAL",
        "quote_currency": "USD", "tick_size": 0.01, "typical_spread": 0.3,
    }
    return AssetConfig(**{**base, **overrides})


def signal(
    symbol="XAUUSD",
    direction=SignalDirection.LONG,
    entry=3000.0,
    stop=2990.0,
    strategy="test",
    confidence=0.8,
) -> Signal:
    return Signal(
        strategy=strategy, symbol=symbol, timeframe="1D", direction=direction,
        confidence=confidence, entry_price=entry, stop_loss=stop,
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
    )


def open_position(portfolio: Portfolio, symbol="EURUSD", risk=100.0, strategy="other",
                  direction=SignalDirection.LONG) -> None:
    from app.execution.models import Fill, Order, OrderSide, OrderStatus, Position

    position = Position(
        symbol=symbol, direction=direction, quantity=1.0, entry_price=100.0,
        entry_time=pd.Timestamp("2024-01-01", tz="UTC"), stop_loss=90.0,
        strategy=strategy, risk_amount=risk,
    )
    fill = Fill(
        order=Order(symbol=symbol, side=OrderSide.BUY, quantity=1.0),
        status=OrderStatus.FILLED, filled_quantity=1.0, price=100.0,
    )
    portfolio.open_position(position, fill)


def unleveraged(**overrides) -> RiskConfig:
    """Config with the notional cap lifted, to isolate the limit under test.

    The default per-position notional cap is 50% of equity, which binds hard on
    high-priced instruments: 10 units of $3,000 gold is $30,000 of notional on
    a $10,000 account. That interaction is correct and is pinned by its own
    test below - but it obscures the risk arithmetic everywhere else, so tests
    aimed at other limits lift it out of the way.
    """
    return RiskConfig(
        **{
            "max_position_notional_pct": 100.0,
            "max_portfolio_notional_pct": 100.0,
            **overrides,
        }
    )


def risk_engine(config: RiskConfig | None = None, **kwargs) -> RiskEngine:
    """A risk engine that has been told its data passed the quality gate.

    The engine refuses to size anything against a series whose grade it was
    never given, so a test aimed at some *other* limit has to state the grade
    or it only ever measures the quality gate. The gate's own tests below
    construct :class:`RiskEngine` directly.
    """
    return RiskEngine(config, data_quality=QualityStatus.PASS, **kwargs)


@pytest.fixture
def portfolio() -> Portfolio:
    return Portfolio(10_000.0)


@pytest.fixture
def engine() -> RiskEngine:
    return risk_engine(unleveraged(), trading_enabled=True)


# ----------------------------------------------------------- core arithmetic

def test_position_size_matches_the_brief_worked_example(engine, portfolio):
    """$10,000 at 1% risk, entry 3000, stop 2990 -> $100 / $10 = 10 units."""
    decision = engine.evaluate(signal(entry=3000.0, stop=2990.0), portfolio, equity=10_000.0)

    assert decision.approved
    assert decision.risk_amount == pytest.approx(100.0)
    assert decision.risk_per_unit == pytest.approx(10.0)
    assert decision.quantity == pytest.approx(10.0)


def test_a_wider_stop_gives_a_smaller_position(engine, portfolio):
    """Risk stays constant; only the size moves. That is the whole point."""
    tight = engine.evaluate(signal(entry=3000.0, stop=2990.0), portfolio, equity=10_000.0)
    wide = engine.evaluate(signal(entry=3000.0, stop=2950.0), portfolio, equity=10_000.0)

    assert wide.quantity < tight.quantity
    assert wide.risk_amount == pytest.approx(tight.risk_amount)


def test_size_scales_with_equity(engine, portfolio):
    small = engine.evaluate(signal(), portfolio, equity=10_000.0)
    large = engine.evaluate(signal(), portfolio, equity=100_000.0)
    assert large.quantity == pytest.approx(small.quantity * 10)


def test_short_signals_size_identically(engine, portfolio):
    decision = engine.evaluate(
        signal(direction=SignalDirection.SHORT, entry=3000.0, stop=3010.0),
        portfolio, equity=10_000.0,
    )
    assert decision.quantity == pytest.approx(10.0)


def test_fixed_risk_sizing():
    engine = risk_engine(
        unleveraged(sizing_method="fixed_risk", fixed_risk_amount=250.0), trading_enabled=True
    )
    decision = engine.evaluate(
        signal(entry=3000.0, stop=2990.0), Portfolio(10_000.0), equity=10_000.0
    )
    assert decision.risk_amount == pytest.approx(250.0)
    assert decision.quantity == pytest.approx(25.0)


def test_decision_records_the_arithmetic(engine, portfolio):
    decision = engine.evaluate(signal(), portfolio, equity=10_000.0)
    joined = " ".join(decision.reasoning)
    assert "Base risk" in joined
    assert "Final size" in joined


# ----------------------------------------------------------------- kill switch

def test_kill_switch_blocks_everything(portfolio):
    """The graph still analyses; nothing is opened."""
    engine = risk_engine(RiskConfig(), trading_enabled=False)
    decision = engine.evaluate(signal(), portfolio, equity=10_000.0)

    assert not decision.approved
    assert decision.block_reason is RiskBlockReason.KILL_SWITCH
    assert decision.quantity == 0.0


def test_kill_switch_defaults_to_disabled():
    assert RiskEngine(RiskConfig()).trading_enabled is False


# ------------------------------------------------------------- invalid inputs

def test_wait_signal_is_rejected(engine, portfolio):
    wait = Signal.wait("test", "XAUUSD", "1D")
    decision = engine.evaluate(wait, portfolio, equity=10_000.0)
    assert decision.block_reason is RiskBlockReason.NO_SIGNAL


def test_zero_equity_is_rejected(engine, portfolio):
    decision = engine.evaluate(signal(), portfolio, equity=0.0)
    assert decision.block_reason is RiskBlockReason.INSUFFICIENT_CASH


# --------------------------------------------------------------- position caps

def test_max_concurrent_positions(portfolio):
    engine = risk_engine(unleveraged(max_concurrent_positions=2), trading_enabled=True)
    open_position(portfolio, "EURUSD")
    open_position(portfolio, "GBPUSD")

    decision = engine.evaluate(signal(), portfolio, equity=10_000.0)
    assert decision.block_reason is RiskBlockReason.MAX_POSITIONS


def test_max_positions_per_symbol(portfolio):
    engine = risk_engine(unleveraged(max_positions_per_symbol=1), trading_enabled=True)
    open_position(portfolio, "XAUUSD")

    decision = engine.evaluate(signal(symbol="XAUUSD"), portfolio, equity=10_000.0)
    assert decision.block_reason is RiskBlockReason.MAX_POSITIONS_PER_SYMBOL


def test_per_position_notional_cap(portfolio):
    """A very tight stop would otherwise imply an enormous position."""
    engine = risk_engine(RiskConfig(max_position_notional_pct=0.2), trading_enabled=True)
    decision = engine.evaluate(
        signal(entry=100.0, stop=99.99), portfolio, equity=10_000.0, asset=asset()
    )
    assert decision.verdict is RiskVerdict.REDUCED
    assert decision.notional <= 2000.0 + 1e-6


def test_notional_cap_binds_on_high_priced_instruments(portfolio):
    """Risk sizing alone can imply heavy leverage, and the cap must stop it.

    The brief's worked example - $100 risk over a $10 stop distance - gives 10
    units. On $3,000 gold that is $30,000 of notional against $10,000 of
    equity, which is 3x leverage arrived at without anyone deciding to use
    leverage. The notional cap is what prevents risk-based sizing from
    quietly implying it.
    """
    engine = risk_engine(RiskConfig(max_position_notional_pct=0.5), trading_enabled=True)
    decision = engine.evaluate(
        signal(entry=3000.0, stop=2990.0), portfolio, equity=10_000.0, asset=asset()
    )

    assert decision.verdict is RiskVerdict.REDUCED
    assert decision.notional <= 5000.0 + 1e-6
    assert decision.risk_amount < 100.0  # less risk than requested, never more
    assert any("notional" in r for r in decision.reasoning)


# ---------------------------------------------------------------- risk budgets

def test_portfolio_risk_budget_trims_the_position(portfolio):
    engine = risk_engine(
        unleveraged(max_portfolio_risk=0.015, risk_per_trade=0.01), trading_enabled=True
    )
    open_position(portfolio, "EURUSD", risk=100.0)

    decision = engine.evaluate(signal(), portfolio, equity=10_000.0)
    assert decision.verdict is RiskVerdict.REDUCED
    assert decision.risk_amount == pytest.approx(50.0)


def test_exhausted_portfolio_budget_blocks(portfolio):
    engine = risk_engine(unleveraged(max_portfolio_risk=0.01), trading_enabled=True)
    open_position(portfolio, "EURUSD", risk=100.0)

    decision = engine.evaluate(signal(), portfolio, equity=10_000.0)
    assert decision.block_reason is RiskBlockReason.PORTFOLIO_RISK


def test_per_strategy_budget_is_enforced(portfolio):
    engine = risk_engine(
        unleveraged(max_risk_per_strategy=0.01, max_portfolio_risk=0.10), trading_enabled=True
    )
    open_position(portfolio, "EURUSD", risk=100.0, strategy="trend_following")

    decision = engine.evaluate(
        signal(strategy="trend_following"), portfolio, equity=10_000.0
    )
    assert decision.block_reason is RiskBlockReason.STRATEGY_RISK


def test_a_different_strategy_has_its_own_budget(portfolio):
    engine = risk_engine(
        unleveraged(max_risk_per_strategy=0.01, max_portfolio_risk=0.10), trading_enabled=True
    )
    open_position(portfolio, "EURUSD", risk=100.0, strategy="trend_following")

    decision = engine.evaluate(signal(strategy="momentum"), portfolio, equity=10_000.0)
    assert decision.approved


# -------------------------------------------------------------- drawdown rules

def test_drawdown_reduces_position_size(portfolio):
    engine = risk_engine(
        unleveraged(drawdown_reduce_threshold=0.10, drawdown_reduce_factor=0.5),
        trading_enabled=True,
    )
    # A drawdown built up over previous days, not lost today - otherwise the
    # daily-loss limit fires first and blocks the trade outright.
    portfolio._peak_equity = 10_000.0
    portfolio._day_start_equity = 8_800.0

    decision = engine.evaluate(signal(), portfolio, equity=8_800.0)  # 12% drawdown
    assert decision.verdict is RiskVerdict.REDUCED
    assert decision.risk_amount == pytest.approx(0.5 * 0.01 * 8_800.0)


def test_max_drawdown_halts_new_trades(portfolio):
    engine = risk_engine(unleveraged(max_drawdown_limit=0.20), trading_enabled=True)
    portfolio._peak_equity = 10_000.0
    portfolio._day_start_equity = 7_500.0

    decision = engine.evaluate(signal(), portfolio, equity=7_500.0)  # 25% drawdown
    assert decision.block_reason is RiskBlockReason.MAX_DRAWDOWN


def test_daily_loss_limit_halts_trading(portfolio):
    engine = risk_engine(unleveraged(daily_loss_limit=0.03), trading_enabled=True)
    portfolio._day_start_equity = 10_000.0

    decision = engine.evaluate(signal(), portfolio, equity=9_600.0)  # 4% down today
    assert decision.block_reason is RiskBlockReason.DAILY_LOSS_LIMIT


def test_drawdown_thresholds_must_be_ordered():
    with pytest.raises(ValueError, match="must be below"):
        RiskConfig(drawdown_reduce_threshold=0.30, max_drawdown_limit=0.20)


# ----------------------------------------------------------------- correlation

def test_correlated_positions_share_one_budget(portfolio):
    """Three correlated longs are one bet, and sizing them separately triples it."""
    engine = risk_engine(
        unleveraged(max_correlated_risk=0.015, correlation_threshold=0.7),
        trading_enabled=True,
    )
    open_position(portfolio, "EURUSD", risk=100.0, direction=SignalDirection.LONG)

    correlations = pd.DataFrame(
        {"XAUUSD": [1.0, 0.85], "EURUSD": [0.85, 1.0]}, index=["XAUUSD", "EURUSD"]
    )
    decision = engine.evaluate(
        signal(symbol="XAUUSD"), portfolio, equity=10_000.0, correlations=correlations
    )
    assert decision.verdict is RiskVerdict.REDUCED
    assert decision.risk_amount == pytest.approx(50.0)


def test_uncorrelated_positions_do_not_share_a_budget(portfolio):
    engine = risk_engine(unleveraged(max_correlated_risk=0.015), trading_enabled=True)
    open_position(portfolio, "EURUSD", risk=100.0)

    correlations = pd.DataFrame(
        {"XAUUSD": [1.0, 0.05], "EURUSD": [0.05, 1.0]}, index=["XAUUSD", "EURUSD"]
    )
    decision = engine.evaluate(
        signal(symbol="XAUUSD"), portfolio, equity=10_000.0, correlations=correlations
    )
    assert decision.verdict is RiskVerdict.APPROVED


def test_opposite_positions_in_negatively_correlated_symbols_are_concentration(portfolio):
    """Short EURUSD and long USDCHF is one dollar trade wearing two hats."""
    engine = risk_engine(
        unleveraged(max_correlated_risk=0.015, correlation_threshold=0.7),
        trading_enabled=True,
    )
    open_position(portfolio, "USDCHF", risk=100.0, direction=SignalDirection.SHORT)

    correlations = pd.DataFrame(
        {"EURUSD": [1.0, -0.9], "USDCHF": [-0.9, 1.0]}, index=["EURUSD", "USDCHF"]
    )
    decision = engine.evaluate(
        signal(symbol="EURUSD", direction=SignalDirection.LONG),
        portfolio, equity=10_000.0, correlations=correlations,
    )
    assert decision.verdict is RiskVerdict.REDUCED


def test_missing_correlations_are_treated_as_unknown_not_zero(portfolio, engine):
    open_position(portfolio, "EURUSD", risk=100.0)
    decision = engine.evaluate(signal(), portfolio, equity=10_000.0, correlations=None)
    assert decision.approved


def test_rolling_correlations_uses_returns_not_prices():
    """Two rising price series correlate; their returns need not."""
    index = pd.date_range("2020-01-01", periods=200, freq="1D", tz="UTC")
    rising_a = pd.Series(range(100, 300), index=index, dtype="float64")
    rising_b = pd.Series([100 + (i % 7) + i * 0.5 for i in range(200)], index=index)

    result = rolling_correlations({"A": rising_a, "B": rising_b}, lookback=60)
    assert not result.empty
    assert abs(result.loc["A", "B"]) < 0.99


def test_rolling_correlations_needs_two_symbols():
    index = pd.date_range("2020-01-01", periods=100, freq="1D", tz="UTC")
    single = {"A": pd.Series(range(100), index=index, dtype="float64")}
    assert rolling_correlations(single).empty


# ---------------------------------------------------------------- serialisation

def test_decision_serialises(engine, portfolio):
    payload = engine.evaluate(signal(), portfolio, equity=10_000.0).to_dict()
    assert payload["verdict"] in {"APPROVED", "REDUCED"}
    assert payload["quantity"] > 0
    assert isinstance(payload["reasoning"], list)


def test_rejection_describes_itself(engine, portfolio):
    engine.trading_enabled = False
    text = engine.evaluate(signal(), portfolio, equity=10_000.0).describe()
    assert "REJECTED" in text and "KILL_SWITCH" in text


# --------------------------------------------------------------- data quality

def test_unstated_data_quality_is_refused(portfolio):
    """'Nobody checked' must not size the same position as 'checked and passed'."""
    engine = RiskEngine(unleveraged(), trading_enabled=True)
    decision = engine.evaluate(signal(), portfolio, equity=10_000.0)

    assert not decision.approved
    assert decision.block_reason is RiskBlockReason.DATA_QUALITY_UNKNOWN
    assert decision.quantity == 0.0


def test_the_unknown_placeholder_is_not_a_pass(portfolio):
    """The literal string several call sites use as a placeholder is not a grade."""
    engine = RiskEngine(unleveraged(), trading_enabled=True, data_quality="UNKNOWN")
    decision = engine.evaluate(signal(), portfolio, equity=10_000.0)

    assert decision.block_reason is RiskBlockReason.DATA_QUALITY_UNKNOWN


def test_failed_data_quality_blocks_sizing(portfolio):
    engine = RiskEngine(unleveraged(), trading_enabled=True, data_quality=QualityStatus.FAIL)
    decision = engine.evaluate(signal(), portfolio, equity=10_000.0)

    assert not decision.approved
    assert decision.block_reason is RiskBlockReason.DATA_QUALITY


def test_warning_data_passes_by_default_and_is_recorded(portfolio):
    """Most long historical windows carry a warning; blocking them all is useless."""
    engine = RiskEngine(unleveraged(), trading_enabled=True, data_quality=QualityStatus.WARNING)
    decision = engine.evaluate(signal(), portfolio, equity=10_000.0)

    assert decision.approved
    assert decision.metrics["data_quality"] == "WARNING"


def test_warning_data_can_be_configured_to_block(portfolio):
    engine = RiskEngine(
        unleveraged(block_on_data_quality_warning=True),
        trading_enabled=True,
        data_quality=QualityStatus.WARNING,
    )
    decision = engine.evaluate(signal(), portfolio, equity=10_000.0)

    assert not decision.approved
    assert decision.block_reason is RiskBlockReason.DATA_QUALITY


def test_unknown_quality_can_be_accepted_deliberately(portfolio):
    """Turning the check off is allowed; doing it by accident is what is not."""
    engine = RiskEngine(unleveraged(require_known_data_quality=False), trading_enabled=True)
    assert engine.evaluate(signal(), portfolio, equity=10_000.0).approved


def test_per_call_quality_overrides_the_engine_default(portfolio):
    engine = RiskEngine(unleveraged(), trading_enabled=True, data_quality=QualityStatus.PASS)
    decision = engine.evaluate(
        signal(), portfolio, equity=10_000.0, data_quality=QualityStatus.FAIL
    )

    assert not decision.approved
    assert decision.block_reason is RiskBlockReason.DATA_QUALITY


def test_a_failed_grade_cannot_be_talked_past_by_the_kill_switch(portfolio):
    """Both gates block; the point is that neither depends on the other."""
    engine = RiskEngine(unleveraged(), trading_enabled=False, data_quality=QualityStatus.FAIL)
    assert not engine.evaluate(signal(), portfolio, equity=10_000.0).approved
