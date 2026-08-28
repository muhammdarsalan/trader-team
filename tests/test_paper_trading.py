"""Tests for the paper-trading loop, persistence, performance and dashboard view.

Three properties carry most of the weight here.

**The paper engine and the backtester must agree.** They are two drivers of one
execution model, and a divergence between them would be invisible: both produce
plausible trades, and only a bar-for-bar comparison shows that the paper run and
the backtest have stopped describing the same system.
:func:`test_paper_engine_reproduces_backtester_trades` is that comparison.

**A restart must change nothing.** A paper session is long-lived, so the
interesting bugs are the ones that only appear across a process boundary: a
drawdown peak that resets and quietly widens the risk limit, an excursion that
restarts and makes every surviving trade look like it never went against the
entry, a replayed bar that opens a second position.
:func:`test_restart_reproduces_continuous_run_exactly` asserts the whole
outcome, not a sampled field.

**Nothing may be fabricated.** An engine with no data must not report itself
healthy, must not colour a stage green, and must not present a starting balance
as a measurement. Those assertions are the point of the ``no_fabrication``
tests: a monitoring surface that looks fine when the feed is dead is worse than
no monitoring surface at all.

The handful of tests that replay a multi-year frame end to end are marked
``slow`` so ``pytest -m "not network and not slow"`` stays a fast loop. They are
still part of the default ``pytest -m "not network"`` run: the properties above
cannot be demonstrated on ten bars.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from app.backtest.engine import Backtester
from app.config.loader import get_config, override_config
from app.config.models import RiskConfig
from app.data.loaders.synthetic import SyntheticProvider
from app.data.schema import MarketData, coerce_schema
from app.data.validators.quality import DataQualityEngine
from app.execution.models import ExitReason
from app.paper_trading.engine import PaperTradingEngine
from app.paper_trading.graph_view import build_graph_visualization
from app.paper_trading.performance import build_performance, summarise_trades
from app.paper_trading.records import assess_freshness
from app.portfolio.portfolio import Portfolio
from app.risk.engine import RiskEngine
from app.risk.models import RiskBlockReason
from app.signals.models import Signal, SignalDirection
from app.utils.paths import paper_state_path
from dashboard.view import build_view, quality_rows
from tests.conftest import make_ohlcv

# --------------------------------------------------------------------- helpers


def _cfg(config_dir, **overrides):
    """Config with the kill switch off, so orders actually reach the simulator.

    A paper test with ``trading_enabled: false`` exercises the analysis path and
    nothing else: risk refuses every signal on the kill switch before sizing, so
    no fill, exit or P&L is ever produced and the test proves only that the
    engine does not crash.
    """
    cfg = get_config(config_dir)
    platform_updates = {"trading_enabled": True, **overrides.pop("platform", {})}
    sections = {"platform": cfg.platform.model_copy(update=platform_updates), **overrides}
    return override_config(cfg, **sections)


def _market(symbol="XAUUSD", periods=80, seed=42, start="2020-01-01"):
    return MarketData(
        symbol=symbol,
        timeframe="1D",
        df=make_ohlcv(periods=periods, start=start, seed=seed),
        provider="synthetic",
    )


def _real_shaped_market(symbol="XAUUSD", start="2015-01-01", end="2019-12-31"):
    """A long synthetic series, long enough to clear warm-up and trade."""
    return SyntheticProvider().get_historical_data(symbol, "1D", start, end)


def _engine(config_dir, **kwargs):
    kwargs.setdefault("symbol", "XAUUSD")
    kwargs.setdefault("timeframe", "1D")
    return PaperTradingEngine(config=_cfg(config_dir), **kwargs)


def _frame(rows, start="2021-01-04"):
    """A hand-built frame, for the execution edge cases.

    Generated data cannot be relied on to contain a gap through a stop or a bar
    holding both the stop and the target; those have to be constructed.
    """
    index = pd.bdate_range(start=start, periods=len(rows), tz="UTC")
    return coerce_schema(
        pd.DataFrame(
            rows,
            index=index,
            columns=["open", "high", "low", "close", "volume"],
        )
    )


def _trade_signature(portfolio: Portfolio) -> str:
    """A digest of every field of every closed trade.

    Comparing a digest rather than a trade count is deliberate: two runs can
    close the same number of trades at different prices, and a count would call
    that agreement.
    """
    parts = []
    for trade in portfolio.closed_trades:
        parts.append(
            "|".join(
                str(x)
                for x in (
                    trade.symbol,
                    trade.strategy,
                    trade.direction,
                    f"{trade.quantity:.10f}",
                    trade.entry_time,
                    f"{trade.entry_price:.10f}",
                    trade.exit_time,
                    f"{trade.exit_price:.10f}",
                    trade.exit_reason,
                    f"{trade.net_pnl:.10f}",
                    f"{trade.costs:.10f}",
                    trade.bars_held,
                    f"{trade.mae:.10f}",
                    f"{trade.mfe:.10f}",
                    trade.entry_regime,
                )
            )
        )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


# ------------------------------------------------------- backtester agreement


@pytest.mark.slow
def test_paper_engine_reproduces_backtester_trades(config_dir):
    """Bar-for-bar agreement with the Phase 3 backtester over the same frame.

    The paper engine exists to run the *same* execution model against a live
    feed, not a second one. If these diverge, one of them is wrong and the
    difference will not show up as an error - only as two sets of plausible
    numbers that no longer describe the same system.

    The backtester closes open positions at the final bar and the paper engine
    does not (its session continues), so its trade list may be one longer. Every
    trade they share must be identical.
    """
    cfg = _cfg(config_dir)
    data = _real_shaped_market(end="2018-12-31")

    result = Backtester(cfg, asset=cfg.assets.get("XAUUSD"), experiment_id="TEST").run(
        data, quality_status="PASS", trading_enabled=True
    )
    engine = PaperTradingEngine(config=cfg, symbol="XAUUSD", timeframe="1D")
    engine.catch_up(data, data_quality="PASS")

    backtest = result.trades.reset_index(drop=True)
    paper = engine.portfolio.trades_frame().reset_index(drop=True)

    assert not paper.empty, "the paper run closed no trades, so nothing was compared"
    assert len(backtest) - len(paper) in (0, 1), (
        f"trade counts differ by more than the backtester's end-of-run close: "
        f"backtest {len(backtest)}, paper {len(paper)}"
    )

    columns = [
        "entry_time", "entry_price", "exit_time", "exit_price", "direction",
        "quantity", "stop_loss", "take_profit", "exit_reason", "net_pnl",
        "costs", "bars_held", "mae", "mfe", "entry_regime",
    ]
    shared = min(len(backtest), len(paper))
    for index in range(shared):
        for column in columns:
            expected, actual = backtest.iloc[index][column], paper.iloc[index][column]
            if isinstance(expected, float) and pd.notna(expected):
                assert actual == pytest.approx(expected, abs=1e-9), (
                    f"trade {index} column {column}: backtest {expected!r} "
                    f"vs paper {actual!r}"
                )
            else:
                assert actual == expected, (
                    f"trade {index} column {column}: backtest {expected!r} "
                    f"vs paper {actual!r}"
                )


@pytest.mark.slow
def test_replay_is_deterministic_across_processes(config_dir):
    """Identical inputs must give an identical outcome, run to run.

    This regressed once for a reason worth remembering: the synthetic provider
    seeded itself from ``hash(symbol)``, and Python randomises string hashing per
    process. Every run produced different bars while the provider's docstring
    promised reproducibility, and the defect was invisible inside a single
    process - which is where it was always tested. Hence the explicit digest of
    the *data* as well as of the trades.
    """
    cfg = _cfg(config_dir)
    digests = set()
    data_digests = set()
    for _ in range(3):
        data = _real_shaped_market(end="2017-12-31")
        data_digests.add(hashlib.sha256(data.df.round(10).to_csv().encode()).hexdigest())
        engine = PaperTradingEngine(config=cfg, symbol="XAUUSD", timeframe="1D")
        engine.catch_up(data, data_quality="PASS")
        digests.add((_trade_signature(engine.portfolio), round(engine.portfolio.cash, 9)))

    assert len(data_digests) == 1, "the provider returned different bars for one request"
    assert len(digests) == 1, "the same replay produced different trades"


def test_synthetic_provider_seed_is_process_stable():
    """The per-symbol seed offset must not depend on the interpreter's hash seed."""
    from app.data.loaders.synthetic import _symbol_offset

    # Values pinned so a change to the hashing scheme is a deliberate act rather
    # than a silent one: a new scheme changes every historical replay.
    assert _symbol_offset("XAUUSD") == _symbol_offset("xauusd")
    assert _symbol_offset("XAUUSD") != _symbol_offset("EURUSD")
    assert 0 <= _symbol_offset("XAUUSD") < 10_000
    expected = (
        int.from_bytes(
            hashlib.blake2s(b"XAUUSD", digest_size=4).digest(), "big"
        )
        % 10_000
    )
    assert _symbol_offset("XAUUSD") == expected


# ------------------------------------------------------------------ warm-up


def test_no_decision_inside_warmup(config_dir):
    """Bars inside the warm-up are marked and processed, but never decided.

    They still have to be *processed*: equity is marked, and an inherited open
    position still needs its stop checked. What must not happen is a decision
    taken from indicators that are not yet defined.
    """
    engine = _engine(config_dir)
    data = _market(periods=60)
    engine.catch_up(data, data_quality="PASS")

    assert engine.state["bars_processed"] == 60
    assert engine.state["bars_decided"] == 0, "a decision was taken inside warm-up"
    assert engine.state["latest_decision"] == "WARMUP"
    assert not engine.portfolio.positions
    assert not engine.portfolio.closed_trades
    assert engine.portfolio.snapshots, "equity was not marked during warm-up"

    view = engine.state["latest_state"]
    assert view["warm"] is False
    assert "warm-up" in view["skipped_reason"]


def test_warmup_honours_backtest_config_override(config_dir):
    """``backtest.warmup_bars`` governs the paper run too.

    If the two engines resolved warm-up differently, an operator who lengthened
    it to stabilise a backtest would find the paper run still deciding on bars
    the backtest had discarded - and the two results would silently stop being
    comparable.
    """
    cfg = get_config(config_dir)
    cfg = override_config(
        cfg,
        platform=cfg.platform.model_copy(update={"trading_enabled": True}),
        backtest=cfg.backtest.model_copy(update={"warmup_bars": 400}),
    )
    engine = PaperTradingEngine(config=cfg, symbol="XAUUSD", timeframe="1D")
    data = _real_shaped_market(end="2017-12-31")
    features = engine.feature_engine.compute(data, engine.asset)

    assert features.warmup_bars < 400, "fixture no longer exercises the override"
    assert engine.warmup_bars(features) == 400

    engine.catch_up(data, data_quality="PASS")
    assert engine.state["bars_decided"] == max(0, len(data.df) - 400)


# ----------------------------------------------------------- fills and exits


def _long_setup(config_dir, *, stop, target, bars):
    """Drive a single known LONG position through ``bars`` and return the engine.

    The graph is bypassed for the execution edge cases: making a strategy emit a
    signal with a chosen stop and target on a chosen bar is indirect and
    fragile, and what is under test is the fill and exit arithmetic, not signal
    generation. The queued-order shape is exactly what ``_queue_order`` writes.
    """
    engine = _engine(config_dir)
    data = MarketData(symbol="XAUUSD", timeframe="1D", df=_frame(bars), provider="synthetic")
    engine._state["pending_orders"] = [
        {
            "fill_key": "test-order",
            "symbol": "XAUUSD",
            "side": "BUY",
            "quantity": 2.0,
            "created_at": data.df.index[0].isoformat(),
            "queued_on_bar": data.df.index[0].isoformat(),
            "strategy": "test",
            "reference_price": float(data.df.iloc[0]["close"]),
            "stop_loss": stop,
            "take_profit": target,
            "risk_amount": 100.0,
            "regime": "TRENDING_UP",
            "contributing": ["momentum", "breakout"],
            "aggregation_method": "WEIGHTED",
            "metadata": {},
        }
    ]
    return engine, data


def test_order_fills_at_next_bar_open_with_costs(config_dir):
    """A queued order fills at the next bar's OPEN, worse than the reference.

    Both directions of this matter. Filling at the signal bar's close would be
    look-ahead - the price had already traded. And a fill exactly at the open
    would mean the spread and slippage models are not running, which is how a
    simulated result quietly becomes a costless one.
    """
    engine, data = _long_setup(
        config_dir,
        stop=95.0,
        target=115.0,
        bars=[
            [100.0, 101.0, 99.0, 100.0, 5_000.0],
            [102.0, 103.0, 101.0, 102.5, 5_000.0],
        ],
    )
    engine.process_bar(data, timestamp=data.df.index[1], features=None, data_quality="PASS")

    assert len(engine.portfolio.positions) == 1
    position = engine.portfolio.positions[0]
    assert position.entry_time == data.df.index[1]
    assert position.entry_price > 102.0, "a buy filled at or better than the open"
    assert position.stop_loss == 95.0
    assert position.take_profit == 115.0
    assert position.entry_costs > 0.0

    fill = engine.state["fills"][-1]
    assert fill["reference_price"] == pytest.approx(102.0)
    assert fill["spread_cost"] > 0.0
    assert fill["total_cost"] == pytest.approx(
        fill["spread_cost"] + fill["slippage_cost"] + fill["commission"]
    )
    # Risk is recomputed from the actual fill, not the size the signal assumed.
    assert position.risk_amount == pytest.approx(
        abs(position.entry_price - 95.0) * position.quantity
    )
    assert position.metadata["contributing"] == ["momentum", "breakout"]


def test_stop_exit_closes_position_and_records_trade(config_dir):
    """A touched stop closes the position, and the trade lands in every record."""
    engine, data = _long_setup(
        config_dir,
        stop=99.0,
        target=130.0,
        bars=[
            [100.0, 101.0, 99.5, 100.0, 5_000.0],
            [102.0, 103.0, 101.0, 102.0, 5_000.0],
            [101.0, 101.5, 97.0, 98.0, 5_000.0],
        ],
    )
    engine.process_bar(data, timestamp=data.df.index[1], data_quality="PASS")
    engine.process_bar(data, timestamp=data.df.index[2], data_quality="PASS")

    assert not engine.portfolio.positions
    assert len(engine.portfolio.closed_trades) == 1
    trade = engine.portfolio.closed_trades[0]
    assert trade.exit_reason is ExitReason.STOP_LOSS
    # The stop fills at its level, then pays the spread and slippage on the way
    # out - so slightly worse than 99.
    assert trade.exit_price < 99.0
    assert trade.net_pnl < 0
    assert trade.bars_held == 1
    assert trade.exit_time == data.df.index[2]

    assert engine.state["closed_trades"][-1]["exit_reason"] == "STOP_LOSS"
    assert any(e["event"] == "paper_exit" for e in engine.state["recent_events"])
    assert engine.state["fills"][-1]["closes_position"] is True


def test_target_exit_records_take_profit(config_dir):
    engine, data = _long_setup(
        config_dir,
        stop=95.0,
        target=105.0,
        bars=[
            [100.0, 101.0, 99.0, 100.0, 5_000.0],
            [100.5, 101.0, 100.0, 100.5, 5_000.0],
            [101.0, 106.0, 100.5, 105.5, 5_000.0],
        ],
    )
    engine.process_bar(data, timestamp=data.df.index[1], data_quality="PASS")
    engine.process_bar(data, timestamp=data.df.index[2], data_quality="PASS")

    trade = engine.portfolio.closed_trades[0]
    assert trade.exit_reason is ExitReason.TAKE_PROFIT
    assert trade.net_pnl > 0


def test_gap_through_stop_fills_at_the_open_not_the_stop(config_dir):
    """A bar that opens below the stop fills there, at the worse price.

    Filling a gapped stop at the stop price deletes exactly the losses that hurt
    most, which is the single most flattering assumption a simulator can make.
    """
    engine, data = _long_setup(
        config_dir,
        stop=99.0,
        target=130.0,
        bars=[
            [100.0, 101.0, 99.5, 100.0, 5_000.0],
            [102.0, 103.0, 101.5, 102.0, 5_000.0],
            [90.0, 92.0, 89.0, 91.0, 5_000.0],
        ],
    )
    engine.process_bar(data, timestamp=data.df.index[1], data_quality="PASS")
    engine.process_bar(data, timestamp=data.df.index[2], data_quality="PASS")

    trade = engine.portfolio.closed_trades[0]
    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.exit_price < 91.0, (
        "the gapped stop filled at or better than the open, deleting the gap loss"
    )
    assert trade.exit_price < 99.0


def test_entry_gapping_through_its_own_stop_is_abandoned(config_dir):
    """An entry that fills already beyond its stop is not opened.

    Such a position has a negative risk distance and would be closed on the bar
    that opened it. The backtester refuses it; so must this.
    """
    engine, data = _long_setup(
        config_dir,
        stop=99.0,
        target=130.0,
        bars=[
            [100.0, 101.0, 99.5, 100.0, 5_000.0],
            [90.0, 92.0, 89.0, 91.0, 5_000.0],
        ],
    )
    engine.process_bar(data, timestamp=data.df.index[1], data_quality="PASS")

    assert not engine.portfolio.positions
    assert not engine.portfolio.closed_trades
    assert not engine.state["pending_orders"], "the order was carried to a later bar"
    events = [e for e in engine.state["recent_events"] if e["event"] == "paper_order_abandoned"]
    assert events, "the abandoned entry was not recorded"
    assert "gapped through the stop" in events[0]["message"]


def test_ambiguous_bar_is_resolved_against_the_trader_and_recorded(config_dir):
    """One bar holding both stop and target resolves to the stop, and says so.

    OHLC cannot say which level was touched first. Assuming the target believes
    the flattering possibility every single time; the configured resolution
    decides, and the fact that a resolution was needed is recorded so a reader
    can see which trades rest on it.
    """
    engine, data = _long_setup(
        config_dir,
        stop=99.0,
        target=105.0,
        bars=[
            [100.0, 101.0, 99.5, 100.0, 5_000.0],
            [101.0, 101.5, 100.5, 101.0, 5_000.0],
            [101.0, 106.0, 97.0, 100.0, 5_000.0],
        ],
    )
    assert engine.config.execution.same_bar_resolution == "stop_first"

    engine.process_bar(data, timestamp=data.df.index[1], data_quality="PASS")
    engine.process_bar(data, timestamp=data.df.index[2], data_quality="PASS")

    trade = engine.portfolio.closed_trades[0]
    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert engine.state["ambiguous_exits"] == 1
    assert engine.state["closed_trades"][-1]["ambiguous_bar"] is True
    exit_event = [e for e in engine.state["recent_events"] if e["event"] == "paper_exit"][-1]
    assert exit_event["ambiguous_bar"] is True
    assert "unknowable from OHLC" in exit_event["message"]
    assert engine.performance_view()["overall"]["ambiguous_exits"] == 1


def test_order_without_a_stop_is_refused(config_dir):
    """An unstopped order has undefined risk and cannot be sized or limited."""
    engine, data = _long_setup(
        config_dir,
        stop=None,
        target=None,
        bars=[
            [100.0, 101.0, 99.0, 100.0, 5_000.0],
            [102.0, 103.0, 101.0, 102.0, 5_000.0],
        ],
    )
    engine._state["pending_orders"][0]["stop_loss"] = None
    engine.process_bar(data, timestamp=data.df.index[1], data_quality="PASS")

    assert not engine.portfolio.positions
    assert any("carried no stop" in e["error"] for e in engine.state["recent_errors"])


# --------------------------------------------------- mark to market and P&L


def test_mark_to_market_tracks_unrealised_pnl_and_drawdown(config_dir):
    """Unrealised P&L follows the latest close, and drawdown follows equity.

    Marking open positions at their own entry price - which this engine once did
    - makes unrealised P&L identically zero and drawdown blind to every open
    loss, which is precisely the drawdown a risk limit exists to catch.
    """
    engine, data = _long_setup(
        config_dir,
        stop=90.0,
        target=200.0,
        bars=[
            [100.0, 101.0, 99.0, 100.0, 5_000.0],
            [100.0, 101.0, 99.5, 100.0, 5_000.0],
            [100.0, 101.0, 95.5, 96.0, 5_000.0],
            [96.0, 110.0, 95.5, 109.0, 5_000.0],
        ],
    )
    engine.process_bar(data, timestamp=data.df.index[1], data_quality="PASS")
    entry = engine.portfolio.positions[0].entry_price
    opening_equity = engine.portfolio_snapshot()["equity"]

    engine.process_bar(data, timestamp=data.df.index[2], data_quality="PASS")
    down = engine.portfolio_snapshot()
    assert down["mark_price"] == pytest.approx(96.0)
    assert down["unrealised_pnl"] == pytest.approx((96.0 - entry) * 2.0)
    assert down["unrealised_pnl"] < 0
    assert down["equity"] < opening_equity
    assert down["drawdown"] > 0, "an open loss produced no drawdown"
    assert down["realised_pnl"] == 0.0, "an open position was realised"
    assert down["marked_to_market"] is True

    engine.process_bar(data, timestamp=data.df.index[3], data_quality="PASS")
    up = engine.portfolio_snapshot()
    assert up["unrealised_pnl"] == pytest.approx((109.0 - entry) * 2.0)
    assert up["unrealised_pnl"] > 0
    assert up["equity"] > down["equity"]

    # Bar 0 supplies the reference open and is never processed, so three bars
    # were marked: the entry bar, the drawdown bar and the recovery bar.
    curve = engine.equity_curve()
    assert len(curve) == 3
    assert curve[-1]["equity"] == pytest.approx(up["equity"])
    assert curve[1]["drawdown"] > 0, "the open loss left no drawdown on the curve"


def test_realised_and_unrealised_are_kept_apart(config_dir):
    """A closed trade moves realised P&L; an open one only moves unrealised."""
    engine = _engine(config_dir)
    engine.catch_up(_real_shaped_market(end="2017-12-31"), data_quality="PASS")

    snapshot = engine.portfolio_snapshot()
    realised_from_trades = sum(t.net_pnl for t in engine.portfolio.closed_trades)
    assert snapshot["realised_pnl"] == pytest.approx(realised_from_trades)
    assert snapshot["equity"] == pytest.approx(
        snapshot["cash"] + snapshot["unrealised_pnl"]
        if engine.portfolio.positions
        else snapshot["cash"]
    )
    if not engine.portfolio.positions:
        assert snapshot["unrealised_pnl"] == 0.0


def test_unmarked_position_is_labelled_rather_than_reported_as_zero(config_dir):
    """With no mark price, the payload says so instead of showing a zero.

    An unmarked zero and a genuine zero look identical on a page. The flag is
    what lets the dashboard refuse to present the first as the second.
    """
    engine = _engine(config_dir)
    engine.portfolio.positions.append(
        __import__("app.execution.models", fromlist=["Position"]).Position(
            symbol="XAUUSD",
            direction=SignalDirection.LONG,
            quantity=1.0,
            entry_price=1800.0,
            entry_time=pd.Timestamp("2021-01-04", tz="UTC"),
            stop_loss=1750.0,
            strategy="test",
        )
    )
    snapshot = engine.portfolio_snapshot()
    assert snapshot["mark_price"] is None
    assert snapshot["marked_to_market"] is False

    view = build_view(engine.dashboard_data())
    assert view["portfolio"]["marked_to_market"] is False
    unrealised = next(m for m in view["portfolio"]["metrics"] if m["label"] == "Unrealised")
    assert "not a measurement" in unrealised["help"]


# ------------------------------------------------------- duplicate bars


def test_duplicate_bars_are_skipped_not_retraded(config_dir):
    """Replaying a processed bar is a no-op, not a second trade.

    This is what makes a restart that overlaps history safe, and it is also the
    live path's protection: ``get_latest_data`` returns a window that mostly
    repeats what the last tick already handled.
    """
    engine = _engine(config_dir)
    data = _real_shaped_market(end="2017-12-31")

    engine.catch_up(data, data_quality="PASS")
    first = (
        engine.state["bars_processed"],
        len(engine.state["fills"]),
        len(engine.portfolio.closed_trades),
        round(engine.portfolio.cash, 9),
        _trade_signature(engine.portfolio),
    )

    for _ in range(3):
        engine.catch_up(data, data_quality="PASS")
    assert (
        engine.state["bars_processed"],
        len(engine.state["fills"]),
        len(engine.portfolio.closed_trades),
        round(engine.portfolio.cash, 9),
        _trade_signature(engine.portfolio),
    ) == first

    processed = engine.state["processed_bars"]
    assert len(processed) == len(set(processed))


def test_single_duplicate_bar_reports_skipped(config_dir):
    engine = _engine(config_dir)
    data = _market(periods=40)
    engine.process_bar(data, timestamp=data.df.index[10], data_quality="PASS")
    outcome = engine.process_bar(data, timestamp=data.df.index[10], data_quality="PASS")

    assert outcome["status"] == "skipped"
    assert "already been processed" in outcome["reason"]
    assert engine.state["bars_processed"] == 1

    forced = engine.process_bar(
        data, timestamp=data.df.index[10], data_quality="PASS", force=True
    )
    assert forced["status"] == "ok"
    assert engine.state["bars_processed"] == 2


# --------------------------------------------------------- persistence


def test_state_round_trips_through_the_file(config_dir, tmp_path):
    path = tmp_path / "paper_state.json"
    engine = _engine(config_dir, state_path=path)
    engine._state["latest_decision"] = "ORDER_LONG"
    engine._state["market"] = {"symbol": "XAUUSD", "timeframe": "1D", "quality_status": "PASS"}
    engine._state["recent_events"] = [{"event": "bootstrap", "message": "ready"}]
    engine.save_state()

    assert path.exists()
    json.loads(path.read_text())  # must be valid, complete JSON

    reloaded = _engine(config_dir, state_path=path)
    assert reloaded.state["latest_decision"] == "ORDER_LONG"
    assert reloaded.state["market"]["quality_status"] == "PASS"
    assert reloaded.state["recent_events"][0]["event"] == "bootstrap"


@pytest.mark.slow
def test_restart_reproduces_continuous_run_exactly(config_dir, tmp_path):
    """A run split across a restart must match one that never stopped.

    Asserted over every field of every trade, plus cash, the drawdown peak and
    the equity curve. A restart bug shows up as a plausible-looking difference,
    never as an error: the drawdown peak resets and the risk limit silently
    widens, or excursions restart and every surviving trade reports that it
    never went against the entry.
    """
    cfg = _cfg(config_dir)
    data = _real_shaped_market(end="2018-12-31")
    frame = data.df
    cut = len(frame) // 2

    continuous = PaperTradingEngine(config=cfg, symbol="XAUUSD", timeframe="1D")
    continuous.catch_up(data, data_quality="PASS")

    path = tmp_path / "restart.json"
    first = PaperTradingEngine(config=cfg, symbol="XAUUSD", timeframe="1D", state_path=path)
    first.catch_up(data.replace(df=frame.iloc[:cut]), data_quality="PASS")
    baselines_before = first.portfolio.risk_baselines()
    del first

    second = PaperTradingEngine(config=cfg, symbol="XAUUSD", timeframe="1D", state_path=path)
    assert second.portfolio.risk_baselines()["peak_equity"] == pytest.approx(
        baselines_before["peak_equity"]
    ), "the drawdown peak did not survive the restart, which widens the risk limit"
    second.catch_up(data, data_quality="PASS")

    assert _trade_signature(second.portfolio) == _trade_signature(continuous.portfolio)
    assert second.portfolio.cash == pytest.approx(continuous.portfolio.cash, abs=1e-9)
    assert second.portfolio.realised_pnl == pytest.approx(
        continuous.portfolio.realised_pnl, abs=1e-9
    )
    assert second.portfolio.total_costs == pytest.approx(
        continuous.portfolio.total_costs, abs=1e-9
    )
    assert len(second.portfolio.snapshots) == len(continuous.portfolio.snapshots)
    assert second.portfolio.risk_baselines()["peak_equity"] == pytest.approx(
        continuous.portfolio.risk_baselines()["peak_equity"], abs=1e-9
    )
    assert second.performance.records == continuous.performance.records, (
        "the selector's measured performance did not survive the restart, so strategy "
        "weights would depend on when the process was last restarted"
    )


def test_open_position_survives_restart_with_its_excursions(config_dir, tmp_path):
    """A position reloaded from disk keeps its MAE/MFE history.

    MAE and MFE are the only trade statistics not recomputable from a position's
    fields: they depend on prices the position has already lived through.
    """
    engine, data = _long_setup(
        config_dir,
        stop=90.0,
        target=200.0,
        bars=[
            [100.0, 101.0, 99.0, 100.0, 5_000.0],
            [100.0, 101.0, 99.5, 100.0, 5_000.0],
            [100.0, 101.0, 93.0, 99.0, 5_000.0],
            [99.0, 100.0, 85.0, 88.0, 5_000.0],
        ],
    )
    path = tmp_path / "excursions.json"
    engine.state_path = path
    engine.process_bar(data, timestamp=data.df.index[1], data_quality="PASS")
    engine.process_bar(data, timestamp=data.df.index[2], data_quality="PASS")

    position = engine.portfolio.positions[0]
    low, high = engine.portfolio.excursion_bounds(position)
    assert low == pytest.approx(93.0)

    reloaded = _engine(config_dir, state_path=path)
    restored = reloaded.portfolio.positions[0]
    assert reloaded.portfolio.excursion_bounds(restored) == pytest.approx((low, high))
    assert restored.metadata["entry_bar"] == position.metadata["entry_bar"]

    reloaded.process_bar(data, timestamp=data.df.index[3], data_quality="PASS")
    trade = reloaded.portfolio.closed_trades[0]
    assert trade.mae > 0, "the restored position reports never having gone against the entry"
    assert trade.mae == pytest.approx(restored.entry_price - 93.0)


def test_corrupt_state_file_is_quarantined_and_the_engine_starts_flat(config_dir, tmp_path):
    """A half-read state file must not become a half-applied position book."""
    path = tmp_path / "corrupt.json"
    path.write_text('{"cash": 9000.0, "positions": [', encoding="utf-8")

    engine = _engine(config_dir, state_path=path)
    assert engine.portfolio.cash == engine.config.platform.starting_balance
    assert not engine.portfolio.positions
    assert path.with_suffix(path.suffix + ".corrupt").exists()
    assert any("could not be read" in e["error"] for e in engine.state["recent_errors"])
    assert engine.system_health()["status"] != "HEALTHY"


def test_kill_switch_is_read_from_config_not_from_saved_state(config_dir, tmp_path):
    """A saved ``trading_enabled: true`` must not re-enable a switched-off engine."""
    path = tmp_path / "killswitch.json"
    enabled = _engine(config_dir, state_path=path)
    assert enabled.trading_enabled is True
    enabled.save_state()

    cfg = get_config(config_dir)
    disabled = PaperTradingEngine(
        config=override_config(
            cfg, platform=cfg.platform.model_copy(update={"trading_enabled": False})
        ),
        symbol="XAUUSD",
        timeframe="1D",
        state_path=path,
    )
    assert disabled.trading_enabled is False
    assert disabled.state["trading_enabled"] is False
    assert disabled.state["live_trading_enabled"] is False


def test_persisted_state_never_claims_live_trading(config_dir, tmp_path):
    path = tmp_path / "safety.json"
    engine = _engine(config_dir, state_path=path)
    engine.catch_up(_market(periods=40), data_quality="PASS")

    payload = json.loads(path.read_text())
    assert payload["live_trading_enabled"] is False
    assert payload["mode"] == "PAPER"
    assert engine.dashboard_data()["safety"]["live_trading_enabled"] is False
    assert str(engine.config.platform.mode) != "LIVE"


# ------------------------------------------------- data quality enforcement


@pytest.mark.parametrize(
    ("grade", "block_warning", "expected"),
    [
        (None, False, RiskBlockReason.DATA_QUALITY_UNKNOWN),
        ("UNKNOWN", False, RiskBlockReason.DATA_QUALITY_UNKNOWN),
        ("FAIL", False, RiskBlockReason.DATA_QUALITY),
        ("WARNING", True, RiskBlockReason.DATA_QUALITY),
        ("WARNING", False, None),
        ("PASS", False, None),
    ],
)
def test_risk_enforces_the_quality_grade(grade, block_warning, expected):
    """The grade the paper loop passes decides whether risk will size at all.

    An unstated grade is a refusal, not a pass: "nobody checked" must not size
    the same position as "checked and passed".
    """
    engine = RiskEngine(
        RiskConfig(block_on_data_quality_warning=block_warning),
        data_quality=grade,
        trading_enabled=True,
    )
    signal = Signal(
        strategy="test",
        symbol="XAUUSD",
        timeframe="1D",
        direction=SignalDirection.LONG,
        confidence=0.8,
        entry_price=100.0,
        stop_loss=99.0,
        timestamp=None,
    )
    decision = engine.evaluate(signal, Portfolio(10_000.0), equity=10_000.0)
    assert decision.block_reason is expected
    assert decision.approved is (expected is None)


def test_degraded_data_blocks_every_paper_order(config_dir):
    """A FAIL-graded feed cannot reach a position through the paper loop.

    The quality gate refuses to hand over FAIL data on the normal path, but that
    is not the only path: a caller can disable validation or load a frame from
    disk. The check at the point where size is committed is what makes the gate
    a gate rather than a label.
    """
    engine = _engine(config_dir)
    engine.catch_up(_real_shaped_market(end="2017-12-31"), data_quality="FAIL")

    assert engine.state["bars_decided"] > 0, "no bar reached the risk engine"
    assert not engine.portfolio.positions
    assert not engine.portfolio.closed_trades
    assert not engine.state["pending_orders"]
    assert engine.state["latest_state"]["risk"]["block_reason"] == "DATA_QUALITY"
    assert engine.system_health()["status"] == "DEGRADED"


def test_unverified_data_blocks_every_paper_order(config_dir):
    """No grade at all is treated as a refusal, not as a pass."""
    engine = _engine(config_dir)
    engine.catch_up(_real_shaped_market(end="2017-12-31"), data_quality=None)

    assert engine.state["bars_decided"] > 0
    assert not engine.portfolio.closed_trades
    assert engine.state["latest_state"]["risk"]["block_reason"] == "DATA_QUALITY_UNKNOWN"


@pytest.mark.slow
def test_warning_grade_is_visible_and_configurable(config_dir):
    """WARNING data trades or not per config, and either way says which."""
    cfg = get_config(config_dir)
    blocking = override_config(
        cfg,
        platform=cfg.platform.model_copy(update={"trading_enabled": True}),
        risk=cfg.risk.model_copy(update={"block_on_data_quality_warning": True}),
    )
    engine = PaperTradingEngine(config=blocking, symbol="XAUUSD", timeframe="1D")
    engine.catch_up(_real_shaped_market(end="2016-12-31"), data_quality="WARNING")
    assert not engine.portfolio.closed_trades
    assert engine.state["latest_state"]["risk"]["block_reason"] == "DATA_QUALITY"
    reasons = " ".join(engine.system_health()["reasons"])
    assert "WARNING" in reasons and "no new position is opened" in reasons

    permissive = _cfg(config_dir, risk=cfg.risk.model_copy(
        update={"block_on_data_quality_warning": False}
    ))
    engine2 = PaperTradingEngine(config=permissive, symbol="XAUUSD", timeframe="1D")
    engine2.catch_up(_real_shaped_market(end="2016-12-31"), data_quality="WARNING")
    assert engine2.portfolio.closed_trades, "the permissive setting blocked anyway"
    reasons2 = " ".join(engine2.system_health()["reasons"])
    assert "trading continues on caveated data" in reasons2


def test_ohlc_bound_defects_reach_the_dashboard_with_their_share(config_dir):
    """The gate's repaired-bar finding must be visible, with its ratio.

    This is the FX case made mechanical. Yahoo's OTC FX composites carry OHLC
    bound violations in a measurable share of bars; the gate repairs and grades
    them, and a dashboard that showed only the word WARNING would leave a reader
    unable to tell a 3%-of-bars defect from one stale bar.
    """
    frame = make_ohlcv(periods=200, start="2021-01-01", seed=11)
    broken = frame.copy()
    # A vendor-style defect: the high printed below the bar's own close.
    defect_rows = broken.index[::10]
    broken.loc[defect_rows, "high"] = broken.loc[defect_rows, "close"] * 0.98

    from app.data.processors.normalize import clean_ohlcv

    cleaned, cleaning = clean_ohlcv(broken)
    assert cleaning.ohlc_bounds_repaired > 0, "the fixture no longer contains a defect"

    data = MarketData(symbol="EURUSD", timeframe="1D", df=cleaned, provider="csv")
    cfg = _cfg(config_dir)
    report = DataQualityEngine(cfg.data.quality).validate(
        data, cfg.assets.get("EURUSD"), cleaning=cleaning
    )
    assert report.status.name in {"WARNING", "FAIL"}, (
        f"a feed with {cleaning.ohlc_bounds_repaired} repaired bars graded {report.status}"
    )

    engine = PaperTradingEngine(config=cfg, symbol="EURUSD", timeframe="1D")
    engine.catch_up(data, data_quality=report)

    market = engine.market_summary()
    assert market["quality_status"] == str(report.status)
    assert market["quality"]["stats"]["bars_repaired"] == cleaning.ohlc_bounds_repaired

    findings = quality_rows(engine.dashboard_data())
    repaired = [f for f in findings if f["code"] == "repaired_bars"]
    assert repaired, f"repaired_bars is not on the dashboard; findings: {findings}"
    assert repaired[0]["bars"] == cleaning.ohlc_bounds_repaired
    assert repaired[0]["share"] > 0
    assert "%" in repaired[0]["share_display"]
    assert repaired[0]["severity"] in {"WARNING", "FAIL"}
    assert repaired[0]["tone"] in {"warn", "error"}

    panel = build_view(engine.dashboard_data())["market"]
    repaired_metric = next(m for m in panel["metrics"] if m["label"] == "Bars repaired")
    assert repaired_metric["value"] > 0
    assert repaired_metric["tone"] == "warn"


def test_fx_volume_is_suppressed_and_labelled(config_dir):
    """An OTC FX feed's "volume" is not a traded quantity, and the page says so."""
    cfg = _cfg(config_dir)
    assert cfg.assets.get("EURUSD").has_reliable_volume is False

    engine = PaperTradingEngine(config=cfg, symbol="EURUSD", timeframe="1D")
    engine.catch_up(_market(symbol="EURUSD", periods=40), data_quality="PASS")

    view = build_view(engine.dashboard_data())
    volume = next(m for m in view["market"]["metrics"] if m["label"] == "Volume usable")
    assert volume["value"] is False
    assert "suppressed" in volume["display"]


# --------------------------------------------------- proxy instrument labelling


def test_xauusd_is_labelled_a_futures_proxy_everywhere(config_dir):
    """GC=F must never be presented as true spot XAUUSD.

    Asserted on the engine payload and on the dashboard view, because a caveat
    that lives only in the page is a caveat that the next surface will omit.
    """
    cfg = _cfg(config_dir)
    asset = cfg.assets.get("XAUUSD")
    assert asset.is_proxy is True
    assert "GC=F" in (asset.data_caveat or "")
    assert asset.provider_symbols["yahoo"] == "GC=F"

    engine = PaperTradingEngine(config=cfg, symbol="XAUUSD", timeframe="1D")
    engine.catch_up(_market(periods=40), data_quality="PASS")

    data = engine.dashboard_data()
    assert data["instrument"]["is_proxy"] is True
    caveat = data["instrument"]["data_caveat"]
    assert "proxy" in caveat.lower()
    assert "futures" in caveat.lower()
    assert "not true spot" in caveat.lower()

    banners = build_view(data)["banners"]
    proxy_banners = [b for b in banners if "proxy" in b["title"].lower()]
    assert proxy_banners, f"no proxy banner; titles were {[b['title'] for b in banners]}"
    assert proxy_banners[0]["tone"] == "warn"
    assert proxy_banners[0] is banners[0], "the proxy caveat is not the first thing shown"

    market_node = next(
        n for n in data["graph"]["nodes"] if n["id"] == "market_data"
    )
    assert "proxy" in market_node["detail"].lower()


def test_non_proxy_instrument_is_not_labelled_a_proxy(config_dir):
    cfg = _cfg(config_dir)
    engine = PaperTradingEngine(config=cfg, symbol="EURUSD", timeframe="1D")
    engine.catch_up(_market(symbol="EURUSD", periods=40), data_quality="PASS")
    data = engine.dashboard_data()
    assert data["instrument"]["is_proxy"] is False
    proxy_banners = [
        b for b in build_view(data)["banners"] if "is a proxy series" in b["title"]
    ]
    assert not proxy_banners


# ------------------------------------------------------------- the live path


class _StubService:
    """A ``MarketDataService`` stand-in, so the live path runs without a network."""

    def __init__(self, data, report, *, raises=None):
        self.data, self.report, self.raises = data, report, raises
        self.calls: list[tuple] = []

    def get_latest_data(self, symbol, timeframe, bars=500, validate=True):
        self.calls.append((symbol, timeframe, bars, validate))
        if self.raises is not None:
            raise self.raises
        from app.data.service import DataRequestResult

        return DataRequestResult(data=self.data, quality=self.report, cleaning=None)


def _report_for(cfg, data):
    return DataQualityEngine(cfg.data.quality).validate(
        data, cfg.assets.assets.get(data.symbol.upper())
    )


@pytest.mark.slow
def test_live_tick_uses_the_service_and_the_same_bar_logic(config_dir, tmp_path):
    """The live path differs from replay only in where the frame comes from.

    Asserted by running both over the same bars and comparing the outcome. If
    the two paths could diverge, a bug could live in the one that is harder to
    test - which is always the live one.
    """
    cfg = _cfg(config_dir)
    data = _real_shaped_market(end="2017-12-31")
    report = _report_for(cfg, data)

    replayed = PaperTradingEngine(config=cfg, symbol="XAUUSD", timeframe="1D")
    replayed.catch_up(data, data_quality=report)

    stub = _StubService(data, report)
    live = PaperTradingEngine(
        config=cfg,
        symbol="XAUUSD",
        timeframe="1D",
        state_path=tmp_path / "live.json",
        market_data_service=stub,
    )
    outcome = live.run_live_tick(bars=len(data.df))

    assert outcome["status"] == "ok"
    assert outcome["bars_processed"] == len(data.df)
    assert stub.calls == [("XAUUSD", "1D", len(data.df), True)]
    assert live.market_summary()["source"] == "live_refresh"
    assert live.market_summary()["quality_status"] == str(report.status)
    assert _trade_signature(live.portfolio) == _trade_signature(replayed.portfolio)
    assert live.portfolio.cash == pytest.approx(replayed.portfolio.cash, abs=1e-9)

    # A second tick over the same bars must find nothing new.
    again = live.run_live_tick(bars=len(data.df))
    assert again["status"] == "no_new_bars"
    assert again["bars_processed"] == 0
    assert _trade_signature(live.portfolio) == _trade_signature(replayed.portfolio)


def test_failed_refresh_changes_no_portfolio_state(config_dir, tmp_path):
    """An unreachable provider is reported, never papered over."""
    cfg = _cfg(config_dir)
    data = _real_shaped_market(end="2016-12-31")
    report = _report_for(cfg, data)
    stub = _StubService(data, report)
    engine = PaperTradingEngine(
        config=cfg,
        symbol="XAUUSD",
        timeframe="1D",
        state_path=tmp_path / "fail.json",
        market_data_service=stub,
    )
    engine.run_live_tick(bars=len(data.df))
    before = (round(engine.portfolio.cash, 9), _trade_signature(engine.portfolio))
    good_bar = engine.market_summary()["last_bar"]

    stub.raises = RuntimeError("provider unreachable")
    outcome = engine.run_live_tick(bars=50)

    assert outcome["status"] == "failed"
    assert (round(engine.portfolio.cash, 9), _trade_signature(engine.portfolio)) == before
    market = engine.market_summary()
    assert market["status"] == "ERROR"
    assert "provider unreachable" in market["error"]
    assert market["last_bar"] == good_bar, "stale data was relabelled as current"

    health = engine.system_health()
    assert health["status"] == "DEGRADED"
    assert any("refresh failed" in r for r in health["reasons"])

    node = next(n for n in engine.dashboard_data()["graph"]["nodes"] if n["id"] == "market_data")
    assert node["status"] == "ERROR"
    assert node["reached"] is False
    assert engine.dashboard_data()["graph"]["path"] == []


def test_empty_provider_response_is_reported(config_dir):
    cfg = _cfg(config_dir)
    from app.data.schema import empty_frame

    empty = MarketData(
        symbol="XAUUSD", timeframe="1D", df=empty_frame(), provider="synthetic"
    )
    stub = _StubService(empty, _report_for(cfg, _real_shaped_market(end="2016-12-31")))
    engine = PaperTradingEngine(
        config=cfg, symbol="XAUUSD", timeframe="1D", market_data_service=stub
    )
    outcome = engine.run_live_tick()

    assert outcome["status"] == "failed"
    assert engine.market_summary()["status"] == "EMPTY"
    assert engine.system_health()["status"] == "DEGRADED"


# ------------------------------------------------- no fabrication guarantees


def test_no_fabrication_when_nothing_has_run(config_dir):
    """An engine that has never processed a bar must say exactly that.

    The failure this guards against is specific and was real: health was
    reported HEALTHY whenever the error list happened to be empty, so an engine
    with no data and no history described itself as healthy while every stage of
    the graph showed green.
    """
    engine = _engine(config_dir)
    data = engine.dashboard_data()

    assert data["market"]["status"] == "NO_DATA"
    assert data["market"]["quality_status"] == "UNKNOWN"
    assert data["market"]["last_close"] is None
    assert data["market"]["freshness"]["status"] == "UNKNOWN"
    assert data["regime"] is None
    assert data["strategies"] == []
    assert data["risk"] is None
    assert data["order"] is None
    assert data["final_decision"] is None
    assert data["trades"] == []
    assert data["equity_curve"] == []

    health = data["system_health"]
    assert health["status"] == "IDLE"
    assert health["bars_processed"] == 0
    assert any("No bar has been processed" in r for r in health["reasons"])

    graph = data["graph"]
    assert graph["active_nodes"] == []
    assert graph["path"] == []
    assert graph["stopped_at"] == "market_data"
    statuses = {n["id"]: n["status"] for n in graph["nodes"]}
    assert statuses["market_data"] == "PENDING"
    assert not any(s in {"ACTIVE", "OK"} for s in statuses.values()), (
        f"a stage reported as run with no data: {statuses}"
    )

    performance = data["performance"]
    assert performance["overall"]["trades"] == 0
    assert performance["overall"]["expectancy_r"] is None
    assert performance["by_strategy"] == []
    assert "nothing measured" in performance["overall"]["verdict"]

    view = build_view(data)
    assert view["header"]["status"] == "IDLE"
    assert view["regime"]["detected"] is False
    assert view["performance"]["has_trades"] is False
    assert any("Nothing has run yet" in b["title"] for b in view["banners"])


def test_health_is_never_healthy_while_an_error_stands(config_dir):
    engine = _engine(config_dir)
    engine.catch_up(_market(periods=40), data_quality="PASS")
    engine._record_error("a node failed", node="graph")
    health = engine.system_health()

    assert health["status"] != "HEALTHY"
    assert health["error_count"] == 1
    assert any("a node failed" in r for r in health["reasons"])


def test_health_reports_the_worst_finding_not_the_last(config_dir):
    """Severity is the maximum over findings, not whichever check ran last."""
    engine = _engine(config_dir)
    engine.catch_up(_market(periods=40), data_quality="PASS")
    engine._state["market"]["status"] = "ERROR"
    engine._state["market"]["error"] = "boom"

    health = engine.system_health()
    assert health["status"] == "DEGRADED"
    # The specific failure explains the state; "nothing has run" would not.
    assert not any("Run a catch-up" in r for r in health["reasons"])


def test_stale_data_is_not_reported_as_fresh():
    fresh = assess_freshness("2024-01-02T00:00:00+00:00", "1D", now=pd.Timestamp("2024-01-02T12:00:00Z"))
    stale = assess_freshness("2024-01-02T00:00:00+00:00", "1D", now=pd.Timestamp("2024-03-01T00:00:00Z"))
    unknown = assess_freshness(None, "1D")
    future = assess_freshness(
        "2024-06-01T00:00:00+00:00", "1D", now=pd.Timestamp("2024-01-02T00:00:00Z")
    )

    assert fresh.status == "FRESH" and fresh.is_usable
    assert stale.status == "STALE" and not stale.is_usable
    assert unknown.status == "UNKNOWN" and not unknown.is_usable
    assert "not treated as fresh" in unknown.detail
    assert future.status == "UNKNOWN", "a future-stamped bar was graded as an age"


# ---------------------------------------------------- dashboard correctness


def test_dashboard_numbers_are_the_engine_numbers(config_dir):
    """Every figure on the page traces to something the engine computed."""
    engine = _engine(config_dir)
    engine.catch_up(_real_shaped_market(end="2017-12-31"), data_quality="PASS")

    data = engine.dashboard_data()
    view = build_view(data)
    portfolio = engine.portfolio_snapshot()

    def metric(panel, label):
        return next(m for m in view[panel]["metrics"] if m["label"] == label)

    assert metric("portfolio", "Equity")["value"] == pytest.approx(portfolio["equity"])
    assert metric("portfolio", "Cash")["value"] == pytest.approx(portfolio["cash"])
    assert metric("portfolio", "Realised")["value"] == pytest.approx(portfolio["realised_pnl"])
    assert metric("portfolio", "Unrealised")["value"] == pytest.approx(
        portfolio["unrealised_pnl"]
    )
    assert metric("portfolio", "Drawdown")["value"] == pytest.approx(portfolio["drawdown"])
    assert metric("portfolio", "Closed trades")["value"] == len(engine.portfolio.closed_trades)
    assert metric("header", "Bars processed")["value"] == engine.state["bars_processed"]
    assert metric("market", "Last close")["value"] == pytest.approx(engine.state["last_close"])

    assert len(view["portfolio"]["equity_series"]) == len(engine.portfolio.snapshots)
    assert view["portfolio"]["equity_series"][-1]["equity"] == pytest.approx(
        engine.portfolio.snapshots[-1].equity
    )
    assert view["header"]["symbol"] == "XAUUSD"
    assert view["header"]["timeframe"] == "1D"


def test_dashboard_shows_every_required_section(config_dir):
    """The page must answer each question the brief lists, not a subset."""
    engine = _engine(config_dir)
    engine.catch_up(_real_shaped_market(end="2017-12-31"), data_quality="PASS")
    view = build_view(engine.dashboard_data())

    for section in (
        "banners", "header", "market", "regime", "strategies", "decision",
        "portfolio", "execution", "performance", "quality_findings", "trades",
        "graph", "graph_figure", "events", "errors", "safety",
    ):
        assert section in view, f"the view has no {section} section"

    assert view["market"]["metrics"], "no market metrics"
    assert view["regime"]["detected"] is True
    assert view["strategies"], "no strategy rows"
    assert view["decision"]["steps"], "no decision steps"
    assert view["portfolio"]["metrics"], "no portfolio metrics"
    assert view["execution"]["cost_note"], "no execution cost note"
    assert view["performance"]["metrics"], "no performance metrics"
    assert view["graph_figure"]["nodes"], "no graph nodes"
    assert view["events"], "no events"


def test_strategy_rows_distinguish_suppressed_from_declined(config_dir):
    """A suppressed strategy did not contribute; a declined one chose not to.

    Rendering them identically hides the selector's effect entirely: a strategy
    that signalled LONG and was weighted to zero would read as though it had
    voted.
    """
    engine = _engine(config_dir)
    engine.catch_up(_real_shaped_market(end="2017-12-31"), data_quality="PASS")
    rows = build_view(engine.dashboard_data())["strategies"]

    assert rows, "no strategy rows"
    assert {r["state"] for r in rows} <= {
        "CONTRIBUTING", "SUPPRESSED", "DECLINED", "NO_SIGNAL"
    }
    for row in rows:
        if row["state"] == "SUPPRESSED":
            assert row["weight"] == 0.0
        if row["state"] == "CONTRIBUTING":
            assert row["direction"] in {"LONG", "SHORT"}
            assert row["weight"] is None or row["weight"] > 0
        assert row["reasoning"], f"{row['strategy']} gave no reasoning"

    # Contributing rows sort first: the ones that mattered are read first.
    states = [r["state"] for r in rows]
    assert states == sorted(states, key=lambda s: s != "CONTRIBUTING")


def test_decision_panel_reports_the_risk_block_reason(config_dir):
    engine = _engine(config_dir)
    engine.catch_up(_real_shaped_market(end="2017-12-31"), data_quality="FAIL")
    decision = build_view(engine.dashboard_data())["decision"]

    assert "DATA_QUALITY" in [str(b) for b in decision["risk_blocks"]]
    risk_step = next(s for s in decision["steps"] if s["step"] == "Risk")
    assert risk_step["tone"] == "error"
    assert "Blocked" in risk_step["detail"]


def test_quality_findings_are_structured_rows_not_stringified_dicts(config_dir):
    """The findings table must carry severity, count and share as fields.

    They were previously rendered by interpolating the raw dict into a string,
    which put ``{'code': ..., 'severity': ...}`` on the page - unreadable, and
    unsortable by severity.
    """
    cfg = _cfg(config_dir)
    data = _market(periods=200)
    report = _report_for(cfg, data)
    engine = PaperTradingEngine(config=cfg, symbol="XAUUSD", timeframe="1D")
    engine.catch_up(data, data_quality=report)

    findings = build_view(engine.dashboard_data())["quality_findings"]
    assert findings, "the gate reported findings but none reached the view"
    for finding in findings:
        assert set(finding) >= {
            "code", "severity", "message", "bars", "share", "share_display", "tone"
        }
        assert not finding["message"].startswith("{")
        assert finding["severity"] in {"PASS", "WARNING", "FAIL", "UNKNOWN"}
    severities = [f["severity"] for f in findings]
    order = {"FAIL": 0, "WARNING": 1, "PASS": 2, "UNKNOWN": 3}
    assert severities == sorted(severities, key=lambda s: order[s]), "not worst-first"


# ------------------------------------------------- performance correctness


def test_performance_is_computed_from_closed_trades_only(config_dir):
    """An open position contributes nothing to a performance figure.

    Otherwise the table improves whenever a losing trade is left running, which
    is the oldest way to make a system look better than it is.
    """
    engine, data = _long_setup(
        config_dir,
        stop=90.0,
        target=200.0,
        bars=[
            [100.0, 101.0, 99.0, 100.0, 5_000.0],
            [100.0, 101.0, 99.5, 100.0, 5_000.0],
            [100.0, 101.0, 95.0, 96.0, 5_000.0],
        ],
    )
    engine.process_bar(data, timestamp=data.df.index[1], data_quality="PASS")
    engine.process_bar(data, timestamp=data.df.index[2], data_quality="PASS")

    assert engine.portfolio.positions, "the fixture closed the position"
    assert engine.portfolio_snapshot()["unrealised_pnl"] < 0

    performance = engine.performance_view()
    assert performance["overall"]["trades"] == 0
    assert performance["overall"]["net_pnl"] == 0.0
    assert performance["overall"]["expectancy_r"] is None
    assert performance["open_positions"] == 1
    assert "Closed trades only" in performance["basis"]


def test_performance_splits_by_strategy_and_regime(config_dir):
    engine = _engine(config_dir)
    engine.catch_up(_real_shaped_market(end="2018-12-31"), data_quality="PASS")

    performance = engine.performance_view()
    trades = engine.state["closed_trades"]
    assert trades, "no trade closed, so nothing was sliced"

    assert sum(r["trades"] for r in performance["by_strategy"]) == len(trades)
    assert sum(r["trades"] for r in performance["by_regime"]) == len(trades)
    assert sum(r["trades"] for r in performance["by_strategy_regime"]) == len(trades)
    assert sum(r["trades"] for r in performance["by_exit_reason"]) == len(trades)
    assert {r["regime"] for r in performance["by_regime"]} == {
        t["entry_regime"] for t in trades
    }

    assert performance["overall"]["net_pnl"] == pytest.approx(
        sum(t["net_pnl"] for t in trades)
    )
    assert performance["overall"]["net_pnl"] == pytest.approx(
        engine.portfolio.realised_pnl
    )


def test_performance_attributes_ensemble_trades_to_contributors(config_dir):
    """The aggregator signs its output "ensemble"; the table must go behind that.

    Without contributor attribution, every trade lands in one ``ensemble`` row
    and a per-strategy performance table says nothing at all.
    """
    engine = _engine(config_dir)
    engine.catch_up(_real_shaped_market(end="2018-12-31"), data_quality="PASS")
    performance = engine.performance_view()

    contributors = {r["strategy"] for r in performance["by_contributor"]}
    assert contributors, "no contributor rows"
    assert contributors != {"ensemble"}, (
        "trades were not attributed past the ensemble name"
    )
    assert contributors <= {s.name for s in engine.graph.strategies}
    assert "do not sum" in performance["contributor_note"]

    for row in performance["by_contributor"]:
        assert 0 < row["trades"] <= performance["overall"]["trades"]


def test_thin_samples_are_flagged_not_presented_as_evidence():
    """A mean R from three trades must not read like one from ninety."""
    thin = [
        {"net_pnl": 10.0, "r_multiple": 1.0, "costs": 1.0, "bars_held": 3},
        {"net_pnl": -5.0, "r_multiple": -0.5, "costs": 1.0, "bars_held": 2},
    ]
    stats = summarise_trades(thin, min_samples=10)
    assert stats["insufficient_evidence"] is True
    assert stats["trades"] == 2
    assert stats["expectancy_r"] == pytest.approx(0.25)
    assert stats["win_rate"] == pytest.approx(0.5)
    assert stats["profit_factor"] == pytest.approx(2.0)

    thick = thin * 6
    assert summarise_trades(thick, min_samples=10)["insufficient_evidence"] is False

    built = build_performance(thin, min_samples=10)
    assert "not evidence" in built["overall"]["verdict"]


def test_profit_factor_with_no_losses_is_not_a_number():
    """"Infinite profit factor" is a short sample, not a result."""
    stats = summarise_trades(
        [{"net_pnl": 10.0, "r_multiple": 1.0}, {"net_pnl": 4.0, "r_multiple": 0.4}]
    )
    assert stats["profit_factor"] is None
    assert stats["losses"] == 0


def test_selector_table_says_which_rows_move_a_weight(config_dir):
    engine = _engine(config_dir)
    engine.catch_up(_real_shaped_market(end="2018-12-31"), data_quality="PASS")
    rows = engine.performance_view()["selector_table"]

    assert rows, "the selector recorded nothing"
    threshold = engine.performance.min_samples
    for row in rows:
        assert row["influences_weight"] is (row["samples"] >= threshold)
        expected = engine.performance.expectancy(row["strategy"], row["regime"])
        if expected is None:
            assert row["influences_weight"] is False
        else:
            assert row["mean_r"] == pytest.approx(expected)


@pytest.mark.slow
def test_performance_survives_a_restart(config_dir, tmp_path):
    path = tmp_path / "perf.json"
    engine = _engine(config_dir, state_path=path)
    engine.catch_up(_real_shaped_market(end="2018-12-31"), data_quality="PASS")
    before = engine.performance_view()
    assert before["overall"]["trades"] > 0

    reloaded = _engine(config_dir, state_path=path)
    after = reloaded.performance_view()
    assert after["overall"] == before["overall"]
    assert after["by_contributor"] == before["by_contributor"]
    assert after["selector_table"] == before["selector_table"]


# ---------------------------------------------- graph visualisation


def test_graph_reports_the_full_pipeline_in_order(config_dir):
    engine = _engine(config_dir)
    engine.catch_up(_real_shaped_market(end="2017-12-31"), data_quality="PASS")
    graph = engine.dashboard_data()["graph"]

    stages = [n["stage"] for n in graph["nodes"]]
    for stage in (
        "market_data", "features", "regime", "strategies", "selection",
        "aggregation", "risk", "execution", "paper_position",
    ):
        assert stage in stages, f"the graph has no {stage} stage"

    # Columns must increase along the pipeline, so the picture reads left to right.
    columns = {n["stage"]: n["column"] for n in graph["nodes"]}
    ordered = [
        columns["market_data"], columns["features"], columns["regime"],
        columns["strategies"], columns["selection"], columns["aggregation"],
        columns["risk"], columns["execution"], columns["paper_position"],
    ]
    assert ordered == sorted(ordered)
    assert len(set(ordered)) == len(ordered), "two stages share a column"

    ids = {n["id"] for n in graph["nodes"]}
    for source, target in graph["edges"]:
        assert source in ids and target in ids, f"edge {source}->{target} dangles"

    strategy_nodes = [n for n in graph["nodes"] if n["stage"] == "strategies"]
    assert len(strategy_nodes) == len(engine.graph.strategies)
    assert {n["colour"] for n in graph["nodes"]} <= set(graph["legend"].values())


def test_graph_marks_suppressed_rejected_and_blocked(config_dir):
    """The picture must distinguish the four ways a bar can fail to trade."""
    weight = type("W", (), {"weight": 0.0, "reasoning": ("avoided in this regime",)})
    live = type("W", (), {"weight": 0.9, "reasoning": ()})
    state = {
        "strategy_signals": {
            "momentum": Signal(
                strategy="momentum", symbol="XAUUSD", timeframe="1D",
                direction=SignalDirection.LONG, confidence=0.8,
                entry_price=100.0, stop_loss=99.0,
            ),
            "mean_reversion": Signal(
                strategy="mean_reversion", symbol="XAUUSD", timeframe="1D",
                direction=SignalDirection.SHORT, confidence=0.7,
                entry_price=100.0, stop_loss=101.0,
            ),
            "trend_following": Signal.wait("trend_following", "XAUUSD", "1D"),
        },
        "strategy_weights": {
            "momentum": live(), "mean_reversion": weight(), "trend_following": live()
        },
        "risk_decision": type(
            "R", (), {
                "verdict": "BLOCKED", "approved": False,
                "block_reason": RiskBlockReason.DATA_QUALITY, "quantity": 0.0,
                "risk_amount": 0.0, "reasoning": ("the feed graded FAIL",),
            }
        )(),
        "aggregated": type(
            "A", (), {
                "direction": SignalDirection.LONG, "confidence": 0.7,
                "is_actionable": True, "contributing": ("momentum",), "opposing": (),
            }
        )(),
        "regime": "TRENDING_UP",
        "decision": "BLOCKED_DATA_QUALITY",
    }
    graph = build_graph_visualization(
        state,
        market={"status": "OK", "rows": 500, "quality_status": "FAIL",
                "freshness": {"status": "FRESH"}, "provider": "yahoo"},
        open_positions=0,
        trading_enabled=True,
    )
    statuses = {n["id"]: n["status"] for n in graph["nodes"]}

    assert statuses["strategy_momentum"] == "ACTIVE"
    assert statuses["strategy_mean_reversion"] == "SUPPRESSED"
    assert statuses["strategy_trend_following"] == "REJECTED"
    assert statuses["risk"] == "BLOCKED"
    assert statuses["execution"] == "BLOCKED"

    assert graph["suppressed"] == ["mean_reversion"]
    assert "trend_following" in graph["rejected"]
    assert graph["risk_blocks"] == ["DATA_QUALITY"]
    assert graph["final_decision"] == "BLOCKED_DATA_QUALITY"

    # Risk is reached and refuses, so it is on the path and ends it there.
    assert graph["stopped_at"] == "risk"
    assert graph["path"][-1] == "risk"
    assert "execution" not in graph["path"]
    assert "Refused" in graph["stopped_reason"]


def test_graph_never_shows_an_active_stage_without_data():
    """With no market data, nothing downstream may read as having run."""
    graph = build_graph_visualization(
        {
            "strategy_signals": {
                "momentum": Signal(
                    strategy="momentum", symbol="XAUUSD", timeframe="1D",
                    direction=SignalDirection.LONG, confidence=0.8,
                    entry_price=100.0, stop_loss=99.0,
                )
            },
            "strategy_weights": {"momentum": type("W", (), {"weight": 0.9})()},
        }
    )
    assert "market_data" not in graph["active_nodes"], (
        "market data reported active with no market data supplied"
    )
    market = next(n for n in graph["nodes"] if n["id"] == "market_data")
    assert market["status"] == "PENDING"
    assert market["reached"] is False
    assert graph["path"] == []
    assert graph["stopped_at"] == "market_data"


def test_graph_path_is_not_truncated_by_a_data_caveat():
    """A completed bar on caveated data still shows as having completed.

    The market node keeps its amber status - the caveat is the reason risk will
    refuse - but the path follows what actually ran. Conflating the two once made
    a bar that had passed every stage report that "nothing progressed past
    market data".
    """
    common = {
        "strategy_signals": {
            "momentum": Signal(
                strategy="momentum", symbol="XAUUSD", timeframe="1D",
                direction=SignalDirection.LONG, confidence=0.8,
                entry_price=100.0, stop_loss=99.0,
            )
        },
        "strategy_weights": {"momentum": type("W", (), {"weight": 0.9, "reasoning": ()})()},
        "aggregated": type(
            "A", (), {
                "direction": SignalDirection.LONG, "confidence": 0.8,
                "is_actionable": True, "contributing": ("momentum",), "opposing": (),
            }
        )(),
        "risk_decision": type(
            "R", (), {
                "verdict": "APPROVED", "approved": True, "block_reason": None,
                "quantity": 1.0, "risk_amount": 100.0, "reasoning": (),
            }
        )(),
        "order": object(),
        "regime": "TRENDING_UP",
        "decision": "ORDER_LONG",
        "warm": True,
    }
    stale = build_graph_visualization(
        common,
        market={"status": "OK", "rows": 500, "quality_status": "PASS",
                "freshness": {"status": "STALE"}, "provider": "yahoo"},
        open_positions=1,
        trading_enabled=True,
    )
    market_node = next(n for n in stale["nodes"] if n["id"] == "market_data")
    assert market_node["status"] == "WAIT", "the staleness caveat vanished"
    assert market_node["reached"] is True
    assert "stale" in market_node["detail"].lower()

    assert stale["stopped_at"] is None, "a caveat truncated the path"
    assert stale["path"] == [
        "market_data", "features", "regime", "selection",
        "aggregation", "risk", "execution", "paper_position",
    ]

    summary = build_view({"graph": stale})["graph_figure"]["summary"]
    assert "completed every stage" in summary
    assert "did not start" not in summary


def test_graph_marks_warmup_as_the_stopping_point(config_dir):
    engine = _engine(config_dir)
    engine.catch_up(_market(periods=50), data_quality="PASS")
    graph = engine.dashboard_data()["graph"]

    features = next(n for n in graph["nodes"] if n["id"] == "features")
    assert features["status"] == "WARMUP"
    assert features["reached"] is True
    assert graph["stopped_at"] == "features"
    assert graph["path"] == ["market_data", "features"]
    assert "warm-up" in graph["stopped_reason"]


def test_graph_marks_the_kill_switch_at_execution(config_dir):
    cfg = get_config(config_dir)
    engine = PaperTradingEngine(
        config=override_config(
            cfg, platform=cfg.platform.model_copy(update={"trading_enabled": False})
        ),
        symbol="XAUUSD",
        timeframe="1D",
    )
    engine.catch_up(_real_shaped_market(end="2017-12-31"), data_quality="PASS")

    assert not engine.portfolio.positions
    assert engine.dashboard_data()["safety"]["trading_enabled"] is False
    banners = build_view(engine.dashboard_data())["banners"]
    assert any("Kill switch on" in b["title"] for b in banners)


def test_graph_figure_spec_places_every_node(config_dir):
    """The plot spec is the same data the graph view produced, with coordinates."""
    engine = _engine(config_dir)
    engine.catch_up(_real_shaped_market(end="2017-12-31"), data_quality="PASS")
    view = build_view(engine.dashboard_data())
    graph, spec = view["graph"], view["graph_figure"]

    assert len(spec["nodes"]) == len(graph["nodes"])
    for node, plotted in zip(graph["nodes"], spec["nodes"], strict=True):
        assert plotted["id"] == node["id"]
        assert plotted["x"] == node["column"]
        assert plotted["y"] == -node["row"]
        assert plotted["colour"] == node["colour"]
        assert node["status"] in plotted["hover"]

    xs = [n["x"] for n in spec["nodes"]]
    assert spec["x_range"][0] < min(xs) and spec["x_range"][1] > max(xs)
    assert len(spec["edges"]) == len(graph["edges"])
    assert spec["stage_ticks"], "no stage labels"
    tick_positions = [t["x"] for t in spec["stage_ticks"]]
    assert tick_positions == sorted(tick_positions)


def test_graph_figure_spec_is_safe_on_an_empty_graph():
    spec = build_view({})["graph_figure"]
    assert spec["nodes"] == []
    assert spec["edges"] == []
    assert spec["x_range"] and spec["y_range"]
    assert "no recorded state" in spec["summary"]


# ----------------------------------------------------- execution transparency


def test_execution_panel_keeps_cost_components_separate(config_dir):
    """Spread, slippage and commission stay in their own columns.

    A single "cost" number makes it impossible to answer which execution
    assumption drove a result, which is the only question a simulated fill can
    answer.
    """
    engine = _engine(config_dir)
    engine.catch_up(_real_shaped_market(end="2017-12-31"), data_quality="PASS")
    panel = build_view(engine.dashboard_data())["execution"]

    assert panel["fills"], "no fills to inspect"
    for fill in panel["fills"]:
        assert fill["spread_cost"] is not None
        assert fill["slippage_cost"] is not None
        assert fill["commission"] is not None
        assert fill["total_cost"] == pytest.approx(
            fill["spread_cost"] + fill["slippage_cost"] + fill["commission"]
        )
        assert fill["reference_price"] is not None
        assert fill["difference"] is not None
    assert "next-bar open" in panel["cost_note"]
    assert engine.portfolio.total_costs > 0


def test_atr_slippage_model_receives_the_bar_atr(config_dir):
    """The configured slippage model must actually run.

    Omitting ATR does not disable slippage - it silently substitutes a flat
    0.05%, replacing the configured model with a different one and making the
    paper run's costs differ from the backtest's.
    """
    engine = _engine(config_dir)
    data = _real_shaped_market(end="2017-12-31")
    features = engine.feature_engine.compute(data, engine.asset)
    warm_bar = features.df.index[features.warmup_bars + 5]

    atr = engine._atr_at(features, warm_bar)
    assert atr is not None and atr > 0
    assert atr == pytest.approx(float(features.df["atr"].loc[warm_bar]))

    cold_bar = features.df.index[0]
    assert engine._atr_at(features, cold_bar) is None


# ------------------------------------------------------------------- the CLI


def _run_cli(argv, monkeypatch, config_dir):
    """Invoke the paper runner's ``main`` in-process and capture its exit code.

    The config cache is reset *after* the environment is redirected: it is keyed
    by the requested directory, so a cached ``get_config()`` from earlier in the
    test would otherwise survive the redirect and the runner would read the real
    configs.
    """
    import importlib
    import sys
    from pathlib import Path as _Path

    from app.config.loader import reset_config_cache

    monkeypatch.setenv("GTP_CONFIG_DIR", str(config_dir))
    reset_config_cache()

    scripts = str(_Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("run_paper_trading").main(argv)


def test_cli_status_on_an_unstarted_session_reports_and_fails(config_dir, monkeypatch, capsys):
    code = _run_cli(["--symbol", "XAUUSD", "--timeframe", "1D", "--status"], monkeypatch, config_dir)
    assert code == 1
    out = capsys.readouterr().out
    assert "No saved session" in out
    assert "Start one with --replay or --live" in out


def test_cli_status_reads_a_saved_session_without_advancing_it(
    config_dir, monkeypatch, capsys
):
    engine = _engine(config_dir, state_path=paper_state_path("XAUUSD", "1D"))
    engine.catch_up(_market(periods=60), data_quality="PASS")
    engine.save_state()
    before = paper_state_path("XAUUSD", "1D").read_text()

    code = _run_cli(["--symbol", "XAUUSD", "--timeframe", "1D", "--status"], monkeypatch, config_dir)
    assert code == 0

    out = capsys.readouterr().out
    assert "paper trading status" in out
    assert "PROXY DATA" in out, "the proxy caveat is missing from the status output"
    assert "Live trading: DISABLED" in out
    assert "processed=60" in out
    assert paper_state_path("XAUUSD", "1D").read_text() == before, (
        "--status modified the session it was asked to report on"
    )


def test_cli_rejects_an_unknown_provider(config_dir, monkeypatch, capsys):
    code = _run_cli(
        ["--symbol", "XAUUSD", "--provider", "bloomberg", "--status"], monkeypatch, config_dir
    )
    assert code == 1
    assert "Unknown provider" in capsys.readouterr().out


def test_cli_rejects_an_unknown_symbol(config_dir, monkeypatch, capsys):
    code = _run_cli(["--symbol", "DOGEUSD", "--status"], monkeypatch, config_dir)
    assert code == 1
    assert "Unknown symbol" in capsys.readouterr().out


def test_cli_replays_a_csv_the_operator_supplied(config_dir, monkeypatch, capsys, tmp_path):
    """``--provider csv`` must run a whole session off a local file.

    The free Yahoo feeds are caveated, so "bring your own bars" is the documented
    escape hatch — and the paper runner was the one entry point that could not
    take it.
    """
    from app.utils.paths import raw_dir

    frame = make_ohlcv(periods=320, start="2021-01-01", seed=5)
    raw_dir().mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_dir() / "XAUUSD_1D.csv", index_label="timestamp")

    code = _run_cli(
        ["--symbol", "XAUUSD", "--timeframe", "1D", "--provider", "csv",
         "--replay", "--start", "2021-01-01", "--log-level", "ERROR"],
        monkeypatch,
        config_dir,
    )
    assert code == 0

    out = capsys.readouterr().out
    assert "Replaying" in out
    assert "Processed" in out
    assert "Live trading: DISABLED" in out

    state = paper_state_path("XAUUSD", "1D")
    assert state.exists(), "the CLI run persisted no session"
    payload = json.loads(state.read_text())
    assert payload["bars_processed"] == len(frame)
    assert payload["live_trading_enabled"] is False
    assert payload["market"]["provider"] == "csv"
