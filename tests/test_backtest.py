"""The backtester: bar ordering, causality, reproducibility and metrics.

The causality tests are the most important in the phase. A backtest that can
see the future is not a slightly optimistic backtest - it is a random number
generator that reports whatever you hoped for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import Backtester
from app.backtest.metrics import compute_metrics
from app.config.loader import get_config
from app.data.schema import MarketData, coerce_schema
from app.execution.models import ExitReason
from app.portfolio.portfolio import Portfolio


@pytest.fixture
def config(config_dir):
    return get_config(config_dir)


def market(df: pd.DataFrame, symbol: str = "XAUUSD") -> MarketData:
    return MarketData(symbol=symbol, timeframe="1D", df=df, provider="synthetic")


def trending(n: int = 700, slope: float = 1.2, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-01", periods=n, freq="1D", tz="UTC")
    base = 500.0 + max(0.0, -slope) * n * 1.5
    close = base + slope * np.arange(n) + rng.normal(0, 2.0, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return coerce_schema(
        pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) + 2.0,
                "low": np.minimum(open_, close) - 2.0,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=index,
        )
    )


# -------------------------------------------------------------------- basics

def test_backtest_runs_and_produces_a_result(config):
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(trending()))

    assert result.metrics.initial_equity == config.backtest.initial_balance
    assert not result.equity_curve.empty
    assert result.provenance.bars == 700


def test_backtest_opens_positions_on_a_trending_market(config):
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(trending()))
    assert result.metrics.total_trades > 0


def test_kill_switch_produces_no_trades(config):
    result = Backtester(config, config.assets.get("XAUUSD")).run(
        market(trending()), trading_enabled=False
    )
    assert result.metrics.total_trades == 0
    assert any("No trades were taken" in w for w in result.metrics.warnings)


def test_no_trades_before_the_warmup(config):
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(trending()))
    if result.trades.empty:
        pytest.skip("no trades in this fixture")

    warmup = result.provenance.config_snapshot["warmup_bars_used"]
    first_entry = pd.Timestamp(result.trades["entry_time"].min())
    data_start = result.equity_curve.index[0]
    bars_before = len(result.equity_curve.loc[:first_entry]) - 1
    assert bars_before >= warmup, f"a trade opened at bar {bars_before}, inside the warm-up"
    assert first_entry > data_start


# ------------------------------------------------------------------ causality

def test_backtest_is_causal(config):
    """The central guarantee.

    Running over the first N bars must produce exactly the trades that a run
    over the full series produced within that window. Any difference means the
    full run used information from bars that had not happened yet.
    """
    df = trending(n=700)
    asset = config.assets.get("XAUUSD")

    full = Backtester(config, asset, experiment_id="full").run(market(df))
    cutoff = 500
    truncated = Backtester(config, asset, experiment_id="truncated").run(
        market(df.iloc[:cutoff])
    )

    if truncated.trades.empty:
        pytest.skip("truncated run produced no trades to compare")

    boundary = df.index[cutoff - 1]

    full_trades = full.trades[pd.to_datetime(full.trades["exit_time"]) < boundary]
    partial_trades = truncated.trades[
        pd.to_datetime(truncated.trades["exit_time"]) < boundary
    ]

    assert len(partial_trades) == len(full_trades), (
        "truncating the data changed how many trades occurred before the cutoff, "
        "which means the full run used future information"
    )

    columns = ["entry_time", "entry_price", "exit_price", "direction", "quantity", "net_pnl"]
    pd.testing.assert_frame_equal(
        partial_trades[columns].reset_index(drop=True),
        full_trades[columns].reset_index(drop=True),
        check_dtype=False,
        rtol=1e-9,
    )


def test_equity_curve_is_causal(config):
    """Equity at bar t must not depend on bars after t."""
    df = trending(n=600)
    asset = config.assets.get("XAUUSD")

    full = Backtester(config, asset, experiment_id="a").run(market(df))
    cutoff = 450
    truncated = Backtester(config, asset, experiment_id="b").run(market(df.iloc[:cutoff]))

    overlap = truncated.equity_curve.index[:-1]   # last bar closes open positions
    pd.testing.assert_series_equal(
        truncated.equity_curve.loc[overlap],
        full.equity_curve.loc[overlap],
        rtol=1e-9,
        check_names=False,
    )


def test_entries_never_fill_at_the_signal_bar_close(config):
    """A fill at the signal bar's close is trading on a price that already passed."""
    df = trending(n=600)
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(df))

    if result.trades.empty:
        pytest.skip("no trades in this fixture")

    for _, trade in result.trades.iterrows():
        entry_time = pd.Timestamp(trade["entry_time"])
        position = df.index.get_loc(entry_time)
        # The entry price derives from the entry bar's open, plus costs. It must
        # not equal the previous bar's close, which is what a same-bar fill
        # would produce.
        previous_close = float(df.iloc[position - 1]["close"])
        assert trade["entry_price"] != pytest.approx(previous_close, abs=1e-12)


def test_entry_price_derives_from_the_entry_bar_open(config):
    df = trending(n=600)
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(df))

    if result.trades.empty:
        pytest.skip("no trades in this fixture")

    trade = result.trades.iloc[0]
    entry_time = pd.Timestamp(trade["entry_time"])
    bar_open = float(df.loc[entry_time, "open"])

    # Costs move the fill away from the open, but only slightly.
    assert abs(trade["entry_price"] - bar_open) / bar_open < 0.05


# ------------------------------------------------------------ reproducibility

def test_same_inputs_give_identical_results(config):
    """Every trade must be reproducible."""
    df = trending(n=500)
    asset = config.assets.get("XAUUSD")

    first = Backtester(config, asset, experiment_id="run-1").run(market(df))
    second = Backtester(config, asset, experiment_id="run-2").run(market(df))

    pd.testing.assert_frame_equal(first.trades, second.trades)
    pd.testing.assert_series_equal(first.equity_curve, second.equity_curve)


def test_provenance_records_everything_needed_to_reproduce(config):
    result = Backtester(config, config.assets.get("XAUUSD")).run(
        market(trending()), quality_status="WARNING"
    )
    provenance = result.provenance

    assert provenance.experiment_id
    assert provenance.data_checksum
    assert provenance.data_quality == "WARNING"
    assert provenance.random_seed == config.platform.random_seed
    assert provenance.git_revision
    assert "risk" in provenance.config_snapshot
    assert "execution" in provenance.config_snapshot


def test_data_checksum_changes_with_the_data(config):
    asset = config.assets.get("XAUUSD")
    a = Backtester(config, asset, experiment_id="a").run(market(trending(seed=1)))
    b = Backtester(config, asset, experiment_id="b").run(market(trending(seed=2)))
    assert a.provenance.data_checksum != b.provenance.data_checksum


def test_result_saves_to_disk(config, tmp_path):
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(trending()))
    path = result.save(tmp_path)

    assert (path / "provenance.json").exists()
    assert (path / "metrics.json").exists()
    assert (path / "report.txt").exists()


# --------------------------------------------------------------- trade records

def test_every_trade_is_fully_logged(config):
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(trending()))
    if result.trades.empty:
        pytest.skip("no trades in this fixture")

    required = [
        "entry_time", "entry_price", "exit_time", "exit_price", "direction",
        "quantity", "stop_loss", "take_profit", "strategy", "entry_regime",
        "net_pnl", "gross_pnl", "costs", "r_multiple", "exit_reason", "bars_held",
        "mae", "mfe",
    ]
    missing = [column for column in required if column not in result.trades.columns]
    assert not missing, f"trade log is missing {missing}"


def test_exit_reasons_are_recorded(config):
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(trending()))
    if result.trades.empty:
        pytest.skip("no trades in this fixture")

    reasons = set(result.trades["exit_reason"])
    assert reasons <= {str(r) for r in ExitReason}


def test_no_trade_decisions_are_recorded(config):
    """Section 67: the system must record why it did not trade."""
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(trending()))

    assert result.blocked_counts
    assert not result.decisions.empty
    assert "decision" in result.decisions.columns


def test_open_positions_are_closed_at_the_end(config):
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(trending()))
    if result.trades.empty:
        pytest.skip("no trades in this fixture")

    # Nothing may be left open, or its outcome would be missing from the metrics.
    assert result.snapshots["open_positions"].iloc[-1] == 0


def test_costs_are_charged(config):
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(trending()))
    if result.trades.empty:
        pytest.skip("no trades in this fixture")
    assert result.metrics.total_costs > 0
    assert (result.trades["costs"] > 0).all()


def test_regime_performance_table_is_produced(config):
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(trending()))
    if result.trades.empty:
        pytest.skip("no trades in this fixture")
    assert not result.regime_performance.empty
    assert {"strategy", "regime", "trades", "mean_r"} <= set(result.regime_performance.columns)


# ------------------------------------------------------------------- metrics

def test_metrics_on_a_known_equity_curve():
    equity = pd.Series(
        [100.0, 110.0, 105.0, 120.0],
        index=pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC"),
    )
    trades = pd.DataFrame(
        {
            "net_pnl": [10.0, -5.0, 15.0],
            "r_multiple": [1.0, -0.5, 1.5],
            "bars_held": [1, 1, 1],
        }
    )
    metrics = compute_metrics(equity, trades, initial_equity=100.0)

    assert metrics.total_return == pytest.approx(0.20)
    assert metrics.total_trades == 3
    assert metrics.winning_trades == 2
    assert metrics.win_rate == pytest.approx(2 / 3)
    assert metrics.profit_factor == pytest.approx(25.0 / 5.0)
    assert metrics.expectancy == pytest.approx(20.0 / 3)


def test_max_drawdown_is_measured_from_the_peak():
    equity = pd.Series(
        [100.0, 120.0, 90.0, 110.0],
        index=pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC"),
    )
    metrics = compute_metrics(equity, pd.DataFrame(), initial_equity=100.0)
    assert metrics.max_drawdown == pytest.approx(0.25)   # 120 -> 90


def test_win_rate_alone_does_not_indicate_profit():
    """Nine small wins and one huge loss: 90% win rate, net loss."""
    trades = pd.DataFrame({"net_pnl": [1.0] * 9 + [-50.0], "r_multiple": [0.1] * 9 + [-5.0]})
    equity = pd.Series(
        [100.0, 59.0], index=pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    )
    metrics = compute_metrics(equity, trades, initial_equity=100.0)

    assert metrics.win_rate == pytest.approx(0.9)
    assert metrics.expectancy < 0
    assert metrics.profit_factor < 1.0


def test_empty_equity_curve_is_handled():
    metrics = compute_metrics(pd.Series(dtype="float64"), pd.DataFrame(), initial_equity=100.0)
    assert metrics.total_trades == 0
    assert any("empty" in w.lower() for w in metrics.warnings)


def test_small_sample_is_flagged():
    trades = pd.DataFrame({"net_pnl": [1.0, 2.0], "r_multiple": [1.0, 2.0]})
    equity = pd.Series(
        [100.0, 103.0], index=pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    )
    metrics = compute_metrics(equity, trades, initial_equity=100.0)
    assert any("Only 2 trades" in w for w in metrics.warnings)


def test_flawless_results_are_flagged_as_suspicious():
    """No losing trade is a sign of a bug or a tiny sample, not of skill."""
    trades = pd.DataFrame({"net_pnl": [1.0] * 5, "r_multiple": [1.0] * 5})
    equity = pd.Series(
        [100.0, 105.0], index=pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    )
    metrics = compute_metrics(equity, trades, initial_equity=100.0)

    assert metrics.profit_factor == float("inf")
    assert any("infinite" in w.lower() for w in metrics.warnings)


def test_report_carries_the_caveat(config):
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(trending()))
    text = result.render()
    assert "not a forecast" in text
    assert "weakest form of evidence" in text


# ------------------------------------------------------------------ portfolio

def test_portfolio_rejects_a_non_positive_balance():
    with pytest.raises(ValueError, match="initial_balance must be positive"):
        Portfolio(0.0)


def test_portfolio_tracks_drawdown():
    portfolio = Portfolio(10_000.0)
    timestamps = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")

    portfolio.record_snapshot(timestamps[0], {})
    portfolio.cash = 12_000.0
    portfolio.record_snapshot(timestamps[1], {})
    portfolio.cash = 9_000.0
    snapshot = portfolio.record_snapshot(timestamps[2], {})

    assert snapshot.drawdown == pytest.approx(0.25)   # 12,000 -> 9,000


def test_equity_curve_has_one_point_per_bar(config):
    df = trending(n=400)
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(df))
    assert len(result.equity_curve) == len(df)


def test_max_bars_limits_the_run(config):
    limited = config.__class__(
        **{
            **{f: getattr(config, f) for f in config.__dataclass_fields__},
            "backtest": config.backtest.model_copy(update={"max_bars": 300}),
        }
    )
    result = Backtester(limited, limited.assets.get("XAUUSD")).run(market(trending(n=700)))
    assert len(result.equity_curve) == 300
