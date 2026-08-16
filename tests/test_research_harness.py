"""The segment runner: warm-up handling, scoping and per-window quality.

The property that matters most here is that the warm-up prefix is data the run
is *given* but never *scored on*. If a prefix leaked into the metrics, every
segment result in the platform would be quietly wrong in a direction that
flatters low-volatility periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.loader import get_config, override_config
from app.data.schema import MarketData, coerce_schema
from app.research.harness import HarnessError, ResearchHarness, stitch_equity
from app.research.splits import Segment, SegmentRole, chronological_split


@pytest.fixture
def config(config_dir):
    return get_config(config_dir)


def trending(n: int = 1_200, slope: float = 0.8, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2016-01-01", periods=n, freq="1D", tz="UTC")
    close = 500.0 + slope * np.arange(n) + np.cumsum(rng.normal(0, 2.0, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return coerce_schema(
        pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) + 2.0,
                "low": np.minimum(open_, close) - 2.0,
                "close": close,
                "volume": np.full(n, 5_000.0),
            },
            index=index,
        )
    )


@pytest.fixture
def data() -> MarketData:
    return MarketData(
        symbol="XAUUSD", timeframe="1D", df=trending(), provider="synthetic"
    )


@pytest.fixture
def harness(config, data) -> ResearchHarness:
    return ResearchHarness(config, data, config.assets.get("XAUUSD"))


@pytest.fixture
def segments(harness, data):
    return chronological_split(
        data.df.index, warmup_bars=harness.required_warmup(), embargo_bars=5
    )


# --------------------------------------------------------------------- basics

def test_a_segment_runs_and_is_scored_on_its_own_window(harness, segments):
    run = harness.run(segments[0])

    assert run.segment.name == "in_sample"
    assert not run.equity_curve.empty
    assert run.equity_curve.index[0] >= segments[0].start_time


def test_the_warmup_prefix_is_not_in_the_equity_curve(harness, segments):
    """Flat warm-up bars would deflate volatility and inflate every ratio."""
    segment = segments[1]
    run = harness.run(segment)

    assert run.equity_curve.index.min() >= segment.start_time
    assert len(run.equity_curve) <= segment.bars


def test_no_trade_starts_inside_the_warmup(harness, segments):
    run = harness.run(segments[0])
    if run.trades.empty:
        pytest.skip("no trades in this window")

    entries = pd.to_datetime(run.trades["entry_time"], utc=True)
    assert entries.min() >= segments[0].start_time


def test_a_short_warmup_is_refused_rather_than_run(harness, data):
    """Half-warm indicators are different numbers, not rougher ones."""
    index = data.df.index
    stunted = Segment(
        name="stunted", role=SegmentRole.TEST, start=30, end=300, data_start=0,
        start_time=index[30], end_time=index[299],
    )
    with pytest.raises(HarnessError, match="partially-formed indicators"):
        harness.run(stunted)


def test_each_window_is_graded_on_its_own_data(harness, segments):
    """A grade for a decade says nothing about which year inside it was gappy."""
    run = harness.run(segments[0])
    assert run.data_quality in {"PASS", "WARNING", "FAIL"}
    assert run.result.provenance.data_quality == run.data_quality


def test_positions_do_not_survive_the_end_of_a_window(harness, segments):
    run = harness.run(segments[0])
    assert run.result.snapshots["open_positions"].iloc[-1] == 0


def test_running_the_same_segment_twice_gives_the_same_answer(harness, segments):
    first = harness.run(segments[2])
    second = harness.run(segments[2])

    pd.testing.assert_series_equal(first.equity_curve, second.equity_curve)
    assert first.metrics.total_trades == second.metrics.total_trades


def test_a_variant_configuration_changes_the_result(harness, segments, config):
    """If configuration changes did nothing, every comparison would be vacuous."""
    tighter = override_config(
        config, risk=config.risk.model_copy(update={"risk_per_trade": 0.005})
    )
    baseline = harness.run(segments[0])
    variant = harness.run(segments[0], config=tighter, variant="half_risk")

    assert variant.variant == "half_risk"
    if baseline.metrics.total_trades:
        assert variant.metrics.total_return != baseline.metrics.total_return


def test_the_variant_label_is_carried_through(harness, segments):
    assert harness.run(segments[0], variant="candidate_a").variant == "candidate_a"


def test_summary_rows_carry_the_window_and_its_quality(harness, segments):
    row = harness.run(segments[0]).summary_row()

    assert row["segment"] == "in_sample"
    assert row["role"] == "IN_SAMPLE"
    assert "data_quality" in row


# ------------------------------------------------------------- data quality

def test_a_window_the_risk_engine_refuses_is_explained(config, data):
    """Blocked trading must read as a refusal, not as a quiet strategy."""
    strict = override_config(
        config, risk=config.risk.model_copy(update={"block_on_data_quality_warning": True})
    )
    harness = ResearchHarness(strict, data, strict.assets.get("XAUUSD"))
    segments = chronological_split(
        data.df.index, warmup_bars=harness.required_warmup(), embargo_bars=5
    )
    run = harness.run(segments[0])

    if run.data_quality == "PASS":
        pytest.skip("this window graded clean, so nothing was refused")

    assert run.metrics.total_trades == 0
    assert any("refused by the risk engine" in note for note in run.notes)


# ----------------------------------------------------------------- stitching

def curve(values, start="2020-01-01"):
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="1D", tz="UTC")
    )


def test_stitching_compounds_window_returns():
    """Each window restarts from the same balance, so they cannot be concatenated."""

    class FakeRun:
        def __init__(self, equity, start):
            self.equity_curve = equity
            self.segment = type("S", (), {"start": start})()

    runs = [
        FakeRun(curve([100.0, 110.0]), 0),
        FakeRun(curve([100.0, 120.0], start="2020-02-01"), 10),
    ]
    stitched = stitch_equity(runs, initial_equity=100.0)

    # +10% then +20% compounds to 132, not to a concatenation of the raw curves.
    assert stitched.iloc[-1] == pytest.approx(132.0)


def test_stitching_nothing_returns_an_empty_curve():
    assert stitch_equity([], initial_equity=100.0).empty
