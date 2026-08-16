"""Monte Carlo resampling of the trade sequence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research.monte_carlo import monte_carlo_trade_sequence


def trades(pnl: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "net_pnl": pnl,
            "r_multiple": [p / 100.0 for p in pnl],
            "costs": [1.0] * len(pnl),
        }
    )


def mixed(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return trades(list(rng.normal(5.0, 60.0, n)))


# ------------------------------------------------------------------ mechanics

def test_permutation_preserves_the_total():
    """Reordering trades cannot change what they sum to."""
    frame = mixed()
    report = monte_carlo_trade_sequence(
        frame, initial_equity=10_000.0, method="permutation", simulations=200
    )

    total = frame["net_pnl"].sum() / 10_000.0
    assert report.final_return["mean"] == pytest.approx(total, abs=1e-9)
    assert report.final_return["worst"] == pytest.approx(total, abs=1e-9)


def test_permutation_still_changes_the_drawdown():
    """The whole point: the same trades in a different order hurt differently."""
    report = monte_carlo_trade_sequence(
        mixed(), initial_equity=10_000.0, method="permutation", simulations=300
    )
    assert report.max_drawdown["p95"] > report.max_drawdown["p5"]


def test_bootstrap_spreads_the_return_as_well():
    report = monte_carlo_trade_sequence(
        mixed(), initial_equity=10_000.0, method="bootstrap", simulations=300
    )
    assert report.final_return["p95"] > report.final_return["p5"]


def test_results_are_reproducible_from_the_seed():
    frame = mixed()
    a = monte_carlo_trade_sequence(frame, 10_000.0, simulations=100, seed=7)
    b = monte_carlo_trade_sequence(frame, 10_000.0, simulations=100, seed=7)
    assert a.max_drawdown == b.max_drawdown


def test_a_different_seed_gives_a_different_sample():
    frame = mixed()
    a = monte_carlo_trade_sequence(frame, 10_000.0, simulations=100, seed=7)
    b = monte_carlo_trade_sequence(frame, 10_000.0, simulations=100, seed=8)
    assert a.max_drawdown["p50"] != b.max_drawdown["p50"]


def test_block_resampling_keeps_the_path_length():
    frame = mixed(n=100)
    report = monte_carlo_trade_sequence(
        frame, 10_000.0, method="bootstrap", block_size=10, simulations=50
    )
    assert report.trades_per_path == 100
    assert report.block_size == 10


# ------------------------------------------------------------- interpretation

def test_a_losing_system_shows_a_high_probability_of_loss():
    losing = trades([-50.0] * 40 + [30.0] * 20)
    report = monte_carlo_trade_sequence(losing, 10_000.0, simulations=200)
    assert report.probability_of_loss == pytest.approx(1.0)


def test_drawdown_limit_breach_is_reported():
    frame = trades(list(np.random.default_rng(1).normal(0, 900.0, 80)))
    report = monte_carlo_trade_sequence(
        frame, 10_000.0, simulations=300, drawdown_limit=0.20
    )
    assert 0.0 < report.probability_drawdown_exceeds_limit <= 1.0
    assert report.drawdown_limit == 0.20


def test_ruin_is_detected():
    catastrophic = trades([-3_000.0] * 10)
    report = monte_carlo_trade_sequence(catastrophic, 10_000.0, simulations=50)
    assert report.probability_of_ruin > 0


def test_the_observed_path_is_reported_alongside_the_distribution():
    frame = mixed()
    report = monte_carlo_trade_sequence(frame, 10_000.0, simulations=100)
    assert report.observed_max_drawdown > 0
    assert report.observed_return == pytest.approx(frame["net_pnl"].sum() / 10_000.0)


# ------------------------------------------------------------------- caveats

def test_no_trades_produces_a_report_that_says_so():
    report = monte_carlo_trade_sequence(pd.DataFrame(), 10_000.0)
    assert report.simulations == 0
    assert any("No trades" in w for w in report.warnings)


def test_a_single_trade_is_refused():
    report = monte_carlo_trade_sequence(trades([10.0]), 10_000.0)
    assert report.simulations == 0
    assert any("report its own input" in w for w in report.warnings)


def test_a_thin_sample_is_flagged():
    report = monte_carlo_trade_sequence(trades([1.0, -2.0, 3.0]), 10_000.0, simulations=50)
    assert any("too small" in w for w in report.warnings)


def test_block_size_one_carries_its_own_caveat():
    report = monte_carlo_trade_sequence(mixed(), 10_000.0, simulations=50, block_size=1)
    assert any("independent" in w for w in report.warnings)


def test_the_method_assumption_is_stated():
    permuted = monte_carlo_trade_sequence(mixed(), 10_000.0, simulations=50,
                                          method="permutation")
    bootstrapped = monte_carlo_trade_sequence(mixed(), 10_000.0, simulations=50,
                                              method="bootstrap")
    assert any("sequencing risk" in w for w in permuted.warnings)
    assert any("stable process" in w for w in bootstrapped.warnings)


def test_the_report_renders_and_serialises():
    report = monte_carlo_trade_sequence(mixed(), 10_000.0, simulations=50)
    text = report.render()
    assert "MONTE CARLO" in text
    assert "Max drawdown distribution" in text
    assert report.to_dict()["simulations"] == 50
