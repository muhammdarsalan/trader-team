"""Strategy correlation, redundancy findings and weight variants.

The behavioural requirement under test is as much about restraint as about
arithmetic: finding that two strategies are correlated must never remove one of
them. It produces a hypothesis and a configuration that can test it, and
nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.loader import get_config
from app.research.correlation import (
    analyse_correlations,
    correlation_of_equity_curves,
    signal_agreement_matrix,
    strategy_pnl_series,
    summarise_variant_comparison,
    variant_row,
    weight_variants,
)


@pytest.fixture
def config(config_dir):
    return get_config(config_dir)


def trades_for(strategies: dict[str, list[float]], start: str = "2020-01-01") -> pd.DataFrame:
    rows = []
    for name, pnl in strategies.items():
        times = pd.date_range(start, periods=len(pnl), freq="1D", tz="UTC")
        rows += [
            {"strategy": name, "exit_time": t, "net_pnl": p, "r_multiple": p / 100.0}
            for t, p in zip(times, pnl, strict=True)
        ]
    return pd.DataFrame(rows)


def curve(values: list[float], start: str = "2020-01-01") -> pd.Series:
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="1D", tz="UTC")
    )


# ---------------------------------------------------------------- ingredients

def test_pnl_is_attributed_to_the_exit_bar():
    frame = trades_for({"a": [10.0, -5.0]})
    series = strategy_pnl_series(frame)

    assert list(series.columns) == ["a"]
    assert series["a"].sum() == pytest.approx(5.0)


def test_quiet_bars_count_as_zero_not_missing():
    """Two strategies trading in different months are genuinely uncorrelated."""
    a = trades_for({"a": [10.0] * 5}, start="2020-01-01")
    b = trades_for({"b": [10.0] * 5}, start="2020-06-01")
    series = strategy_pnl_series(pd.concat([a, b]))

    assert not series.isna().any().any()
    assert series["a"].loc["2020-06-01":].sum() == 0.0


def test_signal_agreement_is_measured_over_every_bar():
    decisions = pd.DataFrame(
        {
            "signal_alpha": ["LONG", "WAIT", "WAIT", "LONG"],
            "signal_beta": ["LONG", "WAIT", "SHORT", "LONG"],
        }
    )
    matrix = signal_agreement_matrix(decisions)

    assert matrix.loc["alpha", "beta"] == pytest.approx(0.75)
    assert matrix.loc["alpha", "alpha"] == 1.0


def test_signal_agreement_needs_two_strategies():
    assert signal_agreement_matrix(pd.DataFrame({"signal_only": ["LONG"]})).empty


def test_equity_correlation_uses_returns_not_levels():
    """Two rising curves correlate on levels whether or not they are related."""
    rng = np.random.default_rng(5)
    a = curve(list(100 + np.cumsum(rng.normal(0, 1, 200))))
    b = curve(list(100 + np.cumsum(rng.normal(0, 1, 200))))

    matrix = correlation_of_equity_curves({"a": a, "b": b})
    assert abs(matrix.loc["a", "b"]) < 0.5


def test_identical_curves_correlate_perfectly():
    rng = np.random.default_rng(5)
    values = list(100 + np.cumsum(rng.normal(0, 1, 200)))
    matrix = correlation_of_equity_curves({"a": curve(values), "b": curve(values)})
    assert matrix.loc["a", "b"] == pytest.approx(1.0)


# -------------------------------------------------------------------- finding

def test_correlated_strategies_are_flagged():
    rng = np.random.default_rng(3)
    base = np.cumsum(rng.normal(0, 1, 300))
    twin = base + rng.normal(0, 0.05, 300)

    report = analyse_correlations(
        pd.DataFrame(),
        isolated_equity={"a": curve(list(100 + base)), "b": curve(list(100 + twin))},
        isolated_trades={"a": 40, "b": 35},
        threshold=0.7,
    )

    assert report.findings
    assert report.redundant_pairs == (("a", "b"),)


def test_uncorrelated_strategies_are_not_flagged():
    rng = np.random.default_rng(3)
    a = curve(list(100 + np.cumsum(rng.normal(0, 1, 300))))
    b = curve(list(100 + np.cumsum(rng.normal(0, 1, 300))))

    report = analyse_correlations(
        pd.DataFrame(), isolated_equity={"a": a, "b": b},
        isolated_trades={"a": 40, "b": 35}, threshold=0.7,
    )
    assert not report.findings


def test_a_thin_sample_is_reported_but_not_flagged():
    """A 0.9 correlation over eleven observations is a coincidence with a decimal."""
    values = [100.0, 101.0, 102.5, 101.5, 103.0]
    report = analyse_correlations(
        pd.DataFrame(),
        isolated_equity={"a": curve(values), "b": curve(values)},
        isolated_trades={"a": 2, "b": 2},
        threshold=0.7,
        min_observations=30,
    )

    assert not report.findings
    assert any("too few" in note for note in report.notes)


def test_ensemble_labelled_trades_are_explained_rather_than_silently_empty():
    """Every trade carries the aggregated label, so the log cannot show pairs."""
    report = analyse_correlations(trades_for({"ensemble": [10.0, -4.0, 6.0]}))

    assert report.return_correlations.empty
    assert any("aggregated signal" in note for note in report.notes)


def test_no_trades_produces_an_honest_empty_report():
    report = analyse_correlations(pd.DataFrame())
    assert not report.findings
    assert any("undefined" in note for note in report.notes)


def test_the_report_says_nothing_was_removed():
    rng = np.random.default_rng(3)
    base = np.cumsum(rng.normal(0, 1, 300))
    report = analyse_correlations(
        pd.DataFrame(),
        isolated_equity={
            "a": curve(list(100 + base)),
            "b": curve(list(100 + base + rng.normal(0, 0.05, 300))),
        },
        isolated_trades={"a": 40, "b": 35},
    )
    text = report.render()

    assert "No strategy has been removed" in text
    assert "hypothesis about" in text


# ------------------------------------------------------------------- variants

def test_variants_are_produced_but_never_applied(config):
    report = analyse_correlations(
        pd.DataFrame(),
        isolated_equity={
            "trend_following": curve(list(100 + np.cumsum(np.ones(300)))),
            "momentum": curve(list(100 + np.cumsum(np.ones(300) * 1.001))),
        },
        isolated_trades={"trend_following": 40, "momentum": 20},
    )
    variants = weight_variants(config, report)

    assert "baseline" in variants
    assert variants["baseline"] is config
    # The original configuration is untouched: every strategy still enabled.
    assert config.strategies.strategies["momentum"].enabled
    assert config.strategies.strategies["momentum"].weight == 1.0


def test_a_variant_down_weights_the_thinner_evidence_base(config):
    report = analyse_correlations(
        pd.DataFrame(),
        isolated_equity={
            "trend_following": curve(list(100 + np.cumsum(np.ones(300)))),
            "momentum": curve(list(100 + np.cumsum(np.ones(300) * 1.001))),
        },
        isolated_trades={"trend_following": 40, "momentum": 20},
    )
    variants = weight_variants(config, report, down_weight=0.5)

    assert "halve_momentum" in variants
    assert variants["halve_momentum"].strategies.strategies["momentum"].weight == 0.5
    assert variants["drop_momentum"].strategies.strategies["momentum"].enabled is False
    # Everything else is identical, so a comparison varies only the hypothesis.
    assert (
        variants["halve_momentum"].strategies.strategies["trend_following"]
        == config.strategies.strategies["trend_following"]
    )


def test_no_findings_means_no_variants(config):
    empty = analyse_correlations(pd.DataFrame())
    assert set(weight_variants(config, empty)) == {"baseline"}


# ---------------------------------------------------------------- comparison

def test_a_trade_off_is_not_called_an_improvement():
    rows = [
        variant_row("baseline", _metrics(expectancy_r=0.10, max_drawdown=0.10)),
        variant_row("halve_x", _metrics(expectancy_r=0.15, max_drawdown=0.18)),
    ]
    text = summarise_variant_comparison(rows)
    assert "a trade-off, not an improvement" in text


def test_better_on_both_is_still_qualified():
    rows = [
        variant_row("baseline", _metrics(expectancy_r=0.10, max_drawdown=0.20)),
        variant_row("halve_x", _metrics(expectancy_r=0.15, max_drawdown=0.15)),
    ]
    text = summarise_variant_comparison(rows)
    assert "on this one window" in text
    assert "left unchanged unless a walk-forward run agrees" in text


def test_a_comparison_without_a_baseline_says_so():
    rows = [variant_row("halve_x", _metrics(expectancy_r=0.1, max_drawdown=0.1))]
    assert "No baseline" in summarise_variant_comparison(rows)


def _metrics(**overrides):
    from app.backtest.metrics import PerformanceMetrics

    return PerformanceMetrics(total_trades=50, **overrides)
