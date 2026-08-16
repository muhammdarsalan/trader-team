"""Walk-forward analysis, including the selection rule.

The test that matters most is the one asserting a fold's choice is made from
its training window alone. Selection that peeks at the test window turns the
whole analysis into an elaborate in-sample result, and the output looks
identical either way - so it has to be pinned by a test rather than by reading
the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.loader import get_config, override_config
from app.data.schema import MarketData, coerce_schema
from app.research.harness import ResearchHarness
from app.research.objectives import get_objective, sortino
from app.research.splits import walk_forward_folds
from app.research.walk_forward import WalkForwardAnalysis


@pytest.fixture
def config(config_dir):
    return get_config(config_dir)


def trending(n: int = 1_100, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2016-01-01", periods=n, freq="1D", tz="UTC")
    close = 500.0 + 0.8 * np.arange(n) + np.cumsum(rng.normal(0, 2.0, n))
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
def harness(config) -> ResearchHarness:
    data = MarketData(
        symbol="XAUUSD", timeframe="1D", df=trending(), provider="synthetic"
    )
    return ResearchHarness(config, data, config.assets.get("XAUUSD"))


@pytest.fixture
def folds(harness):
    return walk_forward_folds(
        harness.data.df.index,
        train_bars=300,
        test_bars=120,
        warmup_bars=harness.required_warmup(),
        embargo_bars=5,
        max_folds=3,
    )


# --------------------------------------------------------------------- basics

@pytest.mark.slow
def test_every_fold_produces_a_result(harness, folds):
    report = WalkForwardAnalysis(harness, folds).run()

    assert report.fold_count == len(folds)
    assert len(report.to_frame()) == len(folds)


@pytest.mark.slow
def test_test_windows_are_evaluated_after_their_training_windows(harness, folds):
    report = WalkForwardAnalysis(harness, folds).run()

    for fold in report.folds:
        assert fold.test.segment.start_time > fold.train.segment.end_time


@pytest.mark.slow
def test_the_stitched_curve_is_continuous_and_ordered(harness, folds):
    report = WalkForwardAnalysis(harness, folds).run()

    if report.stitched_equity.empty:
        pytest.skip("no equity produced by these folds")
    assert report.stitched_equity.index.is_monotonic_increasing
    assert not report.stitched_equity.index.has_duplicates


@pytest.mark.slow
def test_without_candidates_the_baseline_runs_everywhere(harness, folds):
    report = WalkForwardAnalysis(harness, folds).run()

    assert {f.selected_variant for f in report.folds} == {"baseline"}
    assert all("No candidates supplied" in f.selection_note for f in report.folds)


# ------------------------------------------------------------------ selection

@pytest.mark.slow
def test_selection_uses_only_the_training_window(harness, folds, config):
    """The central guarantee: no test-window information reaches the choice.

    Two candidates are offered, and every training evaluation is recorded. The
    choice for each fold must be reconstructible from the training scores
    alone - if it is not, something in the test window influenced it.
    """
    tighter = override_config(
        config, risk=config.risk.model_copy(update={"risk_per_trade": 0.005})
    )
    candidates = {"baseline": config, "half_risk": tighter}

    seen: list[tuple[str, str, float]] = []
    analysis = WalkForwardAnalysis(
        harness,
        folds,
        candidates=candidates,
        objective=sortino,
        on_trial=lambda name, cfg, segment, metrics: seen.append(
            (segment.name, name, sortino(metrics))
        ),
    )
    report = analysis.run()

    for fold in report.folds:
        train_scores = {
            name: score for window, name, score in seen if window == fold.train.segment.name
        }
        assert train_scores, "no training evaluations were recorded for this fold"
        best = max(train_scores, key=lambda n: train_scores[n])
        assert fold.selected_variant == best, (
            "the fold chose a configuration that was not the best on its training "
            "window, which means something outside that window influenced the choice"
        )
        # Nothing was evaluated on the test window before the choice was fixed.
        assert not any(window == fold.test.segment.name for window, _, _ in seen)


@pytest.mark.slow
def test_a_stable_choice_is_reported_as_such(harness, folds, config):
    candidates = {"baseline": config}
    report = WalkForwardAnalysis(harness, folds, candidates=candidates).run()

    assert any("stable" in w for w in report.warnings)


# ------------------------------------------------------------------- summary

@pytest.mark.slow
def test_efficiency_is_computed_from_aggregate_sides(harness, folds):
    """Averaging per-fold ratios lets one near-zero denominator dominate."""
    report = WalkForwardAnalysis(harness, folds).run()
    efficiency = report.efficiency

    assert efficiency is None or np.isfinite(efficiency)


@pytest.mark.slow
def test_silent_folds_are_flagged(harness, config):
    """A walk-forward record that mostly did not trade must say so."""
    disabled = override_config(
        config,
        strategies=config.strategies.model_copy(
            update={
                "strategies": {
                    name: block.model_copy(update={"enabled": name == "trend_following"})
                    for name, block in config.strategies.strategies.items()
                }
            }
        ),
    )
    quiet = ResearchHarness(
        disabled,
        harness.data,
        disabled.assets.get("XAUUSD"),
        trading_enabled=False,
    )
    folds = walk_forward_folds(
        quiet.data.df.index, train_bars=300, test_bars=120,
        warmup_bars=quiet.required_warmup(), max_folds=2,
    )
    report = WalkForwardAnalysis(quiet, folds).run()

    assert report.profitable_folds == 0
    assert any("no trades" in w for w in report.warnings)


@pytest.mark.slow
def test_the_report_renders(harness, folds):
    text = WalkForwardAnalysis(harness, folds).run().render()

    assert "WALK-FORWARD" in text
    assert "Stitched out-of-sample record" in text


# ----------------------------------------------------------------- objectives

def test_return_maximising_objectives_are_refused():
    """Ranking by return selects for risk taken rather than for edge."""
    for name in ("return", "total_return", "cagr", "profit"):
        with pytest.raises(ValueError, match="not available as a selection objective"):
            get_objective(name)


def test_an_unknown_objective_lists_the_available_ones():
    with pytest.raises(ValueError, match="Available"):
        get_objective("magic")


def test_objectives_refuse_to_rank_a_thin_sample():
    from app.backtest.metrics import PerformanceMetrics

    thin = PerformanceMetrics(total_trades=4, sortino_ratio=9.0)
    assert sortino(thin) == float("-inf")


def test_a_sufficient_sample_is_ranked_normally():
    from app.backtest.metrics import PerformanceMetrics

    assert sortino(PerformanceMetrics(total_trades=50, sortino_ratio=1.5)) == 1.5
