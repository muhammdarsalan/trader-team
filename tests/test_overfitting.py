"""Overfitting diagnostics and the data-snooping arithmetic."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.research.overfitting import (
    Severity,
    assess,
    bar_returns,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    normal_cdf,
    normal_ppf,
)

# --------------------------------------------------------------- normal maths

@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.5, 0.0), (0.975, 1.959_964), (0.025, -1.959_964), (0.99, 2.326_348)],
)
def test_normal_ppf_matches_known_quantiles(p, expected):
    assert normal_ppf(p) == pytest.approx(expected, abs=1e-5)


def test_normal_cdf_and_ppf_are_inverses():
    for p in (0.01, 0.1, 0.4, 0.6, 0.9, 0.99):
        assert normal_cdf(normal_ppf(p)) == pytest.approx(p, abs=1e-7)


def test_ppf_rejects_impossible_probabilities():
    with pytest.raises(ValueError):
        normal_ppf(0.0)


# ------------------------------------------------------ expected maximum

def test_a_single_trial_has_no_selection_advantage():
    assert expected_max_sharpe(1) == 0.0


def test_more_trials_raise_the_bar():
    """The core arithmetic: search alone produces impressive-looking results."""
    assert expected_max_sharpe(100) > expected_max_sharpe(20) > expected_max_sharpe(5) > 0


def test_the_benchmark_scales_with_the_spread_of_trials():
    assert expected_max_sharpe(50, sharpe_std=2.0) == pytest.approx(
        2 * expected_max_sharpe(50, sharpe_std=1.0)
    )


def test_a_hundred_worthless_strategies_expect_a_flattering_best():
    """With 100 trials the expected best is far above zero even with no edge."""
    assert expected_max_sharpe(100) > 2.0


# -------------------------------------------------------- deflated Sharpe

def test_deflated_sharpe_falls_as_the_search_widens():
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0.002, 0.01, 500))

    one_trial, _ = deflated_sharpe_ratio(returns, trials=1)
    many_trials, _ = deflated_sharpe_ratio(returns, trials=500)

    assert one_trial is not None and many_trials is not None
    assert many_trials < one_trial


def test_pure_noise_does_not_survive_deflation():
    rng = np.random.default_rng(9)
    noise = pd.Series(rng.normal(0.0, 0.01, 800))
    probability, _ = deflated_sharpe_ratio(noise, trials=50)
    assert probability is not None
    assert probability < 0.95


def test_a_short_sample_returns_nothing_rather_than_a_number():
    probability, diagnostics = deflated_sharpe_ratio(pd.Series([0.01, -0.01, 0.02]), trials=1)
    assert probability is None
    assert diagnostics["observations"] == 3


def test_a_flat_curve_returns_nothing():
    probability, _ = deflated_sharpe_ratio(pd.Series([0.0] * 100), trials=1)
    assert probability is None


def test_bar_returns_handles_an_empty_curve():
    assert bar_returns(pd.Series(dtype="float64")).empty


# ------------------------------------------------------------- the assessment

def good_returns(n: int = 400, seed: int = 2) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.001, 0.005, n))


def test_a_sign_flip_out_of_sample_is_severe():
    assessment = assess(
        trials=5, in_sample_sharpe=2.0, out_of_sample_sharpe=-0.4,
        in_sample_trades=100, out_of_sample_trades=60,
    )
    codes = {f.code for f in assessment.findings}
    assert "OOS_SIGN_FLIP" in codes
    assert assessment.worst_severity is Severity.SEVERE


def test_heavy_degradation_is_flagged():
    assessment = assess(
        trials=3, in_sample_sharpe=2.0, out_of_sample_sharpe=0.4,
        in_sample_trades=100, out_of_sample_trades=60,
    )
    assert "OOS_DEGRADATION" in {f.code for f in assessment.findings}
    assert assessment.degradation == pytest.approx(0.2)


def test_a_thin_out_of_sample_sample_is_flagged():
    assessment = assess(trials=1, out_of_sample_trades=6)
    finding = next(f for f in assessment.findings if f.code == "THIN_OOS_SAMPLE")
    assert finding.severity is Severity.SEVERE


def test_a_wide_search_is_flagged_on_its_own():
    assessment = assess(trials=250, out_of_sample_trades=200)
    assert "HEAVY_SEARCH" in {f.code for f in assessment.findings}


def test_a_moderate_search_is_a_caution_not_a_severity():
    assessment = assess(trials=25, out_of_sample_trades=200)
    finding = next(f for f in assessment.findings if f.code == "REPEATED_SEARCH")
    assert finding.severity is Severity.CAUTION


def test_inconsistent_folds_are_flagged():
    assessment = assess(trials=1, out_of_sample_trades=100, profitable_folds=(2, 6))
    assert "INCONSISTENT_FOLDS" in {f.code for f in assessment.findings}


def test_low_walk_forward_efficiency_is_flagged():
    assessment = assess(trials=1, out_of_sample_trades=100, walk_forward_efficiency=0.2)
    assert "LOW_WALK_FORWARD_EFFICIENCY" in {f.code for f in assessment.findings}


def test_fragile_parameters_are_severe():
    assessment = assess(
        trials=1, out_of_sample_trades=100, sensitivity_fragile=("features.rsi_period",)
    )
    finding = next(f for f in assessment.findings if f.code == "PARAMETER_FRAGILITY")
    assert finding.severity is Severity.SEVERE


def test_degraded_segments_are_reported():
    assessment = assess(
        trials=1, out_of_sample_trades=100, degraded_segments=("out_of_sample",)
    )
    assert "DEGRADED_SEGMENT_DATA" in {f.code for f in assessment.findings}


def test_sharpe_units_are_kept_apart():
    """An annualised Sharpe next to a per-bar benchmark is off by a large factor."""
    assessment = assess(
        trials=10, out_of_sample_sharpe=1.8, out_of_sample_returns=good_returns(),
        out_of_sample_trades=80,
    )
    assert assessment.observed_sharpe == 1.8
    assert assessment.observed_sharpe_per_bar is not None
    assert abs(assessment.observed_sharpe_per_bar) < abs(assessment.observed_sharpe)


def test_a_clean_assessment_still_refuses_to_endorse():
    assessment = assess(
        trials=1, in_sample_sharpe=1.0, out_of_sample_sharpe=0.9,
        in_sample_trades=200, out_of_sample_trades=180,
        out_of_sample_returns=good_returns(), walk_forward_efficiency=0.9,
    )
    text = assessment.render()
    assert "not the same as a validated result" in text or assessment.findings


def test_the_assessment_serialises():
    assessment = assess(trials=4, out_of_sample_trades=50)
    payload = assessment.to_dict()
    assert payload["trials"] == 4
    assert isinstance(payload["findings"], list)


def test_infinite_values_never_reach_the_report():
    assessment = assess(trials=1, in_sample_sharpe=0.0, out_of_sample_sharpe=1.0,
                        out_of_sample_trades=100)
    assert assessment.degradation is None or math.isfinite(assessment.degradation)
