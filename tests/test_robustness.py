"""Parameter surgery and sensitivity shape.

The sweep machinery itself is tested here on real configurations. The
end-to-end sweep over market data is exercised by the study test, which is
marked slow.
"""

from __future__ import annotations

import pytest

from app.backtest.metrics import PerformanceMetrics
from app.config.loader import get_config
from app.research.robustness import (
    ParameterPathError,
    SweepPoint,
    _describe,
    get_parameter,
    neighbourhood,
    set_parameter,
)


@pytest.fixture
def config(config_dir):
    return get_config(config_dir)


# ------------------------------------------------------------ reading values

def test_a_top_level_field_is_readable(config):
    assert get_parameter(config, "risk.risk_per_trade") == config.risk.risk_per_trade


def test_a_nested_strategy_parameter_is_readable(config):
    value = get_parameter(config, "strategies.strategies.trend_following.params.fast_ema")
    assert value == config.strategies.strategies["trend_following"].params["fast_ema"]


def test_an_unknown_field_names_the_alternatives(config):
    with pytest.raises(ParameterPathError, match="Available"):
        get_parameter(config, "risk.risk_per_trad")


def test_an_unknown_strategy_key_is_caught(config):
    with pytest.raises(ParameterPathError):
        get_parameter(config, "strategies.strategies.nonexistent.params.fast_ema")


# ------------------------------------------------------------ writing values

def test_setting_a_value_returns_a_new_configuration(config):
    changed = set_parameter(config, "risk.risk_per_trade", 0.02)

    assert changed.risk.risk_per_trade == 0.02
    assert config.risk.risk_per_trade != 0.02, "the original must not be mutated"


def test_setting_a_nested_strategy_parameter_leaves_the_rest_alone(config):
    changed = set_parameter(
        config, "strategies.strategies.trend_following.params.fast_ema", 34
    )

    assert changed.strategies.strategies["trend_following"].params["fast_ema"] == 34
    assert (
        changed.strategies.strategies["momentum"]
        == config.strategies.strategies["momentum"]
    )
    assert changed.risk == config.risk


def test_an_invalid_value_is_rejected_rather_than_silently_accepted(config):
    """A sweep must not quietly produce settings the platform would reject."""
    with pytest.raises(ParameterPathError, match="invalid"):
        set_parameter(config, "risk.risk_per_trade", -0.5)


def test_a_cross_field_rule_is_still_enforced(config):
    """macd_fast above macd_slow inverts the indicator; the model says so."""
    with pytest.raises(ParameterPathError, match="invalid"):
        set_parameter(config, "features.macd_fast", 99)


def test_naming_a_whole_section_is_an_error(config):
    with pytest.raises(ParameterPathError, match="whole configuration section"):
        set_parameter(config, "risk", None)


def test_an_unknown_section_is_an_error(config):
    with pytest.raises(ParameterPathError, match="Unknown configuration section"):
        set_parameter(config, "nonsense.field", 1)


# ------------------------------------------------------------ neighbourhoods

def test_an_integer_neighbourhood_stays_integral():
    values = neighbourhood(14, span=0.3, points=5)
    assert all(isinstance(v, int) for v in values)
    assert 14 in values
    assert min(values) >= 1


def test_a_float_neighbourhood_brackets_the_baseline():
    values = neighbourhood(0.01, span=0.3, points=5)
    assert min(values) < 0.01 < max(values)
    assert len(values) == 5


def test_a_boolean_neighbourhood_is_both_values():
    assert set(neighbourhood(True)) == {True, False}


def test_a_non_numeric_baseline_is_refused():
    with pytest.raises(TypeError):
        neighbourhood("sometimes")


# ------------------------------------------------------------ reading a shape

def point(value, objective, baseline=False) -> SweepPoint:
    return SweepPoint(
        parameter="p", value=value, is_baseline=baseline, objective=objective,
        metrics=PerformanceMetrics(total_trades=50),
    )


def test_a_flat_neighbourhood_is_not_fragile():
    sensitivity = _describe(
        "p", 3, [point(1, 1.00), point(2, 1.02), point(3, 1.01, True),
                 point(4, 0.99), point(5, 1.00)], "sortino",
    )

    assert not sensitivity.fragile
    assert any("flat" in note for note in sensitivity.notes)


def test_a_cliff_is_fragile():
    sensitivity = _describe(
        "p", 3, [point(1, 1.0), point(2, 1.0), point(3, 1.0, True),
                 point(4, -2.0), point(5, -2.0)], "sortino",
    )

    assert sensitivity.fragile
    assert any("cliff" in note for note in sensitivity.notes)


def test_a_peak_at_the_shipped_value_is_treated_as_suspicious():
    """A tuned parameter looks exactly like this, so it is a red flag."""
    sensitivity = _describe(
        "p", 3, [point(1, 0.2), point(2, 0.6), point(3, 1.4, True),
                 point(4, 0.5), point(5, 0.1)], "sortino",
    )

    assert sensitivity.baseline_is_peak
    assert sensitivity.fragile
    assert any("suspicion rather than reassurance" in note for note in sensitivity.notes)


def test_a_sign_change_across_the_neighbourhood_is_reported():
    sensitivity = _describe(
        "p", 3, [point(1, 0.5), point(2, 0.3), point(3, 0.1, True),
                 point(4, -0.2), point(5, -0.4)], "sortino",
    )
    assert 0 < sensitivity.positive_fraction < 1
    assert any("sign of the result" in note for note in sensitivity.notes)


def test_too_few_usable_points_declines_to_read_a_shape():
    failed = SweepPoint(
        parameter="p", value=9, is_baseline=False, objective=float("nan"),
        metrics=PerformanceMetrics(), error="boom",
    )
    sensitivity = _describe("p", 3, [point(3, 1.0, True), failed, failed], "sortino")

    assert not sensitivity.fragile
    assert any("cannot be read" in note for note in sensitivity.notes)
