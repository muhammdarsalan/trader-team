"""The standardized Signal contract."""

from __future__ import annotations

import pandas as pd
import pytest

from app.signals.models import Signal, SignalDirection


def long_signal(**overrides) -> Signal:
    base = {
        "strategy": "test", "symbol": "XAUUSD", "timeframe": "1D",
        "direction": SignalDirection.LONG, "confidence": 0.7,
        "entry_price": 2000.0, "stop_loss": 1980.0, "take_profit": 2040.0,
    }
    return Signal(**{**base, **overrides})


def short_signal(**overrides) -> Signal:
    base = {
        "strategy": "test", "symbol": "XAUUSD", "timeframe": "1D",
        "direction": SignalDirection.SHORT, "confidence": 0.7,
        "entry_price": 2000.0, "stop_loss": 2020.0, "take_profit": 1960.0,
    }
    return Signal(**{**base, **overrides})


# ------------------------------------------------------------------- direction

def test_wait_is_not_actionable():
    assert Signal.wait("test", "XAUUSD", "1D").is_actionable is False


def test_direction_signs():
    assert SignalDirection.LONG.sign == 1
    assert SignalDirection.SHORT.sign == -1
    assert SignalDirection.WAIT.sign == 0


# ------------------------------------------------------------------ validation

def test_valid_long_signal_is_accepted():
    assert long_signal().is_actionable


def test_valid_short_signal_is_accepted():
    assert short_signal().is_actionable


def test_long_stop_must_be_below_entry():
    """A stop on the wrong side yields a negative risk distance downstream."""
    with pytest.raises(ValueError, match="LONG stop_loss.*must be below"):
        long_signal(stop_loss=2010.0)


def test_short_stop_must_be_above_entry():
    with pytest.raises(ValueError, match="SHORT stop_loss.*must be above"):
        short_signal(stop_loss=1990.0)


def test_long_target_must_be_above_entry():
    with pytest.raises(ValueError, match="LONG take_profit.*must be above"):
        long_signal(take_profit=1990.0)


def test_short_target_must_be_below_entry():
    with pytest.raises(ValueError, match="SHORT take_profit.*must be below"):
        short_signal(take_profit=2010.0)


def test_actionable_signal_requires_a_stop():
    """Without a stop the risk engine cannot size the position at all."""
    with pytest.raises(ValueError, match="requires stop_loss"):
        long_signal(stop_loss=None)


def test_actionable_signal_requires_an_entry():
    with pytest.raises(ValueError, match="requires entry_price"):
        long_signal(entry_price=None)


def test_prices_must_be_positive():
    with pytest.raises(ValueError, match="entry_price must be positive"):
        long_signal(entry_price=-1.0)
    with pytest.raises(ValueError, match="stop_loss must be positive"):
        long_signal(stop_loss=0.0)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, 2.0])
def test_confidence_must_be_bounded(confidence):
    with pytest.raises(ValueError, match=r"confidence must be in \[0, 1\]"):
        long_signal(confidence=confidence)


def test_wait_signals_skip_price_validation():
    """A WAIT has no entry or stop, and that is correct, not an error."""
    signal = Signal.wait("test", "XAUUSD", "1D", reasoning=["nothing to do"])
    assert signal.entry_price is None
    assert signal.stop_loss is None


def test_target_is_optional():
    assert long_signal(take_profit=None).take_profit is None


# ------------------------------------------------------------------- arithmetic

def test_risk_per_unit_is_the_stop_distance():
    assert long_signal().risk_per_unit == pytest.approx(20.0)
    assert short_signal().risk_per_unit == pytest.approx(20.0)


def test_reward_risk_ratio():
    assert long_signal().reward_risk_ratio == pytest.approx(2.0)


def test_reward_risk_is_none_without_a_target():
    assert long_signal(take_profit=None).reward_risk_ratio is None


def test_wait_has_no_risk_arithmetic():
    signal = Signal.wait("test", "XAUUSD", "1D")
    assert signal.risk_per_unit is None
    assert signal.reward_risk_ratio is None


# --------------------------------------------------------------------- records

def test_wait_records_its_reason():
    """Knowing why nothing happened distinguishes 'saw nothing' from 'was broken'."""
    signal = Signal.wait(
        "test", "XAUUSD", "1D", reasoning=["ADX below threshold", "no setup"]
    )
    assert len(signal.reasoning) == 2
    assert "ADX below threshold" in signal.reasoning


def test_wait_accepts_metadata():
    signal = Signal.wait("test", "XAUUSD", "1D", adx=15.2)
    assert signal.metadata["adx"] == 15.2


def test_to_dict_is_serialisable():
    payload = long_signal(timestamp=pd.Timestamp("2024-01-01", tz="UTC")).to_dict()
    assert payload["direction"] == "LONG"
    assert payload["timestamp"] == "2024-01-01T00:00:00+00:00"
    assert payload["reward_risk_ratio"] == pytest.approx(2.0)


def test_to_dict_stringifies_exotic_metadata():
    signal = long_signal(metadata={"when": pd.Timestamp("2024-01-01", tz="UTC")})
    assert isinstance(signal.to_dict()["metadata"]["when"], str)


def test_describe_reports_the_trade():
    text = long_signal().describe()
    assert "LONG" in text and "entry" in text and "stop" in text


def test_describe_of_wait_reports_the_reason():
    text = Signal.wait("test", "XAUUSD", "1D", reasoning=["no setup"]).describe()
    assert "WAIT" in text and "no setup" in text


def test_signal_is_immutable():
    signal = long_signal()
    with pytest.raises((AttributeError, TypeError)):
        signal.confidence = 0.9  # type: ignore[misc]


def test_error_message_names_the_strategy():
    """With five strategies running, the message must say which one is broken."""
    with pytest.raises(ValueError, match="breakout"):
        Signal(
            strategy="breakout", symbol="XAUUSD", timeframe="1D",
            direction=SignalDirection.LONG, confidence=0.5,
            entry_price=100.0, stop_loss=110.0,
        )
