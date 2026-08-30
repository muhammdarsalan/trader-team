"""Frozen holdout protection.

These tests exercise the real ledger - a SQLite file on disk - not a mock, so
the persistence and restart behaviour is the actual behaviour. The point of the
mechanism is that a holdout's decay from "out of sample" to "in sample" becomes
recorded rather than trusted, and that development code can be *stopped* from
consuming it; both are asserted here against real seals and real accesses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.cache import frame_checksum
from app.research.holdout import (
    HoldoutIntegrityError,
    HoldoutRegistry,
    HoldoutViolationError,
    window_fingerprint,
)


def _series(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2015-01-01", periods=n, freq="1D", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=index,
    )


@pytest.fixture
def registry(tmp_path) -> HoldoutRegistry:
    return HoldoutRegistry(tmp_path / "holdout.db")


@pytest.fixture
def window():
    df = _series()
    split = df.index[int(len(df) * 0.7)]
    return df, split, df.index[-1]


# ------------------------------------------------------------------- sealing

def test_sealing_records_the_window_and_its_fingerprint(registry, window):
    df, start, end = window
    seal = registry.seal(
        symbol="XAUUSD", timeframe="1D", data=df, start_time=start, end_time=end,
        git_revision="abc123", random_seed=42, label="test holdout",
    )
    assert seal.bars == len(df.loc[start:end])
    assert seal.data_fingerprint == window_fingerprint(df, start, end)
    assert registry.is_sealed("XAUUSD", "1D", start, end)
    # Persisted, not just returned.
    assert registry.get_seal(seal.holdout_id) is not None


def test_the_holdout_id_is_deterministic_from_the_window(registry, window):
    df, start, end = window
    a = registry.seal(symbol="XAUUSD", timeframe="1D", data=df, start_time=start, end_time=end)
    b = registry.seal(symbol="XAUUSD", timeframe="1D", data=df, start_time=start, end_time=end)
    assert a.holdout_id == b.holdout_id, "re-sealing the same window is the same seal"


def test_sealing_an_empty_window_is_refused(registry):
    df = _series()
    future = df.index[-1] + pd.Timedelta(days=10)
    with pytest.raises(Exception, match="nothing to seal"):
        registry.seal(symbol="XAUUSD", timeframe="1D", data=df,
                      start_time=future, end_time=future + pd.Timedelta(days=5))


# ------------------------------------------------------------ access logging

def test_the_first_evaluation_is_a_clean_single_touch(registry, window):
    df, start, end = window
    seal = registry.seal(symbol="XAUUSD", timeframe="1D", data=df, start_time=start, end_time=end)

    access = registry.evaluate(seal, data=df, experiment_id="EXP-1", purpose="oos")
    assert access.touch_number == 1
    assert access.is_first_touch
    assert access.integrity_ok
    assert access.warnings == ()
    assert registry.touch_count(seal.holdout_id) == 1


def test_access_by_id_resolves_the_seal(registry, window):
    df, start, end = window
    seal = registry.seal(symbol="XAUUSD", timeframe="1D", data=df, start_time=start, end_time=end)
    access = registry.evaluate(seal.holdout_id, data=df, experiment_id="EXP-1")
    assert access.holdout_id == seal.holdout_id


def test_evaluating_an_unknown_holdout_raises(registry, window):
    df, _, _ = window
    with pytest.raises(Exception, match="No holdout is sealed"):
        registry.evaluate("HOLD-doesnotexist", data=df)


# ----------------------------------------------------------- repeated touch

def test_a_second_touch_warns_that_it_is_no_longer_out_of_sample(registry, window):
    df, start, end = window
    seal = registry.seal(symbol="XAUUSD", timeframe="1D", data=df, start_time=start, end_time=end)

    registry.evaluate(seal, data=df, experiment_id="EXP-1")
    second = registry.evaluate(seal, data=df, experiment_id="EXP-2")

    assert second.touch_number == 2
    assert not second.is_first_touch
    assert second.warnings, "a second touch must warn"
    assert "no longer out of sample" in second.warnings[0]
    status = registry.status_for(seal.holdout_id)
    assert status.over_touched
    assert status.touch_count == 2


def test_touch_history_is_ordered_and_complete(registry, window):
    df, start, end = window
    seal = registry.seal(symbol="XAUUSD", timeframe="1D", data=df, start_time=start, end_time=end)
    for i in range(3):
        registry.evaluate(seal, data=df, experiment_id=f"EXP-{i}")
    accesses = registry.accesses(seal.holdout_id)
    assert [a.touch_number for a in accesses] == [1, 2, 3]
    assert [a.experiment_id for a in accesses] == ["EXP-0", "EXP-1", "EXP-2"]


# ------------------------------------------------------ fingerprint mismatch

def test_evaluating_with_changed_data_is_logged_as_an_integrity_failure(registry, window):
    df, start, end = window
    seal = registry.seal(symbol="XAUUSD", timeframe="1D", data=df, start_time=start, end_time=end)

    tampered = df.copy()
    tampered.iloc[-1, tampered.columns.get_loc("close")] *= 1.05
    access = registry.evaluate(seal, data=tampered, experiment_id="EXP-bad")

    assert access.integrity_ok is False
    assert any("did not match" in w for w in access.warnings)
    # Still recorded - a dropped record would be worse than a logged failure.
    assert registry.touch_count(seal.holdout_id) == 1


def test_resealing_the_same_window_with_different_data_is_refused(registry, window):
    df, start, end = window
    registry.seal(symbol="XAUUSD", timeframe="1D", data=df, start_time=start, end_time=end)

    tampered = df.copy()
    tampered.iloc[-1, tampered.columns.get_loc("close")] *= 1.05
    with pytest.raises(HoldoutIntegrityError, match="changed after it was frozen"):
        registry.seal(symbol="XAUUSD", timeframe="1D", data=tampered,
                      start_time=start, end_time=end)


# ----------------------------------------------- use during optimisation

def test_development_may_not_run_on_a_sealed_holdout(registry, window):
    """The technical half of the guarantee: a sweep is stopped, not trusted."""
    df, start, end = window
    registry.seal(symbol="XAUUSD", timeframe="1D", data=df, start_time=start, end_time=end)

    holdout_fp = window_fingerprint(df, start, end)
    with pytest.raises(HoldoutViolationError, match="frozen holdout"):
        registry.assert_available_for_development(holdout_fp, purpose="parameter_sweep")


def test_development_may_run_on_a_non_holdout_window(registry, window):
    df, start, end = window
    registry.seal(symbol="XAUUSD", timeframe="1D", data=df, start_time=start, end_time=end)

    development = df.loc[: df.index[int(len(df) * 0.5)]]
    # Must not raise: this window is not sealed.
    registry.assert_available_for_development(
        frame_checksum(development), purpose="parameter_sweep"
    )


# -------------------------------------------------------- restart / persist

def test_seals_and_touch_counts_survive_a_restart(tmp_path, window):
    df, start, end = window
    path = tmp_path / "holdout.db"

    first = HoldoutRegistry(path)
    seal = first.seal(symbol="XAUUSD", timeframe="1D", data=df, start_time=start, end_time=end)
    first.evaluate(seal, data=df, experiment_id="EXP-1")
    first.evaluate(seal, data=df, experiment_id="EXP-2")

    # A brand-new registry over the same file is a process restart in miniature.
    restarted = HoldoutRegistry(path)
    assert restarted.get_seal(seal.holdout_id) is not None
    assert restarted.touch_count(seal.holdout_id) == 2
    assert restarted.status_for(seal.holdout_id).over_touched

    # And the guard still fires after the restart - the seal is durable.
    with pytest.raises(HoldoutViolationError):
        restarted.assert_available_for_development(
            window_fingerprint(df, start, end), purpose="sweep"
        )


def test_status_summarises_every_seal_for_reporting(registry, window):
    df, start, end = window
    seal = registry.seal(symbol="XAUUSD", timeframe="1D", data=df, start_time=start, end_time=end,
                         label="the one")
    registry.evaluate(seal, data=df, experiment_id="EXP-1")

    statuses = registry.status()
    assert len(statuses) == 1
    payload = statuses[0].to_dict()
    assert payload["seal"]["label"] == "the one"
    assert payload["touch_count"] == 1
    assert payload["untouched"] is False
    assert payload["over_touched"] is False
