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


@pytest.mark.slow
def test_the_study_seals_the_holdout_and_records_repeat_looks(tmp_path):
    """A study given a registry seals its OOS window and flags re-looks honestly."""
    from app.config.loader import get_config, override_config
    from app.data.schema import MarketData, coerce_schema
    from app.research.experiments import ExperimentStore
    from app.research.study import ValidationStudy

    n = 900
    rng = np.random.default_rng(11)
    index = pd.date_range("2016-01-01", periods=n, freq="1D", tz="UTC")
    close = 500.0 + 0.6 * np.arange(n) + np.cumsum(rng.normal(0, 3.0, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    df = coerce_schema(
        pd.DataFrame(
            {"open": open_, "high": np.maximum(open_, close) + 3.0,
             "low": np.minimum(open_, close) - 3.0, "close": close,
             "volume": np.full(n, 5_000.0)},
            index=index,
        )
    )
    data = MarketData(symbol="XAUUSD", timeframe="1D", df=df, provider="synthetic")

    cfg = get_config()
    r = cfg.research
    cfg = override_config(cfg, research=r.model_copy(update={
        "monte_carlo": r.monte_carlo.model_copy(update={"enabled": False}),
        "walk_forward": r.walk_forward.model_copy(update={"enabled": False}),
        "robustness": r.robustness.model_copy(update={"enabled": False}),
        "cost_stress": r.cost_stress.model_copy(update={"enabled": False}),
        "isolate_strategies": False,
    }))
    registry = HoldoutRegistry(tmp_path / "holdout.db")
    store = ExperimentStore(tmp_path / "e.db")

    first = ValidationStudy(cfg, data, cfg.assets.get("XAUUSD"), quality_status="PASS",
                            store=store, holdout_registry=registry).run()
    holdout = first.report.holdout
    assert holdout is not None
    assert holdout["touch_number"] == 1
    assert holdout["first_touch"] is True
    assert holdout["integrity_ok"] is True
    assert holdout["over_touched"] is False
    evidence = {e["category"]: e for e in first.report.evidence_summary()}
    assert evidence["Frozen holdout"]["status"] == "PARTIAL"

    # Re-running the study looks at the same sealed window again - recorded and
    # flagged, even though the configuration is identical.
    second = ValidationStudy(cfg, data, cfg.assets.get("XAUUSD"), quality_status="PASS",
                             store=store, holdout_registry=registry).run()
    assert second.report.holdout["over_touched"] is True
    evidence2 = {e["category"]: e for e in second.report.evidence_summary()}
    assert evidence2["Frozen holdout"]["status"] == "NOT_ESTABLISHED"


@pytest.mark.slow
def test_a_study_refuses_to_optimise_a_window_sealed_as_holdout(tmp_path):
    """If a development window is itself sealed, the study must refuse to sweep it."""
    from app.config.loader import get_config
    from app.data.schema import MarketData, coerce_schema
    from app.research.splits import chronological_split
    from app.research.study import ValidationStudy

    n = 900
    rng = np.random.default_rng(5)
    index = pd.date_range("2016-01-01", periods=n, freq="1D", tz="UTC")
    close = 500.0 + np.cumsum(rng.normal(0, 2.0, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    df = coerce_schema(
        pd.DataFrame(
            {"open": open_, "high": np.maximum(open_, close) + 3.0,
             "low": np.minimum(open_, close) - 3.0, "close": close,
             "volume": np.full(n, 5_000.0)},
            index=index,
        )
    )
    data = MarketData(symbol="XAUUSD", timeframe="1D", df=df, provider="synthetic")
    cfg = get_config()

    registry = HoldoutRegistry(tmp_path / "holdout.db")
    # Seal the study's in-sample window as if it were a holdout.
    study = ValidationStudy(cfg, data, cfg.assets.get("XAUUSD"), quality_status="PASS",
                            holdout_registry=registry)
    warmup = study.harness.required_warmup()
    segments = {s.name: s for s in chronological_split(
        df.index, fractions=tuple(cfg.research.split_fractions),
        warmup_bars=warmup, embargo_bars=cfg.research.embargo_bars)}
    in_sample = segments["in_sample"]
    registry.seal(symbol="XAUUSD", timeframe="1D", data=df,
                  start_time=in_sample.start_time, end_time=in_sample.end_time,
                  label="mis-sealed in-sample")

    with pytest.raises(HoldoutViolationError, match="Development must not touch"):
        study.run()


def test_the_dashboard_panel_reflects_holdout_status_honestly():
    """A first look reads clean; a repeat look reads as a red over-touch."""
    from dashboard.view import research_panel

    def panel_for(**holdout):
        ctx = {"availability": "AVAILABLE",
               "report": {"evidence_summary": [], "segments": [], "holdout": holdout}}
        return research_panel({"research": ctx})["holdout"]

    window = {"start": "2019-01-01", "end": "2021-12-31", "bars": 390}
    first = panel_for(holdout_id="HOLD-a", window=window, touch_number=1, touch_count=1,
                      first_touch=True, over_touched=False, integrity_ok=True, warnings=[])
    assert first["status"] == "SEALED, FIRST LOOK"
    assert first["tone"] == "ok"

    repeat = panel_for(holdout_id="HOLD-a", window=window, touch_number=2, touch_count=2,
                       first_touch=False, over_touched=True, integrity_ok=True,
                       warnings=["evaluated 2 times"])
    assert repeat["status"] == "OVER-TOUCHED"
    assert repeat["tone"] == "error"

    broken = panel_for(holdout_id="HOLD-a", window=window, touch_number=1, touch_count=1,
                       first_touch=True, over_touched=False, integrity_ok=False, warnings=[])
    assert broken["status"] == "INTEGRITY MISMATCH"
    assert broken["tone"] == "error"

    # No holdout attached: the panel must say absent, not invent a green status.
    absent = research_panel(
        {"research": {"availability": "AVAILABLE",
                      "report": {"evidence_summary": [], "segments": []}}}
    )["holdout"]
    assert absent is None


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
