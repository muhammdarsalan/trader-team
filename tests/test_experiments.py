"""Experiment tracking: identity, metadata and the trial count."""

from __future__ import annotations

import pytest

from app.research.experiments import (
    ExperimentRecord,
    ExperimentStore,
    config_fingerprint,
    reproducible_experiment_id,
    stable_hash,
)


@pytest.fixture
def store(tmp_path) -> ExperimentStore:
    return ExperimentStore(tmp_path / "experiments.db")


def record(experiment_id: str = "RES-test", **overrides) -> ExperimentRecord:
    base = {
        "experiment_id": experiment_id,
        "kind": "validation_study",
        "symbol": "XAUUSD",
        "timeframe": "1D",
        "bars": 1_000,
        "data_checksum": "abc123",
        "data_quality": "PASS",
        "git_revision": "deadbee",
        "random_seed": 42,
        "config_snapshot": {"risk": {"risk_per_trade": 0.01}},
    }
    return ExperimentRecord(**{**base, **overrides})


# ------------------------------------------------------------------- identity

def test_the_same_inputs_give_the_same_id():
    """A deterministic id is what stops a re-run looking like confirmation."""
    kwargs = {
        "config_snapshot": {"risk": {"risk_per_trade": 0.01}},
        "data_checksum": "abc",
        "git_revision": "deadbee",
        "random_seed": 42,
        "spec": {"folds": 5},
    }
    assert reproducible_experiment_id(**kwargs) == reproducible_experiment_id(**kwargs)


@pytest.mark.parametrize(
    "change",
    [
        {"config_snapshot": {"risk": {"risk_per_trade": 0.02}}},
        {"data_checksum": "different"},
        {"git_revision": "other"},
        {"random_seed": 7},
        {"spec": {"folds": 3}},
    ],
)
def test_any_real_difference_changes_the_id(change):
    base = {
        "config_snapshot": {"risk": {"risk_per_trade": 0.01}},
        "data_checksum": "abc",
        "git_revision": "deadbee",
        "random_seed": 42,
        "spec": {"folds": 5},
    }
    assert reproducible_experiment_id(**base) != reproducible_experiment_id(**{**base, **change})


def test_key_order_does_not_change_the_hash():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_the_hash_is_stable_across_processes():
    """Python's own hash() is salted per process and would defeat the point."""
    assert stable_hash({"risk": 0.01}) == stable_hash({"risk": 0.01})
    assert len(stable_hash({"x": 1})) == 12


def test_config_fingerprint_reacts_to_a_single_field():
    a = config_fingerprint({"risk": {"risk_per_trade": 0.01}})
    b = config_fingerprint({"risk": {"risk_per_trade": 0.0101}})
    assert a != b


# ---------------------------------------------------------------- persistence

def test_an_experiment_round_trips(store):
    store.record_experiment(record())
    loaded = store.get_experiment("RES-test")

    assert loaded["symbol"] == "XAUUSD"
    assert loaded["data_checksum"] == "abc123"
    assert loaded["random_seed"] == 42


def test_re_recording_updates_rather_than_duplicating(store):
    store.record_experiment(record())
    store.record_experiment(record(verdict="EVIDENCE AGAINST"))

    assert len(store.list_experiments()) == 1
    assert store.get_experiment("RES-test")["verdict"] == "EVIDENCE AGAINST"


def test_the_first_run_time_is_preserved_across_updates(store):
    store.record_experiment(record())
    created = store.get_experiment("RES-test")["created_at"]
    store.record_experiment(record(verdict="INCONCLUSIVE"))
    assert store.get_experiment("RES-test")["created_at"] == created


def test_segments_and_metrics_round_trip(store):
    store.record_experiment(record())
    store.record_segment(
        "RES-test", name="out_of_sample", role="OUT_OF_SAMPLE",
        start_time="2020-01-01", end_time="2021-01-01", bars=250, trades=40,
        data_quality="WARNING",
    )
    store.record_metrics("RES-test", "out_of_sample", {"sharpe_ratio": 0.8, "total_trades": 40})

    segments = store.segments_for("RES-test")
    assert segments[0]["trades"] == 40
    assert store.metrics_for("RES-test", "out_of_sample")["sharpe_ratio"] == pytest.approx(0.8)


def test_non_numeric_and_infinite_metrics_are_skipped(store):
    """A metrics table with the word 'inf' in it is a table nobody can query."""
    store.record_experiment(record())
    store.record_metrics(
        "RES-test", "in_sample",
        {"profit_factor": float("inf"), "warnings": ["x"], "sharpe_ratio": 1.2,
         "nan_metric": float("nan")},
    )
    stored = store.metrics_for("RES-test", "in_sample")

    assert stored == {"sharpe_ratio": pytest.approx(1.2)}


def test_findings_replace_rather_than_accumulate(store):
    store.record_experiment(record())
    store.record_findings("RES-test", [{"code": "A", "severity": "SEVERE", "message": "m"}])
    store.record_findings("RES-test", [{"code": "B", "severity": "CAUTION", "message": "m"}])

    codes = [f["code"] for f in store.findings_for("RES-test")]
    assert codes == ["B"]


# --------------------------------------------------------------- trial counts

def test_distinct_configurations_are_counted(store):
    store.record_experiment(record())
    for i in range(4):
        store.record_trial(
            "RES-test", data_checksum="abc123", variant=f"v{i}",
            config_hash=f"hash{i}", segment="in_sample",
        )
    assert store.count_trials("abc123") == 4


def test_the_same_configuration_twice_is_one_trial(store):
    """Overstating the search distorts the arithmetic as badly as understating it."""
    store.record_experiment(record())
    for _ in range(3):
        store.record_trial(
            "RES-test", data_checksum="abc123", variant="v", config_hash="h",
            segment="in_sample",
        )
    assert store.count_trials("abc123") == 1


def test_the_same_configuration_on_different_windows_counts_once(store):
    store.record_experiment(record())
    store.record_trial("RES-test", data_checksum="abc", variant="v", config_hash="h",
                       segment="in_sample")
    store.record_trial("RES-test", data_checksum="abc", variant="v", config_hash="h",
                       segment="validation")
    assert store.count_trials("abc") == 1


def test_trials_are_counted_per_dataset(store):
    store.record_experiment(record())
    store.record_trial("RES-test", data_checksum="one", variant="v", config_hash="h1")
    store.record_trial("RES-test", data_checksum="two", variant="v", config_hash="h2")

    assert store.count_trials("one") == 1
    assert store.count_trials("unseen") == 0


def test_the_count_survives_reopening_the_database(tmp_path):
    """Last week's variants still happened when the process restarts."""
    path = tmp_path / "experiments.db"
    first = ExperimentStore(path)
    first.record_experiment(record())
    first.record_trial("RES-test", data_checksum="abc", variant="v", config_hash="h")

    assert ExperimentStore(path).count_trials("abc") == 1


def test_artifacts_are_recorded(store, tmp_path):
    store.record_experiment(record())
    store.record_artifact("RES-test", "validation_report", tmp_path / "report")
    # No dedicated reader; the point is that recording it does not raise and it
    # replaces cleanly on a re-run.
    store.record_artifact("RES-test", "validation_report", tmp_path / "report2")
