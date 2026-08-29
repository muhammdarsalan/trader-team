"""The machine-readable report must carry everything the report object holds.

This exists because of a specific defect. ``ValidationReport`` held a
``robustness`` section, rendered it in the text report, and wrote
``parameter_sensitivity.csv`` from it — but ``to_dict()`` never emitted it, and
neither did ``variant_summary`` or ``regime_performance``. Every consumer that
parses ``validation_report.json`` rather than reading the prose therefore saw a
study with no parameter-sensitivity analysis in it at all, and nothing failed.

A hand-written list of expected keys would not have caught it, because the same
oversight that drops a section from ``to_dict()`` drops it from the list. So the
test below derives its expectations from the dataclass's own fields: adding a
section to :class:`ValidationReport` without serialising it fails here
automatically.
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import pytest

from app.backtest.metrics import PerformanceMetrics
from app.research.report import ValidationReport
from app.research.robustness import ParameterSensitivity, RobustnessReport, SweepPoint

#: Fields that are metadata *about* the report rather than sections of it, and
#: are legitimately absent from the serialised form or folded into another key.
#: Empty on purpose: every field currently carries content a reader needs.
NOT_SERIALISED: set[str] = set()


def _sensitivity(parameter: str, *, fragile: bool) -> ParameterSensitivity:
    """A sensitivity built from real metrics objects, not stubs."""
    points = tuple(
        SweepPoint(
            parameter=parameter,
            value=value,
            is_baseline=(value == 2.0),
            metrics=PerformanceMetrics(total_trades=40, expectancy_r=0.1 * i),
            objective=0.5 + 0.1 * i,
        )
        for i, value in enumerate([1.0, 2.0, 3.0])
    )
    return ParameterSensitivity(
        parameter=parameter,
        baseline_value=2.0,
        points=points,
        objective_name="sortino",
        mean=0.6,
        std=0.08,
        coefficient_of_variation=0.13,
        positive_fraction=1.0,
        worst_adjacent_drop=0.1,
        baseline_is_peak=fragile,
        fragile=fragile,
        notes=("swept over three points",),
    )


@pytest.fixture
def populated_report() -> ValidationReport:
    """A report with every optional section filled in.

    The point of the test is that no section is dropped, so a fixture that left
    sections at None would pass while proving nothing.
    """
    report = ValidationReport(
        experiment_id="RES-testfixture01",
        symbol="XAUUSD",
        timeframe="1D",
        period_start="2015-01-01",
        period_end="2018-12-31",
        bars=1045,
        data_provider="synthetic",
        data_checksum="abc123",
        data_quality="WARNING across the series; each window graded separately",
        git_revision="deadbeef",
        random_seed=42,
        created_at="2026-08-29T00:00:00+00:00",
        spec={"objective": "sortino"},
        notes=["a note that must survive serialisation"],
    )
    report.robustness = RobustnessReport(
        segment="in_sample",
        objective_name="sortino",
        sensitivities=[
            _sensitivity("risk.risk_per_trade", fragile=True),
            _sensitivity("execution.slippage_value", fragile=False),
        ],
    )
    report.variant_rows = [
        {"variant": "baseline", "trades": 40, "expectancy_r": 0.05, "max_drawdown": 0.1},
        {"variant": "halve_momentum", "trades": 33, "expectancy_r": 0.02, "max_drawdown": 0.12},
    ]
    report.variant_summary = "  Variant comparison on unseen data (baseline: +0.050R)"
    report.regime_performance = pd.DataFrame(
        [
            {"regime": "TRENDING_UP", "trades": 12, "expectancy_r": 0.2},
            {"regime": "UNCERTAIN", "trades": 9, "expectancy_r": -0.1},
        ]
    )
    return report


def test_to_dict_carries_every_section_the_report_holds(populated_report):
    """Every dataclass field must reach the serialised form.

    Derived from ``dataclasses.fields`` rather than a literal list, so the test
    fails on a section added later and not serialised — which is the failure
    mode that produced it.
    """
    payload = populated_report.to_dict()
    aliases = ValidationReport._DICT_ALIASES

    missing = []
    for field in dataclasses.fields(ValidationReport):
        if field.name in NOT_SERIALISED:
            continue
        key = aliases.get(field.name, field.name)
        if key not in payload:
            missing.append(f"{field.name} (expected key {key!r})")

    assert not missing, (
        "these sections exist on ValidationReport but never reach to_dict(), so any "
        "consumer reading validation_report.json cannot see them: " + ", ".join(missing)
    )


def test_the_previously_dropped_sections_carry_real_content(populated_report):
    """Present-but-empty would satisfy the key check and still lose the finding."""
    payload = populated_report.to_dict()

    robustness = payload["robustness"]
    assert robustness is not None
    assert robustness["segment"] == "in_sample"
    assert robustness["objective_name"] == "sortino"
    assert robustness["fragile_parameters"] == ["risk.risk_per_trade"], (
        "the fragile-parameter finding is the point of a sensitivity sweep and must "
        "survive serialisation"
    )
    assert len(robustness["sensitivities"]) == 2
    first = robustness["sensitivities"][0]
    assert first["fragile"] is True
    assert first["baseline_is_peak"] is True
    assert len(first["points"]) == 3
    assert first["notes"] == ["swept over three points"]

    assert payload["variant_summary"].startswith("  Variant comparison")
    assert payload["regime_performance"] == [
        {"regime": "TRENDING_UP", "trades": 12, "expectancy_r": 0.2},
        {"regime": "UNCERTAIN", "trades": 9, "expectancy_r": -0.1},
    ]


def test_to_dict_is_json_serialisable(populated_report):
    """``save()`` writes this with ``json.dumps``; a non-serialisable value breaks it."""
    text = json.dumps(populated_report.to_dict(), default=str)
    restored = json.loads(text)
    assert restored["robustness"]["fragile_parameters"] == ["risk.risk_per_trade"]
    assert restored["experiment_id"] == "RES-testfixture01"


def test_absent_sections_serialise_as_null_not_as_omitted():
    """A study that skipped a section must say so, not leave the key out.

    An absent key and a null are different to a consumer: the first looks like
    an older schema, the second like an analysis that did not run. The dashboard
    distinguishes "not run" from "unavailable", and it can only do that if the
    key is present.
    """
    payload = ValidationReport(
        experiment_id="RES-empty", symbol="EURUSD", timeframe="1D"
    ).to_dict()

    for key in ("robustness", "walk_forward", "monte_carlo", "correlation", "overfitting"):
        assert key in payload, f"{key} was omitted rather than set to null"
        assert payload[key] is None

    assert payload["regime_performance"] == []
    assert payload["variants"] == []
    assert payload["variant_summary"] == ""


def test_saved_json_matches_to_dict(populated_report, tmp_path):
    """What lands on disk is what ``to_dict`` produced.

    ``save()`` writes several files; this asserts the JSON one is the full
    payload rather than a reduced copy that drifted from it.
    """
    base = populated_report.save(tmp_path)
    on_disk = json.loads((base / "validation_report.json").read_text())

    assert on_disk.keys() == populated_report.to_dict().keys()
    assert on_disk["robustness"]["fragile_parameters"] == ["risk.risk_per_trade"]
    assert (base / "parameter_sensitivity.csv").exists(), (
        "the CSV and the JSON should both carry the sweep, not one or the other"
    )
