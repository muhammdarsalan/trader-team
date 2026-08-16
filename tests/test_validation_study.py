"""The validation study end to end, and the report it produces.

Two kinds of test here. The report's verdict logic is exercised directly on
constructed results, because the interesting cases (a sign flip out of sample,
a thin sample) are hard to provoke reliably from market data. The study itself
is run once over synthetic data, marked slow, to prove the pieces compose and
that the out-of-sample window is reached last.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.metrics import PerformanceMetrics
from app.config.loader import get_config, override_config
from app.data.schema import MarketData, coerce_schema
from app.research.experiments import ExperimentStore
from app.research.harness import SegmentRun
from app.research.overfitting import Finding, OverfittingAssessment, Severity
from app.research.report import (
    VERDICT_INCONCLUSIVE,
    VERDICT_INSUFFICIENT,
    VERDICT_NEGATIVE,
    VERDICT_SURVIVED,
    ValidationReport,
)
from app.research.robustness import RobustnessReport
from app.research.splits import Segment, SegmentRole
from app.research.study import ValidationStudy


@pytest.fixture
def config(config_dir):
    return get_config(config_dir)


# ------------------------------------------------------------------ fixtures

def segment(name: str, role: SegmentRole) -> Segment:
    index = pd.date_range("2018-01-01", periods=800, freq="1D", tz="UTC")
    return Segment(
        name=name, role=role, start=300, end=700, data_start=100,
        start_time=index[300], end_time=index[699],
    )


def run(name: str, role: SegmentRole, **metrics) -> SegmentRun:
    return SegmentRun(
        segment=segment(name, role),
        variant="baseline",
        metrics=PerformanceMetrics(**metrics),
        trades=pd.DataFrame(),
        equity_curve=pd.Series(dtype="float64"),
        data_quality="PASS",
        data_quality_source="caller",
        result=None,  # type: ignore[arg-type]
    )


def report_with(**kwargs) -> ValidationReport:
    return ValidationReport(
        experiment_id="RES-test", symbol="XAUUSD", timeframe="1D", **kwargs
    )


# ------------------------------------------------------------------- verdicts

def test_no_out_of_sample_trades_is_insufficient_evidence():
    report = report_with(segments=[run("out_of_sample", SegmentRole.OUT_OF_SAMPLE)])
    verdict, reasons = report.verdict()

    assert verdict == VERDICT_INSUFFICIENT
    assert any("no trades" in r for r in reasons)


def test_a_thin_out_of_sample_sample_is_insufficient_evidence():
    report = report_with(
        segments=[
            run("out_of_sample", SegmentRole.OUT_OF_SAMPLE,
                total_trades=12, expectancy_r=0.5)
        ]
    )
    assert report.verdict()[0] == VERDICT_INSUFFICIENT


def test_a_negative_out_of_sample_expectancy_is_evidence_against():
    report = report_with(
        segments=[
            run("in_sample", SegmentRole.IN_SAMPLE, total_trades=90, expectancy_r=0.4),
            run("out_of_sample", SegmentRole.OUT_OF_SAMPLE,
                total_trades=60, expectancy_r=-0.2),
        ]
    )
    verdict, reasons = report.verdict()

    assert verdict == VERDICT_NEGATIVE
    assert any("lost money per unit of risk" in r for r in reasons)


def test_evidence_against_is_reported_as_a_successful_outcome():
    report = report_with(
        segments=[
            run("out_of_sample", SegmentRole.OUT_OF_SAMPLE,
                total_trades=60, expectancy_r=-0.2)
        ]
    )
    assert "successful research outcome" in report.render()


def test_a_severe_overfitting_finding_blocks_a_positive_verdict():
    report = report_with(
        segments=[
            run("out_of_sample", SegmentRole.OUT_OF_SAMPLE,
                total_trades=60, expectancy_r=0.3)
        ],
        overfitting=OverfittingAssessment(
            trials=200,
            findings=(Finding("HEAVY_SEARCH", Severity.SEVERE, "too much searching"),),
        ),
    )
    assert report.verdict()[0] == VERDICT_INCONCLUSIVE


def test_a_fragile_parameter_blocks_a_positive_verdict():
    robustness = RobustnessReport(segment="in_sample", objective_name="sortino")
    report = report_with(
        segments=[
            run("out_of_sample", SegmentRole.OUT_OF_SAMPLE,
                total_trades=60, expectancy_r=0.3)
        ],
        robustness=robustness,
    )
    # No fragile parameters yet: the verdict may pass.
    assert report.verdict()[0] == VERDICT_SURVIVED

    class Fragile(RobustnessReport):
        @property
        def fragile_parameters(self):
            return ("features.rsi_period",)

    report.robustness = Fragile(segment="in_sample", objective_name="sortino")
    assert report.verdict()[0] == VERDICT_INCONCLUSIVE


def test_the_best_available_verdict_is_still_hedged():
    """Surviving is the weakest positive result available, and says so."""
    report = report_with(
        segments=[
            run("out_of_sample", SegmentRole.OUT_OF_SAMPLE,
                total_trades=60, expectancy_r=0.3)
        ]
    )
    verdict, _ = report.verdict()
    text = report.render()

    assert verdict == VERDICT_SURVIVED
    assert "not a finding that it works" in text
    assert "weakest positive result" in text


@pytest.mark.parametrize("phrase", ["will make", "guaranteed", "expect to earn", "you will"])
def test_the_report_makes_no_forward_looking_claim(phrase):
    report = report_with(
        segments=[
            run("out_of_sample", SegmentRole.OUT_OF_SAMPLE,
                total_trades=60, expectancy_r=0.9, total_return=1.5)
        ]
    )
    assert phrase not in report.render().lower()


def test_profitability_is_only_ever_mentioned_to_disclaim_it():
    """The word may appear. A claim built from it may not."""
    report = report_with(
        segments=[
            run("out_of_sample", SegmentRole.OUT_OF_SAMPLE,
                total_trades=60, expectancy_r=0.9, total_return=1.5)
        ]
    )
    negations = ("no ", "not ", "never", "claims", "cannot")
    mentions = [
        line for line in report.render().lower().splitlines() if "profitab" in line
    ]

    assert mentions, "the report should state that it makes no profitability claim"
    for line in mentions:
        assert any(word in line for word in negations), f"unqualified claim: {line!r}"


def test_the_report_carries_its_caveats(config):
    report = report_with(segments=[run("out_of_sample", SegmentRole.OUT_OF_SAMPLE)])
    text = report.render()

    assert "not a forecast" in text or "forecasts anything" in text
    assert "hypothetical" in text
    assert "HOW TO READ THIS" in text


def test_every_required_section_appears():
    """The brief lists the sections; their absence would be silent."""
    report = report_with(
        segments=[
            run("in_sample", SegmentRole.IN_SAMPLE, total_trades=80, expectancy_r=0.4),
            run("validation", SegmentRole.VALIDATION, total_trades=30, expectancy_r=0.3),
            run("out_of_sample", SegmentRole.OUT_OF_SAMPLE,
                total_trades=60, expectancy_r=0.2),
        ]
    )
    text = report.render()

    for heading in ("IN-SAMPLE, VALIDATION AND OUT-OF-SAMPLE", "DRAWDOWN BEHAVIOUR"):
        assert heading in text


def test_the_report_serialises_and_saves(tmp_path):
    report = report_with(
        segments=[
            run("out_of_sample", SegmentRole.OUT_OF_SAMPLE,
                total_trades=60, expectancy_r=0.2)
        ]
    )
    payload = report.to_dict()
    assert payload["verdict"]
    assert payload["segments"]

    path = report.save(tmp_path)
    assert (path / "validation_report.txt").exists()
    assert (path / "validation_report.json").exists()


# ------------------------------------------------------------------ the study

def market(n: int = 900, seed: int = 11) -> MarketData:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2016-01-01", periods=n, freq="1D", tz="UTC")
    close = 500.0 + 0.6 * np.arange(n) + np.cumsum(rng.normal(0, 3.0, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    df = coerce_schema(
        pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) + 3.0,
                "low": np.minimum(open_, close) - 3.0,
                "close": close,
                "volume": np.full(n, 5_000.0),
            },
            index=index,
        )
    )
    return MarketData(symbol="XAUUSD", timeframe="1D", df=df, provider="synthetic")


def full(config):
    """Every analysis enabled, trimmed so it runs in a suite rather than a lunch break."""
    research = config.research
    return override_config(
        config,
        research=research.model_copy(
            update={
                "monte_carlo": research.monte_carlo.model_copy(update={"simulations": 50}),
                "robustness": research.robustness.model_copy(
                    update={"parameters": ["features.rsi_period"], "points": 3}
                ),
                "walk_forward": research.walk_forward.model_copy(update={"folds": 2}),
            }
        ),
    )


def minimal(config):
    """Only the three windows. For tests about identity and bookkeeping."""
    research = config.research
    return override_config(
        config,
        research=research.model_copy(
            update={
                "monte_carlo": research.monte_carlo.model_copy(update={"enabled": False}),
                "robustness": research.robustness.model_copy(update={"enabled": False}),
                "walk_forward": research.walk_forward.model_copy(update={"enabled": False}),
                "isolate_strategies": False,
            }
        ),
    )


@pytest.fixture(scope="module")
def full_study(request):
    """One complete study, shared by every assertion about its output.

    Running the whole pipeline is expensive - roughly twenty backtests - so it
    happens once and the assertions read the result rather than each paying for
    their own copy.
    """
    from app.config.loader import reset_config_cache

    tmp = request.getfixturevalue("tmp_path_factory").mktemp("study")
    config = get_config()
    store = ExperimentStore(tmp / "experiments.db")
    study = ValidationStudy(
        full(config), market(), config.assets.get("XAUUSD"),
        quality_status="PASS", store=store,
    )
    result = study.run()
    reset_config_cache()
    return result, store, study


@pytest.mark.slow
def test_the_study_produces_every_analysis(full_study):
    result, _, _ = full_study
    report = result.report

    assert {s.segment.name for s in report.segments} == {
        "in_sample", "validation", "out_of_sample"
    }
    assert report.walk_forward is not None
    assert report.monte_carlo is not None
    assert report.robustness is not None
    assert report.correlation is not None
    assert report.overfitting is not None
    assert report.verdict()[0]


@pytest.mark.slow
def test_the_out_of_sample_window_is_scored_last_and_once(full_study):
    """Every decision is made before the final window is touched."""
    result, _, _ = full_study
    report = result.report

    out_of_sample = report.out_of_sample
    assert out_of_sample is not None
    assert out_of_sample.variant == "baseline"
    # Only one row exists for it, so nothing was compared there.
    assert sum(1 for s in report.segments if s.segment.name == "out_of_sample") == 1
    assert out_of_sample.segment.start_time > report.validation.segment.end_time


@pytest.mark.slow
def test_the_whole_study_is_recorded_not_just_the_headline(full_study):
    result, store, study = full_study

    stored = store.get_experiment(result.experiment_id)
    assert stored["verdict"] == result.report.verdict()[0]
    assert stored["data_checksum"] == study.data_checksum
    assert len(store.segments_for(result.experiment_id)) == 3
    assert store.count_trials(study.data_checksum) >= 1
    assert store.metrics_for(result.experiment_id, "out_of_sample")


@pytest.mark.slow
def test_the_rendered_report_covers_the_required_ground(full_study):
    result, _, _ = full_study
    text = result.report.render()

    for heading in (
        "IN-SAMPLE, VALIDATION AND OUT-OF-SAMPLE",
        "WALK-FORWARD",
        "DRAWDOWN BEHAVIOUR",
        "MONTE CARLO",
        "PARAMETER SENSITIVITY",
        "STRATEGY CORRELATION",
        "OVERFITTING AND DATA-SNOOPING",
        "VERDICT",
    ):
        assert heading in text, f"the report is missing its {heading} section"


@pytest.mark.slow
def test_the_study_id_is_reproducible(config, tmp_path):
    """Running the same study twice must not look like independent confirmation."""
    store = ExperimentStore(tmp_path / "experiments.db")
    settings = minimal(config)

    first = ValidationStudy(
        settings, market(), config.assets.get("XAUUSD"), quality_status="PASS", store=store
    ).run()
    second = ValidationStudy(
        settings, market(), config.assets.get("XAUUSD"), quality_status="PASS", store=store
    ).run()

    assert first.experiment_id == second.experiment_id
    assert len(store.list_experiments()) == 1


@pytest.mark.slow
def test_a_configuration_change_produces_a_different_study(config, tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    settings = minimal(config)
    changed = override_config(
        settings, risk=settings.risk.model_copy(update={"risk_per_trade": 0.02})
    )

    a = ValidationStudy(settings, market(), config.assets.get("XAUUSD"),
                        quality_status="PASS", store=store).run()
    b = ValidationStudy(changed, market(), config.assets.get("XAUUSD"),
                        quality_status="PASS", store=store).run()

    assert a.experiment_id != b.experiment_id
    assert len(store.list_experiments()) == 2
    # Both configurations count toward the search against this data.
    assert store.count_trials(a.report.data_checksum) >= 2


@pytest.mark.slow
def test_the_search_is_counted_across_sessions(config, tmp_path):
    """Last week's variants still happened when the process restarts."""
    path = tmp_path / "experiments.db"
    settings = minimal(config)
    data = market()

    first = ValidationStudy(settings, data, config.assets.get("XAUUSD"),
                            quality_status="PASS", store=ExperimentStore(path)).run()

    changed = override_config(
        settings, risk=settings.risk.model_copy(update={"risk_per_trade": 0.015})
    )
    second = ValidationStudy(changed, data, config.assets.get("XAUUSD"),
                             quality_status="PASS", store=ExperimentStore(path)).run()

    assert second.trials > first.trials


@pytest.mark.slow
def test_a_history_too_short_for_a_study_fails_helpfully(config):
    from app.research.splits import SplitError

    study = ValidationStudy(
        minimal(config), market(n=250), config.assets.get("XAUUSD"), quality_status="PASS"
    )
    with pytest.raises(SplitError, match="longer period"):
        study.run()
