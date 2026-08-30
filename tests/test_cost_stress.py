"""Execution-cost stress testing.

Two layers. The survival judgement is exercised directly on constructed
scenario results, because the cases that matter - an edge that survives, an edge
that does not, no edge to begin with - are hard to provoke on demand from market
data. Then one slow test drives the real backtester and execution simulator to
prove the thing the whole feature rests on: that a scenario's cost overrides
reach the actual fills and change the numbers, rather than being computed in a
parallel calculation that the fills never see.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.backtest.metrics import PerformanceMetrics
from app.config.loader import get_config, override_config
from app.config.models import CostStressSettings
from app.data.schema import MarketData, coerce_schema
from app.research.cost_stress import (
    BASELINE,
    MODEST_ADVERSE,
    CostScenario,
    CostStressReport,
    ScenarioResult,
    default_scenarios,
    run_cost_stress_study,
)
from app.research.harness import ResearchHarness, SegmentRun
from app.research.report import (
    VERDICT_INCONCLUSIVE,
    VERDICT_SURVIVED,
    ValidationReport,
)
from app.research.robustness import set_parameter
from app.research.splits import Segment, SegmentRole, chronological_split


@pytest.fixture
def config(config_dir):
    return get_config(config_dir)


# ------------------------------------------------------------- report builders


def _result(name: str, *, expectancy_r: float, trades: int = 40,
            objective: float = 0.5, overrides=None, error=None) -> ScenarioResult:
    scenario = CostScenario(name, name, overrides or {})
    metrics = PerformanceMetrics(
        total_trades=trades, expectancy_r=expectancy_r,
        total_costs=100.0, cost_drag=0.02, total_return=expectancy_r * 0.1,
    )
    return ScenarioResult(scenario=scenario, objective=objective, metrics=metrics, error=error)


def _report(*results: ScenarioResult) -> CostStressReport:
    return CostStressReport(segment="in_sample", objective_name="sortino", results=list(results))


# -------------------------------------------------------------- the judgement


def test_no_baseline_edge_is_reported_as_such_not_as_survival():
    """A losing baseline has no edge to lose. That must not read as 'survived'."""
    report = _report(
        _result(BASELINE, expectancy_r=-0.3),
        _result(MODEST_ADVERSE, expectancy_r=-0.4, overrides={"execution.spread_multiplier": 1.5}),
    )
    status, detail = report.survival()

    assert status == "NO_BASELINE_EDGE"
    assert report.survives is False
    assert report.baseline_has_edge is False
    assert "no apparent edge" in detail
    # The distinction the whole module turns on: not-surviving-a-test is not the
    # same as never-having-an-edge, and the second must not be dressed as the first.
    assert "did not survive" not in detail.lower()


def test_an_edge_that_dies_under_modest_costs_is_flagged_honestly():
    report = _report(
        _result(BASELINE, expectancy_r=0.15),
        _result(MODEST_ADVERSE, expectancy_r=-0.02,
                overrides={"execution.spread_multiplier": 1.5}),
    )
    status, detail = report.survival()

    assert status == "DID_NOT_SURVIVE"
    assert report.survives is False
    assert "+0.150R" in detail and "-0.020R" in detail
    assert "artefact of the optimistic baseline" in detail


def test_an_edge_that_holds_survives_but_is_never_called_profitable():
    report = _report(
        _result(BASELINE, expectancy_r=0.20),
        _result(MODEST_ADVERSE, expectancy_r=0.11,
                overrides={"execution.spread_multiplier": 1.5}),
    )
    status, detail = report.survival()

    assert status == "SURVIVED"
    assert report.survives is True
    # Surviving cost stress is evidence of one thing only, and profitability is
    # not it. The wording must not let a reader promote it.
    assert "profitable" in detail.lower()  # only ever as a denial
    assert "not evidence" in detail.lower() or "not a profitability" in detail.lower()


def test_survival_is_not_assessed_when_the_baseline_errors():
    report = _report(
        _result(BASELINE, expectancy_r=0.0, trades=0, error="boom"),
        _result(MODEST_ADVERSE, expectancy_r=-0.1),
    )
    status, _ = report.survival()
    assert status == "NOT_ASSESSED"
    assert report.survives is False


def test_a_clean_run_with_too_few_trades_reports_metrics_not_an_error():
    """A NaN objective from the ranking guard must not read as a failed run.

    The ranking objective is undefined below ~20 trades, but expectancy, return
    and costs are perfectly real. Collapsing 'objective could not be scored' into
    'errored' would hide a valid degradation result on any thin window.
    """
    thin = ScenarioResult(
        scenario=CostScenario(BASELINE, "b"),
        objective=float("nan"),
        metrics=PerformanceMetrics(total_trades=8, expectancy_r=0.05, total_costs=20.0),
    )
    assert thin.errored is False
    assert thin.has_metrics is True
    assert thin.objective_finite is False

    frame = _report(thin).to_frame()
    assert frame.loc[0, "expectancy_r"] == pytest.approx(0.05)
    assert pd.isna(frame.loc[0, "objective"])


# --------------------------------------------------------------- scenario set


def test_default_scenarios_are_baseline_first_declared_and_adverse(config):
    settings = config.research.cost_stress
    scenarios = default_scenarios(config, settings)

    assert scenarios[0].name == BASELINE
    assert scenarios[0].is_baseline
    names = {s.name for s in scenarios}
    assert MODEST_ADVERSE in names
    assert any(s.name.startswith("spread_x") for s in scenarios)
    assert any(s.name.startswith("slippage_x") for s in scenarios)

    # The commission scenario sets an absolute pct, because the baseline charges
    # none and a multiple of zero would leave commissions untested.
    commission = next(s for s in scenarios if s.name == "commission_added")
    assert commission.overrides["execution.commission_pct"] == settings.commission_pct

    # The decisive scenario combines the *mildest* of each - the realistic case.
    combined = next(s for s in scenarios if s.name == MODEST_ADVERSE)
    assert combined.overrides["execution.spread_multiplier"] == pytest.approx(
        config.execution.spread_multiplier * min(settings.spread_multipliers)
    )


def test_every_scenario_override_targets_execution_only(config):
    for scenario in default_scenarios(config, config.research.cost_stress):
        for path in scenario.overrides:
            assert path.startswith("execution."), (
                f"{scenario.name} changes {path}, which is not an execution cost"
            )


def test_cost_multipliers_below_one_are_rejected():
    """A 'stress' that lowers costs is a search for a flattering assumption."""
    with pytest.raises(ValueError, match="must be >= 1.0"):
        CostStressSettings(spread_multipliers=[0.5, 1.0])
    with pytest.raises(ValueError, match="must be >= 1.0"):
        CostStressSettings(slippage_multipliers=[0.8])


def test_the_report_serialises_and_round_trips_through_json(config):
    report = _report(
        _result(BASELINE, expectancy_r=-0.3),
        _result("spread_x1.5", expectancy_r=-0.31,
                overrides={"execution.spread_multiplier": 1.5}),
        _result(MODEST_ADVERSE, expectancy_r=-0.35,
                overrides={"execution.spread_multiplier": 1.5}),
    )
    payload = report.to_dict()

    for key in ("segment", "survives", "survival_status", "survival_detail",
                "baseline_has_edge", "baseline_expectancy_r", "cost_drag_range", "scenarios"):
        assert key in payload
    assert len(payload["scenarios"]) == 3
    # Must survive a real serialisation, not just dict construction.
    restored = json.loads(json.dumps(payload, default=str))
    assert restored["survival_status"] == "NO_BASELINE_EDGE"


# --------------------------------------------------- the verdict gate

def _oos_segment_run(expectancy_r: float, trades: int = 60) -> SegmentRun:
    index = pd.date_range("2018-01-01", periods=800, freq="1D", tz="UTC")
    seg = Segment(
        name="out_of_sample", role=SegmentRole.OUT_OF_SAMPLE,
        start=300, end=700, data_start=100,
        start_time=index[300], end_time=index[699],
    )
    return SegmentRun(
        segment=seg, variant="baseline",
        metrics=PerformanceMetrics(total_trades=trades, expectancy_r=expectancy_r),
        trades=pd.DataFrame(), equity_curve=pd.Series(dtype="float64"),
        data_quality="PASS", data_quality_source="test", result=None,  # type: ignore[arg-type]
    )


def _would_survive_report(cost_stress: CostStressReport | None) -> ValidationReport:
    """A report that reaches SURVIVED on its own, so a downgrade is attributable."""
    return ValidationReport(
        experiment_id="RES-cost", symbol="XAUUSD", timeframe="1D",
        segments=[_oos_segment_run(0.30)],
        cost_stress=cost_stress,
    )


def test_a_cost_fragile_edge_downgrades_a_would_be_survived_verdict():
    # Sanity: with no cost stress attached, this report survives.
    assert _would_survive_report(None).verdict()[0] == VERDICT_SURVIVED

    fragile = _report(
        _result(BASELINE, expectancy_r=0.15),
        _result(MODEST_ADVERSE, expectancy_r=-0.03,
                overrides={"execution.spread_multiplier": 1.5}),
    )
    verdict, reasons = _would_survive_report(fragile).verdict()
    assert verdict == VERDICT_INCONCLUSIVE
    assert any("realistic" in r.lower() for r in reasons)


def test_no_baseline_edge_does_not_downgrade_a_positive_out_of_sample_verdict():
    """NO_BASELINE_EDGE is not a cost-fragility signal and must not gate the verdict.

    A configuration can lose on the in-sample window yet be positive out of
    sample; that is unusual but it is the out-of-sample result that carries
    weight, and 'no in-sample edge' is not evidence the out-of-sample one is an
    artefact of costs.
    """
    no_edge = _report(
        _result(BASELINE, expectancy_r=-0.10),
        _result(MODEST_ADVERSE, expectancy_r=-0.20,
                overrides={"execution.spread_multiplier": 1.5}),
    )
    assert _would_survive_report(no_edge).verdict()[0] == VERDICT_SURVIVED


def test_a_surviving_cost_stress_leaves_the_verdict_alone():
    survived = _report(
        _result(BASELINE, expectancy_r=0.20),
        _result(MODEST_ADVERSE, expectancy_r=0.12,
                overrides={"execution.spread_multiplier": 1.5}),
    )
    assert _would_survive_report(survived).verdict()[0] == VERDICT_SURVIVED


# ------------------------------------------------------- the dashboard view

def test_the_dashboard_panel_surfaces_cost_stress_without_fabrication():
    """The research panel must show the cost result when present, and only then."""
    from dashboard.view import research_panel

    report = _report(
        _result(BASELINE, expectancy_r=-0.35),
        _result(MODEST_ADVERSE, expectancy_r=-0.38,
                overrides={"execution.spread_multiplier": 1.5}),
    )
    ctx = {"availability": "AVAILABLE", "report": {
        "evidence_summary": [], "segments": [], "cost_stress": report.to_dict()}}
    panel = research_panel({"research": ctx})["cost_stress"]

    assert panel is not None
    assert panel["survival_status"] == "NO_BASELINE_EDGE"
    assert panel["tone"] == "warn"          # not green - no edge held
    assert panel["baseline_display"] == "-0.350R"
    assert [r["scenario"] for r in panel["rows"]] == [BASELINE, MODEST_ADVERSE]

    # A survived result is the only one that may read green.
    survived = _report(
        _result(BASELINE, expectancy_r=0.2),
        _result(MODEST_ADVERSE, expectancy_r=0.1,
                overrides={"execution.spread_multiplier": 1.5}),
    )
    ctx["report"]["cost_stress"] = survived.to_dict()
    assert research_panel({"research": ctx})["cost_stress"]["tone"] == "ok"


def test_the_dashboard_panel_shows_nothing_when_there_is_no_cost_study():
    """No cost study must render as absent, never as a table of zeros."""
    from dashboard.view import research_panel

    ctx = {"availability": "AVAILABLE",
           "report": {"evidence_summary": [], "segments": []}}
    assert research_panel({"research": ctx})["cost_stress"] is None


# --------------------------------------------------- the real execution path


def _market(n: int = 900, seed: int = 11) -> MarketData:
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


def test_a_scenario_actually_changes_the_execution_configuration(config):
    """The override must reach the configuration the harness will run.

    Fast, and deliberately separate from the backtest: it proves the assumption
    changes at all, so that the slow test below can attribute a change in the
    numbers to a change that definitely happened.
    """
    scenario = CostScenario(
        "wider", "wider spread",
        {"execution.spread_multiplier": config.execution.spread_multiplier * 2},
    )
    changed = config
    for path, value in scenario.overrides.items():
        changed = set_parameter(changed, path, value)

    assert changed.execution.spread_multiplier == config.execution.spread_multiplier * 2
    assert config.execution.spread_multiplier != changed.execution.spread_multiplier, (
        "the baseline configuration must not be mutated"
    )


@pytest.mark.slow
def test_adverse_costs_reach_the_fills_and_worsen_the_result(config):
    """The load-bearing test: cost overrides must change *real* backtest output.

    Runs the baseline and the adverse scenarios through the actual harness -
    same backtester, same ExecutionSimulator a study uses. If the overrides did
    not reach the fills, total costs would be identical across scenarios and this
    would fail.
    """
    cfg = override_config(config, platform=config.platform.model_copy(update={"trading_enabled": True}))
    data = _market()
    harness = ResearchHarness(
        cfg, data, cfg.assets.get("XAUUSD"), trading_enabled=True, quality_status="PASS"
    )
    warmup = harness.required_warmup()
    segments = {
        s.name: s
        for s in chronological_split(
            data.df.index, fractions=(0.5, 0.2, 0.3), warmup_bars=warmup, embargo_bars=5
        )
    }
    scenarios = default_scenarios(cfg, cfg.research.cost_stress)
    report = run_cost_stress_study(harness, segments["in_sample"], scenarios)

    baseline = report.result(BASELINE)
    assert baseline is not None and baseline.has_metrics, "the baseline produced no trades"

    combined = report.result(MODEST_ADVERSE)
    assert combined is not None and combined.has_metrics

    # Same configuration, same window: the only thing that changed is the cost
    # assumption, so identical trade counts but strictly higher costs is the
    # signature of overrides that actually reached the fills.
    assert combined.metrics.total_trades == baseline.metrics.total_trades
    assert combined.metrics.total_costs > baseline.metrics.total_costs, (
        "adverse costs did not increase realised costs - the override never reached the fills"
    )
    assert combined.metrics.cost_drag > baseline.metrics.cost_drag
    # More cost, same gross: net expectancy can only move down or stay.
    assert combined.metrics.expectancy_r <= baseline.metrics.expectancy_r + 1e-9

    # A single-factor scenario must also bite through the real path.
    spread2 = report.result("spread_x2")
    assert spread2 is not None and spread2.has_metrics
    assert spread2.metrics.total_costs > baseline.metrics.total_costs


@pytest.mark.slow
def test_the_survival_verdict_is_consistent_with_the_numbers(config):
    """Whatever the verdict on real output, it must follow from the metrics.

    The earlier version of this test hard-coded 'no edge' on synthetic data.
    That was wrong: the study fixture is a strongly trending series, on which a
    trend strategy does show an in-sample edge - which is precisely why
    in-sample results are description, not evidence, and why the real 'no edge'
    conclusion belongs to the out-of-sample and holdout windows, not here. So
    this asserts the fixture-independent invariant instead: the status the
    report reaches is the status its own numbers imply, and it never promotes
    surviving cost stress into a profitability claim.
    """
    cfg = override_config(config, platform=config.platform.model_copy(update={"trading_enabled": True}))
    data = _market()
    harness = ResearchHarness(
        cfg, data, cfg.assets.get("XAUUSD"), trading_enabled=True, quality_status="PASS"
    )
    warmup = harness.required_warmup()
    segments = {
        s.name: s
        for s in chronological_split(
            data.df.index, fractions=(0.5, 0.2, 0.3), warmup_bars=warmup, embargo_bars=5
        )
    }
    report = run_cost_stress_study(
        harness, segments["in_sample"], default_scenarios(cfg, cfg.research.cost_stress)
    )
    status, detail = report.survival()
    baseline = report.result(BASELINE)
    decisive = report.result(MODEST_ADVERSE)

    if status == "NO_BASELINE_EDGE":
        assert baseline.metrics.expectancy_r <= 0
        assert report.survives is False
    elif status == "SURVIVED":
        assert baseline.metrics.expectancy_r > 0
        assert decisive.has_metrics and decisive.metrics.expectancy_r > 0
        assert report.survives is True
    elif status == "DID_NOT_SURVIVE":
        assert baseline.metrics.expectancy_r > 0
        assert decisive.has_metrics and decisive.metrics.expectancy_r <= 0
        assert report.survives is False
    else:
        pytest.fail(f"unexpected status {status}")

    # No path may present surviving cost stress as evidence of profitability.
    lowered = detail.lower()
    if "profitable" in lowered or "profit " in lowered:
        assert "not " in lowered, f"survival detail implied profitability: {detail}"
