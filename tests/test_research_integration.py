"""Research context on the paper-trading dashboard.

The dashboard runs a configuration; a validation study may or may not exist for
it. What this file protects is mostly the *negative* space — what the page must
not claim when there is nothing to claim.

The failure this guards against is subtle and would never raise. A research
panel that rendered zeros for a study that was never run reads as "tested, found
nothing". That is a far stronger statement than "never tested", it is false, and
nothing about a table of zeros reveals which of the two it means. So the tests
below assert that an unavailable context produces an *empty* panel carrying a
reason, not a populated one carrying defaults.

The second concern is that in-sample and out-of-sample numbers stay
distinguishable all the way to the page. They are the same arithmetic on
different bars; only the label separates them, and a panel that lost the label
would be presenting development-window results as evidence.
"""

from __future__ import annotations

import json

import pytest

from app.config.loader import get_config, override_config
from app.paper_trading.engine import PaperTradingEngine
from app.research.context import (
    EVIDENCE_LADDER,
    ResearchAvailability,
    find_validation_reports,
    load_research_context,
)
from dashboard.view import build_view, research_panel

# --------------------------------------------------------------------- fixtures


def _report_payload(
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "1D",
    experiment_id: str = "RES-abc123",
    oos_trades: int = 7,
    verdict: str = "INSUFFICIENT EVIDENCE",
    fingerprint: str = "cfg-aaaa",
) -> dict:
    """A payload shaped exactly like ValidationReport.to_dict() produces."""
    return {
        "experiment_id": experiment_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "period_start": "2012-01-01",
        "period_end": "2024-12-31",
        "bars": 3200,
        "data_provider": "yahoo",
        "data_checksum": "sum123",
        "data_quality": "WARNING across the series; each window graded separately",
        "git_revision": "abcdef1",
        "random_seed": 42,
        "created_at": "2026-08-29T00:00:00+00:00",
        "spec": {"objective": "sortino", "config_fingerprint": fingerprint},
        "verdict": verdict,
        "verdict_reasons": [
            f"The out-of-sample window contains {oos_trades} trades. Below roughly 30 the "
            "statistics are not stable enough to support any conclusion."
        ],
        "segments": [
            {
                "segment": "in_sample", "role": "TRAIN", "bars": 1600, "trades": 40,
                "total_return": 0.21, "expectancy_r": 0.35, "max_drawdown": 0.09,
                "sharpe": 1.4, "win_rate": 0.55, "data_quality": "WARNING",
            },
            {
                "segment": "validation", "role": "VALIDATION", "bars": 640, "trades": 12,
                "total_return": 0.03, "expectancy_r": 0.10, "max_drawdown": 0.06,
                "sharpe": 0.4, "win_rate": 0.50, "data_quality": "WARNING",
            },
            {
                "segment": "out_of_sample", "role": "TEST", "bars": 960,
                "trades": oos_trades, "total_return": -0.05, "expectancy_r": -0.08,
                "max_drawdown": 0.14, "sharpe": -0.3, "win_rate": 0.43,
                "data_quality": "WARNING",
            },
        ],
        "walk_forward": {
            "folds": 4, "profitable_folds": 1, "efficiency": 0.22,
            "stitched": {"total_trades": 24}, "warnings": ["thin folds"],
        },
        "monte_carlo": {"simulations": 1000},
        "robustness": {
            "segment": "in_sample",
            "objective_name": "sortino",
            "fragile_parameters": ["risk.risk_per_trade"],
            "sensitivities": [
                {"parameter": "risk.risk_per_trade", "fragile": True,
                 "baseline_is_peak": True, "coefficient_of_variation": 1.8},
                {"parameter": "features.rsi_period", "fragile": False,
                 "baseline_is_peak": False, "coefficient_of_variation": 0.12},
            ],
        },
        "correlation": {
            "segment": "in_sample", "threshold": 0.7,
            "trades_per_strategy": {"momentum": 40, "trend_following": 90},
            "findings": [
                {"strategy_a": "momentum", "strategy_b": "trend_following",
                 "return_correlation": 0.91, "signal_agreement": 0.88,
                 "observations": 500, "hypothesis": "momentum may be redundant"}
            ],
            "notes": [],
        },
        "overfitting": {
            "trials": 57,
            "deflated_sharpe": 0.02,
            "findings": [
                {"code": "thin_oos", "severity": "SEVERE",
                 "message": "Too few out-of-sample trades to conclude.", "evidence": {}},
                {"code": "search_intensity", "severity": "WARNING",
                 "message": "57 configurations were tried.", "evidence": {}},
            ],
        },
        "variants": [],
        "variant_summary": "",
        "regime_performance": [],
        "recommendations": {
            "experiment_id": experiment_id,
            "applicable_count": 0,
            "notes": ["No strategy is disabled on the strength of a correlation."],
            "recommendations": [
                {
                    "subject": "momentum / trend_following redundancy",
                    "action": "NO_ACTION",
                    "evidence_tier": "VALIDATION_WINDOW",
                    "strategy": "momentum",
                    "proposed_weight": None,
                    "rationale": ["One window is not enough to move a weight."],
                    "blockers": ["The only comparison available is a single window."],
                    "evidence": {},
                    "applicable": False,
                }
            ],
        },
        "notes": [],
    }


@pytest.fixture
def reports_root(tmp_path):
    """A reports/validation directory the loader will search."""
    root = tmp_path / "validation"
    root.mkdir(parents=True)
    return root


def _write(root, payload: dict) -> None:
    directory = root / payload["experiment_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "validation_report.json").write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------- unavailable is not empty-zero


def test_no_study_reports_missing_with_a_reason_and_no_numbers(reports_root):
    context = load_research_context("XAUUSD", "1D", directory=reports_root)

    assert context.availability is ResearchAvailability.MISSING
    assert not context.is_available
    assert "No validation study has been run" in context.reason
    assert "run_research.py" in context.reason, "the reason should say how to fix it"
    assert context.report == {}
    assert context.verdict is None
    assert context.segment("out_of_sample") is None


def test_a_study_of_another_instrument_is_not_offered_as_this_one(reports_root):
    _write(reports_root, _report_payload(symbol="EURUSD"))
    context = load_research_context("XAUUSD", "1D", directory=reports_root)

    assert context.availability is ResearchAvailability.MISSING
    assert "none for XAUUSD 1D" in context.reason
    assert "says nothing about this one" in context.reason
    assert context.report == {}
    assert context.other_experiments, "the other study should still be visible as provenance"


def test_a_study_of_another_configuration_is_marked_stale(reports_root):
    _write(reports_root, _report_payload(fingerprint="cfg-old"))
    context = load_research_context(
        "XAUUSD", "1D", directory=reports_root, config_fingerprint="cfg-new"
    )

    assert context.availability is ResearchAvailability.STALE
    assert "cfg-old" in context.reason and "cfg-new" in context.reason
    assert "describe a different system" in context.reason
    assert context.report == {}, "a stale study must not supply numbers"


def test_a_matching_fingerprint_is_available(reports_root):
    _write(reports_root, _report_payload(fingerprint="cfg-same"))
    context = load_research_context(
        "XAUUSD", "1D", directory=reports_root, config_fingerprint="cfg-same"
    )
    assert context.availability is ResearchAvailability.AVAILABLE
    assert context.report["experiment_id"] == "RES-abc123"


def test_an_unparseable_report_is_reported_as_unreadable_not_missing(reports_root):
    directory = reports_root / "RES-broken"
    directory.mkdir()
    (directory / "validation_report.json").write_text('{"truncated', encoding="utf-8")

    context = load_research_context("XAUUSD", "1D", directory=reports_root)
    assert context.availability is ResearchAvailability.UNREADABLE
    assert "none could be parsed" in context.reason
    assert context.report == {}


def test_the_newest_matching_study_wins(reports_root):
    import os
    import time

    _write(reports_root, _report_payload(experiment_id="RES-older"))
    time.sleep(0.01)
    _write(reports_root, _report_payload(experiment_id="RES-newer"))
    newer = reports_root / "RES-newer" / "validation_report.json"
    os.utime(newer, (time.time() + 10, time.time() + 10))

    context = load_research_context("XAUUSD", "1D", directory=reports_root)
    assert context.experiment_id == "RES-newer"
    assert [e["experiment_id"] for e in context.other_experiments] == ["RES-older"]


def test_find_validation_reports_handles_an_absent_directory(tmp_path):
    assert find_validation_reports(tmp_path / "nope") == []


# ----------------------------------------------- the panel, when nothing is there


def test_the_panel_is_blank_and_says_why_when_no_study_exists():
    """Not a table of zeros. An empty panel with a reason."""
    panel = research_panel(
        {"research": {"availability": "MISSING", "reason": "No validation study has been run."}}
    )

    assert panel["available"] is False
    assert panel["availability"] == "MISSING"
    assert "No validation study" in panel["reason"]
    assert panel["segments"] == []
    assert panel["verdict"] is None
    assert panel["walk_forward"] is None
    assert panel["robustness"] is None
    assert panel["correlation"] is None
    assert panel["overfitting"] is None
    assert panel["recommendations"] == []
    assert panel["metrics"] == [], (
        "an unavailable study must not produce metric tiles - a tile reading 0 is "
        "indistinguishable from a measured zero"
    )


def test_a_missing_panel_never_reports_a_verdict():
    for availability in ("MISSING", "STALE", "UNREADABLE"):
        panel = research_panel({"research": {"availability": availability, "reason": "x"}})
        assert panel["verdict"] is None
        assert panel["verdict_reasons"] == []
        assert panel["available"] is False


def test_an_absent_research_key_is_treated_as_missing():
    """A payload with no research at all must not raise or fabricate."""
    panel = research_panel({})
    assert panel["available"] is False
    assert panel["availability"] == "MISSING"
    assert panel["metrics"] == []


# ------------------------------------------- IS / OOS cannot be silently confused


def test_windows_stay_labelled_with_what_each_supports(reports_root):
    _write(reports_root, _report_payload())
    context = load_research_context("XAUUSD", "1D", directory=reports_root)
    panel = research_panel({"research": context.to_dict()})

    by_name = {s["segment"]: s for s in panel["segments"]}
    assert set(by_name) == {"in_sample", "validation", "out_of_sample"}

    assert by_name["in_sample"]["is_out_of_sample"] is False
    assert "no evidence" in by_name["in_sample"]["supports"]
    assert by_name["validation"]["is_out_of_sample"] is False
    assert "weak" in by_name["validation"]["supports"]
    assert by_name["out_of_sample"]["is_out_of_sample"] is True
    assert "the evidence that counts" in by_name["out_of_sample"]["supports"]


def test_the_headline_metrics_come_from_the_out_of_sample_window_only(reports_root):
    """The in-sample window looks far better; it must not be what is headlined."""
    _write(reports_root, _report_payload(oos_trades=7))
    context = load_research_context("XAUUSD", "1D", directory=reports_root)
    panel = research_panel({"research": context.to_dict()})
    metrics = {m["label"]: m for m in panel["metrics"]}

    assert metrics["OOS trades"]["value"] == 7
    assert metrics["OOS expectancy"]["value"] == pytest.approx(-0.08), (
        "the headline expectancy must be the out-of-sample one (-0.08), not the "
        "in-sample one (+0.35)"
    )
    assert metrics["OOS expectancy"]["tone"] == "error"
    assert metrics["OOS trades"]["tone"] == "warn", "7 trades is below the threshold"


def test_a_thin_out_of_sample_window_is_flagged_not_smoothed(reports_root):
    _write(reports_root, _report_payload(oos_trades=7, verdict="INSUFFICIENT EVIDENCE"))
    panel = research_panel(
        {"research": load_research_context("XAUUSD", "1D", directory=reports_root).to_dict()}
    )

    assert panel["verdict"] == "INSUFFICIENT EVIDENCE"
    assert panel["verdict_tone"] == "warn"
    assert any("not stable enough" in r for r in panel["verdict_reasons"])


def test_no_verdict_value_means_profitable():
    """Asserted against the report's own vocabulary, not just this panel."""
    from app.research import report as report_module

    verdicts = {
        report_module.VERDICT_INSUFFICIENT,
        report_module.VERDICT_NEGATIVE,
        report_module.VERDICT_INCONCLUSIVE,
        report_module.VERDICT_SURVIVED,
    }
    joined = " ".join(verdicts).lower()
    for word in ("profit", "win", "edge", "works", "recommended"):
        assert word not in joined, f"a verdict implies {word!r}"


# ---------------------------------------------------------- findings are surfaced


def test_every_research_section_reaches_the_panel(reports_root):
    _write(reports_root, _report_payload())
    panel = research_panel(
        {"research": load_research_context("XAUUSD", "1D", directory=reports_root).to_dict()}
    )

    assert panel["walk_forward"]["folds"] == 4
    assert panel["walk_forward"]["profitable_display"] == "1 of 4"
    assert panel["robustness"]["fragile_parameters"] == ["risk.risk_per_trade"]
    assert len(panel["robustness"]["rows"]) == 2
    assert panel["correlation"]["findings"][0]["pair"] == "momentum / trend_following"
    assert panel["overfitting"]["severe_count"] == 1
    assert panel["overfitting"]["trials"] == 57
    assert len(panel["recommendations"]) == 1


def test_fragile_parameters_are_toned_as_a_problem(reports_root):
    _write(reports_root, _report_payload())
    panel = research_panel(
        {"research": load_research_context("XAUUSD", "1D", directory=reports_root).to_dict()}
    )
    rows = {r["parameter"]: r for r in panel["robustness"]["rows"]}

    assert rows["risk.risk_per_trade"]["verdict"] == "FRAGILE"
    assert rows["risk.risk_per_trade"]["tone"] == "error"
    assert rows["features.rsi_period"]["verdict"] == "stable"
    assert rows["features.rsi_period"]["tone"] == "ok"

    fragile_metric = next(
        m for m in panel["metrics"] if m["label"] == "Fragile parameters"
    )
    assert fragile_metric["tone"] == "error"


def test_correlation_findings_are_shown_without_any_disable_action(reports_root):
    _write(reports_root, _report_payload())
    panel = research_panel(
        {"research": load_research_context("XAUUSD", "1D", directory=reports_root).to_dict()}
    )

    assert panel["correlation"]["findings"], "the redundancy finding must be visible"
    actions = {r["action"] for r in panel["recommendations"]}
    assert "DISABLE" not in actions
    assert all(not r["applicable"] for r in panel["recommendations"])
    assert "ships false" in panel["recommendation_note"]


def test_provenance_survives_to_the_panel(reports_root):
    _write(reports_root, _report_payload())
    panel = research_panel(
        {"research": load_research_context("XAUUSD", "1D", directory=reports_root).to_dict()}
    )
    provenance = panel["provenance"]

    assert provenance["experiment_id"] == "RES-abc123"
    assert provenance["data_checksum"] == "sum123"
    assert provenance["git_revision"] == "abcdef1"
    assert provenance["random_seed"] == 42
    assert provenance["config_fingerprint"] == "cfg-aaaa"
    assert provenance["data_quality"].startswith("WARNING")
    assert panel["report_path"] is not None


def test_the_evidence_ladder_separates_code_passing_from_research_validity():
    rungs = {r["rung"] for r in EVIDENCE_LADDER}
    assert "Code and tests" in rungs

    code_rung = next(r for r in EVIDENCE_LADDER if r["rung"] == "Code and tests")
    assert "edge" in code_rung["does_not_support"]
    assert "losing strategy" in code_rung["does_not_support"]

    for rung in EVIDENCE_LADDER:
        assert rung["does_not_support"], f"{rung['rung']} claims no limits"


# --------------------------------------------- the engine surfaces it end to end


def _engine(config_dir, **kwargs):
    cfg = get_config(config_dir)
    cfg = override_config(cfg, platform=cfg.platform.model_copy(update={"trading_enabled": True}))
    return PaperTradingEngine(config=cfg, symbol="XAUUSD", timeframe="1D", **kwargs)


def test_dashboard_data_carries_the_research_context(config_dir):
    """With no study on disk, the payload says MISSING rather than omitting it."""
    data = _engine(config_dir).dashboard_data()

    assert "research" in data
    assert data["research"]["availability"] == "MISSING"
    assert "No validation study has been run" in data["research"]["reason"]

    view = build_view(data)
    assert view["research"]["available"] is False
    assert view["research"]["metrics"] == []


def test_the_engine_reads_a_real_report_from_the_reports_directory(config_dir, monkeypatch, tmp_path):
    """End to end: a report on disk reaches the rendered view model."""
    reports = tmp_path / "reports"
    monkeypatch.setenv("GTP_REPORTS_DIR", str(reports))
    _write(reports / "validation", _report_payload())

    view = build_view(_engine(config_dir).dashboard_data())

    assert view["research"]["available"] is True
    assert view["research"]["experiment_id"] == "RES-abc123"
    assert view["research"]["verdict"] == "INSUFFICIENT EVIDENCE"
    oos = next(s for s in view["research"]["segments"] if s["is_out_of_sample"])
    assert oos["trades"] == 7


def test_research_failure_does_not_break_the_dashboard(config_dir, monkeypatch):
    """A broken research load must degrade to 'unavailable', not take the page down."""
    def boom(*args, **kwargs):
        raise RuntimeError("reports volume unmounted")

    monkeypatch.setattr("app.paper_trading.engine.load_research_context", boom)
    engine = _engine(config_dir)
    data = engine.dashboard_data()

    assert data["research"]["availability"] == "UNREADABLE"
    assert "reports volume unmounted" in data["research"]["reason"]
    assert any(e["node"] == "research" for e in engine.state["recent_errors"])
    assert build_view(data)["research"]["available"] is False


def test_data_quality_stays_visible_alongside_research(config_dir, monkeypatch, tmp_path):
    """The quality grade must not be displaced by the research panel.

    Both describe the same run and both have to remain legible: a study graded
    on WARNING data is a study with a caveat, and the caveat belongs next to it.
    """
    reports = tmp_path / "reports"
    monkeypatch.setenv("GTP_REPORTS_DIR", str(reports))
    _write(reports / "validation", _report_payload())

    view = build_view(_engine(config_dir).dashboard_data())

    assert view["research"]["provenance"]["data_quality"].startswith("WARNING")
    for segment in view["research"]["segments"]:
        assert segment["data_quality"] == "WARNING"
        assert segment["quality_tone"] == "warn"

    # The live market panel's own quality reporting is untouched.
    assert "quality_findings" in view
    assert any(m["label"] == "Quality" for m in view["market"]["metrics"])
