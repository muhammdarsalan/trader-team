"""The Streamlit page must actually render.

Every other dashboard test asserts the *view model* - the numbers and statuses
:mod:`dashboard.view` produces - because that is where the logic lives and it
needs no rendering runtime. These tests cover the remaining risk: that
``streamlit run dashboard/streamlit_app.py`` raises before drawing anything.

That failure mode is real and cheap to miss. A widget argument that a newer
Streamlit removed, a column count that no longer matches the metrics handed to
it, a dataframe built from a key the payload stopped carrying - none of these
show up in a view-model test, and all of them produce a page that is blank
except for a traceback.

``AppTest`` runs the script in-process with no browser and no server, so this
stays a normal offline test. It is marked ``slow`` because it executes the whole
page, including the plotly figure, on every run.
"""

from __future__ import annotations

import json

import pytest

from app.config.loader import get_config, override_config
from app.data.loaders.synthetic import SyntheticProvider
from app.paper_trading.engine import PaperTradingEngine
from app.utils.paths import PROJECT_ROOT, paper_state_path

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

#: Absolute: AppTest resolves a relative path against the *calling* file, which
#: would look for the page under tests/.
APP = str(PROJECT_ROOT / "dashboard" / "streamlit_app.py")

pytestmark = pytest.mark.slow


def _seed_session(config_dir, *, symbol="XAUUSD", bars_end="2017-12-31", quality="PASS"):
    """Write a real paper-trading state file the page will then read.

    The page is deliberately not handed a fabricated payload: it loads the same
    state file the runner writes, so a mismatch between what the engine persists
    and what the page reads is exactly the kind of thing this should catch.
    """
    cfg = get_config(config_dir)
    cfg = override_config(cfg, platform=cfg.platform.model_copy(update={"trading_enabled": True}))
    path = paper_state_path(symbol, "1D")
    engine = PaperTradingEngine(
        config=cfg, symbol=symbol, timeframe="1D", state_path=path
    )
    data = SyntheticProvider().get_historical_data(symbol, "1D", "2015-01-01", bars_end)
    engine.catch_up(data, data_quality=quality)
    engine.save_state()
    return engine


def _run(symbol="XAUUSD"):
    app = AppTest.from_file(APP, default_timeout=300)
    app.session_state["symbol"] = symbol
    app.session_state["timeframe"] = "1D"
    app.run()
    return app


def _assert_no_exception(app) -> None:
    assert not app.exception, "\n".join(str(e.value) for e in app.exception)


def test_page_renders_with_a_live_session(config_dir, monkeypatch):
    monkeypatch.setenv("GTP_CONFIG_DIR", str(config_dir))
    engine = _seed_session(config_dir)
    assert engine.portfolio.closed_trades, "the fixture produced no trades to render"

    app = _run()
    _assert_no_exception(app)

    assert app.title[0].value.startswith("XAUUSD 1D")
    # Market & regime, Strategies & decision, Portfolio, Execution, Performance,
    # Research, Graph, Activity.
    assert len(app.tabs) == 8
    assert len(app.metric) > 20, "the page drew almost no metrics"
    assert app.dataframe, "the page drew no tables"

    labels = {m.label for m in app.metric}
    for required in ("System", "Data freshness", "Data quality", "Equity", "Drawdown"):
        assert any(required in label for label in labels), f"no {required} metric"


def test_page_renders_with_no_session_and_says_so(config_dir, monkeypatch):
    """With no state file the page must render and report the absence.

    A blank page and a page reporting NO_DATA are very different things to an
    operator, and only one of them is honest.
    """
    monkeypatch.setenv("GTP_CONFIG_DIR", str(config_dir))
    assert not paper_state_path("XAUUSD", "1D").exists()

    app = _run()
    _assert_no_exception(app)

    text = " ".join(
        [w.value for w in app.warning] + [i.value for i in app.info]
    )
    assert "No market data is loaded" in text
    assert "No state file yet" in text
    system = next(m for m in app.metric if "System" in m.label)
    assert system.value == "IDLE", f"an empty session reported {system.value}"


def test_page_renders_a_degraded_session_without_hiding_it(config_dir, monkeypatch):
    """A FAIL-graded feed must reach the page as an error, not be smoothed away."""
    monkeypatch.setenv("GTP_CONFIG_DIR", str(config_dir))
    _seed_session(config_dir, quality="FAIL")

    app = _run()
    _assert_no_exception(app)

    banners = " ".join(e.value for e in app.error)
    assert "Data quality FAIL" in banners
    system = next(m for m in app.metric if "System" in m.label)
    assert system.value == "DEGRADED"
    quality = next(m for m in app.metric if "Data quality" in m.label)
    assert quality.value == "FAIL"


def test_page_renders_for_a_proxy_instrument_with_its_caveat(config_dir, monkeypatch):
    monkeypatch.setenv("GTP_CONFIG_DIR", str(config_dir))
    _seed_session(config_dir, symbol="XAUUSD")

    app = _run("XAUUSD")
    _assert_no_exception(app)

    warnings = " ".join(w.value for w in app.warning)
    assert "proxy series" in warnings
    assert "GC=F" in warnings


def _write_report(reports_root, payload: dict) -> None:
    directory = reports_root / "validation" / payload["experiment_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "validation_report.json").write_text(json.dumps(payload), encoding="utf-8")


def test_the_research_tab_renders_a_real_report(config_dir, monkeypatch, tmp_path):
    """The Research tab must draw, including its mixed-type provenance table.

    That table caused a silent failure once: it mixes ints (bars, seed) with
    strings (checksums, dates) in one column, Arrow refused the conversion, and
    the table vanished from the page without raising anything AppTest would
    surface as an exception.
    """
    monkeypatch.setenv("GTP_CONFIG_DIR", str(config_dir))
    reports = tmp_path / "reports"
    monkeypatch.setenv("GTP_REPORTS_DIR", str(reports))
    _write_report(
        reports,
        {
            "experiment_id": "RES-render01",
            "symbol": "XAUUSD",
            "timeframe": "1D",
            "period_start": "2018-01-01",
            "period_end": "2022-12-31",
            "bars": 1300,
            "data_provider": "synthetic",
            "data_checksum": "sum",
            "data_quality": "WARNING across the series",
            "git_revision": "abc1234",
            "random_seed": 42,
            "created_at": "2026-08-29T00:00:00+00:00",
            "spec": {"objective": "sortino", "config_fingerprint": "fp1"},
            "verdict": "INSUFFICIENT EVIDENCE",
            "verdict_reasons": ["13 out-of-sample trades is too few to conclude."],
            "segments": [
                {"segment": "in_sample", "role": "TRAIN", "bars": 650, "trades": 20,
                 "total_return": -0.05, "expectancy_r": -0.234, "max_drawdown": 0.11,
                 "sharpe": -0.2, "win_rate": 0.4, "data_quality": "WARNING"},
                {"segment": "out_of_sample", "role": "TEST", "bars": 390, "trades": 13,
                 "total_return": -0.08, "expectancy_r": -0.447, "max_drawdown": 0.13,
                 "sharpe": -0.5, "win_rate": 0.31, "data_quality": "WARNING"},
            ],
            "walk_forward": {"folds": 4, "profitable_folds": 0, "efficiency": 0.1,
                             "stitched": {}, "warnings": []},
            "monte_carlo": {"simulations": 1000},
            "robustness": {"segment": "in_sample", "objective_name": "sortino",
                           "fragile_parameters": [], "sensitivities": []},
            "correlation": {"segment": "in_sample", "threshold": 0.7,
                            "trades_per_strategy": {}, "findings": [], "notes": []},
            "overfitting": {"trials": 21, "deflated_sharpe": 0.0, "findings": [
                {"code": "thin", "severity": "SEVERE", "message": "Too few trades.",
                 "evidence": {}}]},
            "evidence_summary": [
                {"category": "Code and tests", "status": "SEPARATE QUESTION",
                 "detail": "d", "limitation": "l"},
            ],
            "recommendations": {"experiment_id": "RES-render01", "applicable_count": 0,
                                "notes": [], "recommendations": []},
            "variants": [], "variant_summary": "", "regime_performance": [], "notes": [],
        },
    )

    app = _run()
    _assert_no_exception(app)

    labels = {m.label: m.value for m in app.metric}
    verdict = next(v for k, v in labels.items() if "Verdict" in k)
    assert verdict == "INSUFFICIENT EVIDENCE"
    oos_expectancy = next(v for k, v in labels.items() if "OOS expectancy" in k)
    assert oos_expectancy == "-0.447R", (
        "the headline must be the out-of-sample figure, not the in-sample -0.234R"
    )
    assert app.dataframe, "the research tab drew no tables"


def test_the_research_tab_renders_when_no_study_exists(config_dir, monkeypatch, tmp_path):
    """No study: the tab renders a stated absence, not an empty or broken panel."""
    monkeypatch.setenv("GTP_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("GTP_REPORTS_DIR", str(tmp_path / "empty-reports"))

    app = _run()
    _assert_no_exception(app)

    text = " ".join(w.value for w in app.warning)
    assert "No validation context" in text
    assert "MISSING" in text
    labels = {m.label for m in app.metric}
    assert not any("Verdict" in label for label in labels), (
        "an absent study must produce no verdict tile"
    )
