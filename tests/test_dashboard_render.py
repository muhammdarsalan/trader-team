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
