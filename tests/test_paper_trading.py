"""Tests for the paper-trading loop, persistence and dashboard view."""

from __future__ import annotations

from app.config.loader import get_config, override_config
from app.config.models import RiskConfig
from app.data.schema import MarketData
from app.paper_trading.engine import PaperTradingEngine, build_graph_visualization
from app.portfolio.portfolio import Portfolio
from app.risk.engine import RiskEngine
from app.risk.models import RiskBlockReason
from app.signals.models import Signal, SignalDirection
from tests.conftest import make_ohlcv


def _cfg(config_dir):
    cfg = get_config(config_dir)
    return override_config(
        cfg,
        platform=cfg.platform.model_copy(update={"trading_enabled": True}),
        risk=cfg.risk.model_copy(update={"block_on_data_quality_warning": True}),
    )


def _market(symbol="XAUUSD", periods=80):
    return MarketData(
        symbol=symbol,
        timeframe="1D",
        df=make_ohlcv(periods=periods, start="2020-01-01", seed=42),
        provider="synthetic",
    )


def test_paper_trading_state_persistence(config_dir, tmp_path):
    cfg = _cfg(config_dir)
    state_path = tmp_path / "paper_state.json"
    engine = PaperTradingEngine(config=cfg, symbol="XAUUSD", timeframe="1D", state_path=state_path)
    engine._state["latest_decision"] = "ORDER_LONG"
    engine._state["market"] = {"symbol": "XAUUSD", "timeframe": "1D", "quality_status": "PASS"}
    engine._state["recent_events"] = [{"event": "bootstrap", "message": "ready"}]
    engine.save_state()

    reloaded = PaperTradingEngine(config=cfg, symbol="XAUUSD", timeframe="1D", state_path=state_path)
    assert reloaded.state["latest_decision"] == "ORDER_LONG"
    assert reloaded.state["market"]["quality_status"] == "PASS"
    assert reloaded.state["recent_events"][0]["event"] == "bootstrap"


def test_restart_catchup_idempotency(config_dir, tmp_path):
    cfg = _cfg(config_dir)
    market = _market(periods=15)
    state_path = tmp_path / "paper_state_catchup.json"
    engine = PaperTradingEngine(config=cfg, symbol="XAUUSD", state_path=state_path)

    engine.catch_up(market)
    first_fill_count = len(engine.state.get("fills", []))
    engine.catch_up(market)
    second_fill_count = len(engine.state.get("fills", []))

    assert first_fill_count == second_fill_count
    assert len(engine.state["processed_bars"]) == len(set(engine.state["processed_bars"]))


def test_data_quality_enforced_in_risk_path():
    engine = RiskEngine(RiskConfig(block_on_data_quality_warning=True), data_quality="WARNING", trading_enabled=True)
    signal = Signal(
        strategy="test",
        symbol="XAUUSD",
        timeframe="1D",
        direction=SignalDirection.LONG,
        confidence=0.8,
        entry_price=100.0,
        stop_loss=99.0,
        timestamp=None,
    )
    decision = engine.evaluate(signal, Portfolio(10_000.0), equity=10_000.0)
    assert decision.block_reason is RiskBlockReason.DATA_QUALITY
    assert not decision.approved


def test_dashboard_data_includes_expected_sections(config_dir):
    cfg = _cfg(config_dir)
    engine = PaperTradingEngine(config=cfg, symbol="XAUUSD", timeframe="1D")
    engine._state["market"] = {"symbol": "XAUUSD", "timeframe": "1D", "rows": 50, "last_bar": "2024-01-01T00:00:00+00:00", "quality_status": "PASS"}
    engine._state["latest_decision"] = "WAIT"
    engine._state["latest_state"] = {"regime": "TRENDING", "signals": {"momentum": "LONG"}, "decision": "WAIT"}
    engine._state["recent_events"] = [{"event": "tick", "message": "ok"}]
    engine._state["recent_errors"] = []
    dashboard = engine.dashboard_data()

    assert dashboard["market"]["symbol"] == "XAUUSD"
    assert dashboard["graph"]["nodes"]
    assert dashboard["system_health"]["quality_status"] == "PASS"
    assert dashboard["final_decision"] == "WAIT"


def test_graph_visualization_data_reports_active_and_blocked_nodes():
    state = {
        "signals": {"momentum": Signal(direction=SignalDirection.LONG, confidence=0.8, strategy="momentum", symbol="XAUUSD", timeframe="1D", entry_price=100.0, stop_loss=99.0), "trend": Signal.wait("trend", "XAUUSD", "1D")},
        "strategy_weights": {"momentum": type("W", (), {"weight": 0.9})(), "trend": type("W", (), {"weight": 0.0})()},
        "risk_decision": type("R", (), {"approved": False, "block_reason": RiskBlockReason.DATA_QUALITY})(),
        "order": object(),
    }
    graph = build_graph_visualization(state)
    assert "market_data" in graph["active_nodes"]
    assert "momentum" in graph["rejected"] or "trend" in graph["suppressed"]
    assert graph["risk_blocks"]
