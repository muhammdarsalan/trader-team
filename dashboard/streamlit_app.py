"""Simple Streamlit dashboard for the paper-trading view."""

from __future__ import annotations

import streamlit as st

from app.config.loader import get_config
from app.paper_trading.engine import PaperTradingEngine


def _default_engine() -> PaperTradingEngine:
    config = get_config()
    return PaperTradingEngine(config=config, symbol="XAUUSD", timeframe="1D")


def render_dashboard(engine: PaperTradingEngine | None = None) -> None:
    engine = engine or _default_engine()
    data = engine.dashboard_data()

    st.title("Paper Trading Dashboard")
    st.subheader(f"{data['symbol']} {data['timeframe']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Cash", f"${data['portfolio']['cash']:.2f}")
    col2.metric("Equity", f"${data['portfolio']['equity']:.2f}")
    col3.metric("Drawdown", f"{data['portfolio']['drawdown']:.1%}")

    st.write("### Market / regime")
    st.json(data["market"])
    st.write("### Strategy signals")
    st.json(data.get("strategy_signals", {}))

    st.write("### Decision flow")
    st.json(data["graph"])

    st.write("### Recent events")
    st.json(data.get("events", []))
    st.write("### Recent errors")
    st.json(data.get("errors", []))


def main() -> None:
    render_dashboard()


if __name__ == "__main__":
    main()
