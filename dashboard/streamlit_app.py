"""Paper-trading monitoring dashboard.

A monitoring surface, not a state dump. Every section answers a question an
operator actually has: is the data current and trustworthy, what regime does the
system think it is in, what did each strategy say, why was or was not a trade
taken, where does the portfolio stand, and what did the simulated fills cost.

All shaping happens in :mod:`dashboard.view`; this module only renders. That
means the page can be tested - the numbers, the statuses, the graph layout -
without a browser.

Where the engine has nothing to report, the page says so. It does not
substitute a plausible-looking zero, and it does not show a green node for a
stage that never ran.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.config.loader import get_config  # noqa: E402
from app.paper_trading.engine import PaperTradingEngine  # noqa: E402
from app.utils.paths import paper_state_path  # noqa: E402
from dashboard.view import build_view  # noqa: E402

TONE_ICONS = {"ok": "🟢", "warn": "🟠", "error": "🔴", "info": "🔵", "neutral": "⚪"}


# --------------------------------------------------------------------- engine


def load_engine(symbol: str, timeframe: str) -> PaperTradingEngine:
    """Attach to the paper session's state file, read-only in practice.

    The dashboard does not create trades. It constructs an engine over the same
    state file the runner writes, so what is displayed is what the runner
    recorded - not a second, separately-evolving copy.
    """
    config = get_config()
    return PaperTradingEngine(
        config=config,
        symbol=symbol,
        timeframe=timeframe,
        state_path=paper_state_path(symbol, timeframe),
    )


# --------------------------------------------------------------------- helpers


def _banner(banner: dict[str, str]) -> None:
    body = f"**{banner['title']}** — {banner['body']}"
    tone = banner.get("tone")
    if tone == "error":
        st.error(body)
    elif tone == "warn":
        st.warning(body)
    else:
        st.info(body)


def _metric_columns(metrics: list[dict[str, Any]], per_row: int = 5) -> None:
    for start in range(0, len(metrics), per_row):
        chunk = metrics[start : start + per_row]
        for column, metric in zip(st.columns(len(chunk)), chunk, strict=False):
            icon = TONE_ICONS.get(metric.get("tone", "neutral"), "")
            label = f"{icon} {metric['label']}".strip()
            column.metric(label, metric["display"], help=metric.get("help"))


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    """Render selected columns of ``rows`` as a dataframe."""
    if not rows:
        return
    frame = pd.DataFrame(
        [{title: row.get(key) for key, title in columns} for row in rows]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------- sections


def render_header(view: dict[str, Any]) -> None:
    header = view["header"]
    title = f"{header['symbol']} {header['timeframe']} — paper trading"
    st.title(title)
    if header.get("instrument_name"):
        st.caption(
            f"{header['instrument_name']} · view generated {header.get('generated_at')}"
        )

    for banner in view["banners"]:
        _banner(banner)

    _metric_columns(header["metrics"])
    if header["reasons"]:
        with st.expander(
            f"{TONE_ICONS.get(header['status_tone'], '')} Why the system is "
            f"{header['status']}",
            expanded=header["status_tone"] in {"warn", "error"},
        ):
            for reason in header["reasons"]:
                st.write(f"- {reason}")


def render_market(view: dict[str, Any]) -> None:
    market = view["market"]
    st.subheader("Market")
    if market["status"] in {"NO_DATA", "EMPTY"}:
        st.warning(
            "No market data is loaded for this session. Run "
            "`python scripts/run_paper_trading.py --replay` to backfill from cached "
            "history, or `--live` to fetch the newest bars."
        )
    _metric_columns(market["metrics"])
    if market.get("caveat"):
        st.caption(f"Data caveat: {market['caveat']}")
    issues = (market.get("quality_detail") or {}).get("issues")
    if issues:
        with st.expander("Data-quality findings"):
            for issue in issues:
                st.write(f"- {issue}")


def render_regime(view: dict[str, Any]) -> None:
    regime = view["regime"]
    st.subheader("Regime")
    if not regime["detected"]:
        st.info(regime["message"])
        return
    _metric_columns(regime["metrics"])
    if regime["reasoning"]:
        with st.expander("What the detector based this on", expanded=False):
            for line in regime["reasoning"]:
                st.write(f"- {line}")
    if regime["indicators"]:
        with st.expander("Regime metrics"):
            st.dataframe(
                pd.DataFrame(
                    [{"metric": k, "value": v} for k, v in regime["indicators"].items()]
                ),
                use_container_width=True,
                hide_index=True,
            )


def render_strategies(view: dict[str, Any]) -> None:
    rows = view["strategies"]
    st.subheader("Strategies")
    if not rows:
        st.info(
            "No strategy has reported yet. Strategies run only on bars past the "
            "feature warm-up period."
        )
        return

    contributing = [r for r in rows if r["state"] == "CONTRIBUTING"]
    suppressed = [r for r in rows if r["state"] == "SUPPRESSED"]
    declined = [r for r in rows if r["state"] == "DECLINED"]
    missing = [r for r in rows if r["state"] == "NO_SIGNAL"]

    cols = st.columns(4)
    cols[0].metric("🟢 Contributing", len(contributing))
    cols[1].metric("🟠 Suppressed", len(suppressed))
    cols[2].metric("⚪ Declined", len(declined))
    cols[3].metric("🔴 No signal", len(missing))

    _table(
        rows,
        [
            ("strategy", "Strategy"),
            ("state", "State"),
            ("direction", "Direction"),
            ("confidence_display", "Confidence"),
            ("weight_display", "Weight"),
            ("entry_price", "Entry"),
            ("stop_loss", "Stop"),
            ("take_profit", "Target"),
            ("reason", "Reason"),
        ],
    )

    if suppressed:
        st.caption(
            "Suppressed strategies signalled but were weighted to zero by the "
            "selector, so they could not contribute to the aggregate."
        )
    if missing:
        st.error(
            "A strategy with no signal means its node did not report — that is a "
            "failure, not a decision to wait: "
            + ", ".join(str(r["strategy"]) for r in missing)
        )

    with st.expander("Per-strategy detail"):
        for row in rows:
            st.markdown(f"**{row['strategy']}** — {row['state']} ({row['direction']})")
            for line in row["reasoning"]:
                st.write(f"- {line}")
            for line in row["weight_reasoning"]:
                st.write(f"- weight: {line}")


def render_decision(view: dict[str, Any]) -> None:
    decision = view["decision"]
    st.subheader("Decision")

    left, right = st.columns([1, 2])
    left.metric("Final decision", str(decision["final_decision"]))
    right.write(f"**Bar:** {decision.get('timestamp') or 'none'}")
    if decision.get("trading_enabled") is False:
        right.write("**Kill switch:** on — no paper order will be created.")
    if decision.get("skipped_reason"):
        st.info(decision["skipped_reason"])

    for step in decision["steps"]:
        icon = TONE_ICONS.get(step.get("tone", "neutral"), "")
        st.markdown(f"{icon} **{step['step']}: {step['outcome']}**")
        st.caption(step["detail"])

    if decision["risk_blocks"]:
        st.error("Risk blocked this trade: " + "; ".join(str(b) for b in decision["risk_blocks"]))
    if decision["errors"]:
        st.error(
            "Node errors on this bar: "
            + "; ".join(f"{e.get('node')}: {e.get('error')}" for e in decision["errors"])
        )
    if decision["trace"]:
        with st.expander("Graph trace"):
            for line in decision["trace"]:
                st.write(f"- {line}")


def render_portfolio(view: dict[str, Any]) -> None:
    portfolio = view["portfolio"]
    st.subheader("Portfolio")
    if not portfolio["marked_to_market"]:
        st.warning(
            "Open positions have no mark price, so unrealised P&L and drawdown "
            "below are not measurements. Process a bar to mark them."
        )
    _metric_columns(portfolio["metrics"], per_row=6)

    series = portfolio["equity_series"]
    if series:
        frame = pd.DataFrame(series)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="ISO8601", utc=True)
        frame = frame.set_index("timestamp")
        st.line_chart(frame[["equity"]], height=240)
        st.area_chart(frame[["drawdown"]], height=140)
    else:
        st.info("No equity history yet — the curve appears once bars are processed.")

    st.markdown("**Open positions**")
    if portfolio["positions"]:
        _table(
            portfolio["positions"],
            [
                ("symbol", "Symbol"),
                ("direction", "Direction"),
                ("strategy", "Strategy"),
                ("quantity", "Qty"),
                ("entry_price", "Entry"),
                ("mark_price", "Mark"),
                ("stop_loss", "Stop"),
                ("take_profit", "Target"),
                ("unrealised_display", "Unrealised"),
                ("r_multiple", "R"),
                ("entry_regime", "Entry regime"),
            ],
        )
    else:
        st.caption("Flat — no simulated position is open.")

    st.markdown("**Recent closed trades**")
    if view["trades"]:
        _table(
            view["trades"],
            [
                ("exit_time", "Closed"),
                ("strategy", "Strategy"),
                ("direction", "Direction"),
                ("entry_price", "Entry"),
                ("exit_price", "Exit"),
                ("exit_reason", "Reason"),
                ("net_display", "Net"),
                ("r_display", "R"),
                ("bars_held", "Bars"),
            ],
        )
    else:
        st.caption("No trade has closed yet.")


def render_execution(view: dict[str, Any]) -> None:
    execution = view["execution"]
    st.subheader("Execution")
    st.caption(execution["cost_note"])

    st.markdown("**Pending paper orders**")
    if execution["pending_orders"]:
        _table(
            execution["pending_orders"],
            [
                ("symbol", "Symbol"),
                ("side", "Side"),
                ("quantity", "Qty"),
                ("strategy", "Strategy"),
                ("stop_loss", "Stop"),
                ("take_profit", "Target"),
                ("risk_amount", "Risk"),
                ("queued_on_bar", "Queued on"),
            ],
        )
    else:
        st.caption("Nothing queued for the next bar.")

    st.markdown("**Recent simulated fills**")
    if execution["fills"]:
        _table(
            execution["fills"],
            [
                ("timestamp", "Time"),
                ("side", "Side"),
                ("status", "Status"),
                ("quantity", "Qty"),
                ("reference_price", "Reference"),
                ("price_display", "Filled"),
                ("difference_display", "Diff"),
                ("spread_cost", "Spread"),
                ("slippage_cost", "Slippage"),
                ("commission", "Commission"),
                ("total_cost", "Total cost"),
                ("exit_reason", "Exit reason"),
            ],
        )
    else:
        st.caption("No fill has been simulated yet.")


def build_graph_figure(spec: dict[str, Any]) -> go.Figure:
    """Draw the decision graph: Market Data → … → Paper Position."""
    figure = go.Figure()

    for edge in spec["edges"]:
        figure.add_trace(
            go.Scatter(
                x=[edge["x0"], edge["x1"]],
                y=[edge["y0"], edge["y1"]],
                mode="lines",
                line={
                    "width": 3 if edge["on_path"] else 1,
                    "color": "#1a7f37" if edge["on_path"] else "#d0d7de",
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    nodes = spec["nodes"]
    figure.add_trace(
        go.Scatter(
            x=[n["x"] for n in nodes],
            y=[n["y"] for n in nodes],
            mode="markers+text",
            marker={
                "size": 46,
                "color": [n["colour"] for n in nodes],
                "line": {"width": 2, "color": "#ffffff"},
                "symbol": "square",
            },
            text=[n["label"].replace("\n", "<br>") for n in nodes],
            textposition="bottom center",
            textfont={"size": 10},
            hovertext=[n["hover"] for n in nodes],
            hoverinfo="text",
            showlegend=False,
        )
    )

    figure.update_layout(
        height=460,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={
            "range": spec["x_range"],
            "tickmode": "array",
            "tickvals": [t["x"] for t in spec["stage_ticks"]],
            "ticktext": [t["label"] for t in spec["stage_ticks"]],
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "range": spec["y_range"],
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
        },
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def render_graph(view: dict[str, Any]) -> None:
    graph = view["graph"]
    spec = view["graph_figure"]
    st.subheader("Decision graph")
    st.caption(spec["summary"])

    if not spec["nodes"]:
        st.info("No graph state recorded yet.")
        return

    st.plotly_chart(build_graph_figure(spec), use_container_width=True)

    legend = spec["legend"]
    if legend:
        st.caption(
            " · ".join(
                f"<span style='color:{colour}'>■</span> {status}"
                for status, colour in legend.items()
            ),
            unsafe_allow_html=True,
        )

    cols = st.columns(4)
    cols[0].metric("On the live path", len(spec["path"]))
    cols[1].metric("Suppressed", len(graph.get("suppressed") or []))
    cols[2].metric("Rejected", len(graph.get("rejected") or []))
    cols[3].metric("Failed nodes", len(graph.get("failed") or []))

    with st.expander("Node-by-node status"):
        _table(
            graph.get("nodes") or [],
            [("label", "Node"), ("stage", "Stage"), ("status", "Status"), ("detail", "Detail")],
        )


def render_activity(view: dict[str, Any]) -> None:
    st.subheader("Activity")
    errors, events = view["errors"], view["events"]

    if errors:
        st.error(f"{len(errors)} recorded error(s). Most recent first.")
        _table(errors, [("timestamp", "Time"), ("node", "Node"), ("error", "Error")])
    else:
        st.caption("No errors recorded.")

    st.markdown("**Recent events**")
    if events:
        _table(events, [("timestamp", "Time"), ("event", "Event"), ("message", "Detail")])
    else:
        st.caption("No events recorded.")


# ------------------------------------------------------------------------ page


def render_dashboard(engine: PaperTradingEngine | None = None) -> dict[str, Any]:
    """Render the whole page. Returns the view model, which tests read."""
    if engine is None:
        symbol = st.session_state.get("symbol", "XAUUSD")
        timeframe = st.session_state.get("timeframe", "1D")
        engine = load_engine(symbol, timeframe)

    view = build_view(engine.dashboard_data())

    render_header(view)
    tabs = st.tabs(
        ["Market & regime", "Strategies & decision", "Portfolio", "Execution",
         "Graph", "Activity"]
    )
    with tabs[0]:
        render_market(view)
        st.divider()
        render_regime(view)
    with tabs[1]:
        render_strategies(view)
        st.divider()
        render_decision(view)
    with tabs[2]:
        render_portfolio(view)
    with tabs[3]:
        render_execution(view)
    with tabs[4]:
        render_graph(view)
    with tabs[5]:
        render_activity(view)
    return view


def render_sidebar() -> tuple[str, str]:
    """Session selector and the safety facts that belong in permanent view."""
    config = get_config()
    symbols = config.assets.enabled_symbols()

    st.sidebar.header("Session")
    symbol = st.sidebar.selectbox("Instrument", symbols, key="symbol")
    timeframe = st.sidebar.selectbox(
        "Timeframe", config.data.supported_timeframes,
        index=config.data.supported_timeframes.index("1D")
        if "1D" in config.data.supported_timeframes
        else 0,
        key="timeframe",
    )

    state_file = paper_state_path(symbol, timeframe)
    st.sidebar.caption(f"State file: `{state_file}`")
    if not state_file.exists():
        st.sidebar.warning(
            "No state file yet. Start a session with "
            "`python scripts/run_paper_trading.py --replay`."
        )

    st.sidebar.header("Safety")
    st.sidebar.write(f"Mode: **{config.platform.mode}**")
    st.sidebar.write(
        f"Kill switch: **{'off — paper orders allowed' if config.platform.trading_enabled else 'on — no orders'}**"
    )
    st.sidebar.write("Live trading: **disabled** (no broker integration exists)")
    st.sidebar.caption(
        "This dashboard reads recorded paper-trading state. It does not place "
        "orders and cannot reach a broker."
    )
    if st.sidebar.button("Reload state"):
        st.rerun()
    return symbol, timeframe


def main() -> None:
    st.set_page_config(
        page_title="Paper trading monitor", page_icon="📉", layout="wide"
    )
    symbol, timeframe = render_sidebar()
    try:
        engine = load_engine(symbol, timeframe)
    except Exception as exc:  # noqa: BLE001 - show the real cause, never a placeholder
        st.error(
            f"The paper-trading engine could not be constructed: "
            f"{type(exc).__name__}: {exc}"
        )
        st.caption(
            "This is the underlying failure, shown deliberately rather than "
            "replaced with empty panels. Fix the cause above."
        )
        return
    render_dashboard(engine)


if __name__ == "__main__":
    main()
