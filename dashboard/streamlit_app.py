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
    st.dataframe(frame, width="stretch", hide_index=True)


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

    findings = market.get("quality_findings") or []
    if findings:
        worst = findings[0]
        header = (
            f"Data-quality findings ({len(findings)}) — worst: "
            f"{worst['severity']} {worst['code']}"
        )
        with st.expander(header, expanded=worst["severity"] == "FAIL"):
            st.caption(
                "The quality gate is authoritative: these findings are what decided the "
                "grade above, and the grade is what the risk engine sizes against."
            )
            _table(
                findings,
                [
                    ("severity", "Severity"),
                    ("code", "Finding"),
                    ("bars", "Bars"),
                    ("share_display", "Share of series"),
                    ("message", "Detail"),
                ],
            )
            for finding in findings:
                if finding["samples"]:
                    st.caption(
                        f"{finding['code']} examples: "
                        + ", ".join(str(x) for x in finding["samples"][:5])
                    )
    else:
        st.caption("The quality gate recorded no findings for this series.")


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
                width="stretch",
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


def render_performance(view: dict[str, Any]) -> None:
    """Realised performance, sliced by strategy and by regime."""
    perf = view["performance"]
    st.subheader("Performance")
    st.caption(perf["basis"])

    if not perf["has_trades"]:
        st.info(
            "No trade has closed yet, so there is nothing to measure. Performance here "
            "is realised only — an open position has a mark, not an outcome."
        )
        return

    _metric_columns(perf["metrics"], per_row=4)
    st.write(perf["verdict"])

    columns = [
        ("trades", "Trades"),
        ("wins", "Wins"),
        ("losses", "Losses"),
        ("win_rate_display", "Win rate"),
        ("expectancy_display", "Expectancy"),
        ("profit_factor_display", "Profit factor"),
        ("net_display", "Net"),
        ("avg_bars_held", "Avg bars"),
        ("evidence_display", "Evidence"),
    ]

    st.markdown("**By contributing strategy**")
    st.caption(perf["contributor_note"])
    _table(perf["by_contributor"], [("strategy", "Strategy"), *columns])

    st.markdown("**By signal source**")
    st.caption(
        "The strategy name carried on the trade itself. The aggregator signs its output "
        "\u201censemble\u201d, so a combined decision appears under that name here and is "
        "broken out by contributor above."
    )
    _table(perf["by_strategy"], [("strategy", "Strategy"), *columns])

    st.markdown("**By entry regime**")
    _table(perf["by_regime"], [("regime", "Regime"), *columns])

    st.markdown("**By strategy within regime**")
    st.caption(
        "The pairing the selector actually weights on. Rows marked "
        "\u201csample only\u201d are below the evidence threshold and are not moving any "
        "weight yet."
    )
    _table(
        perf["by_strategy_regime"],
        [("strategy", "Strategy"), ("regime", "Regime"), *columns],
    )

    thin = [r for r in perf["by_strategy_regime"] if r["insufficient_evidence"]]
    if thin:
        st.warning(
            f"{len(thin)} of {len(perf['by_strategy_regime'])} strategy/regime pairings "
            f"have fewer than {perf['min_samples']} closed trades. Their expectancy is a "
            "small sample, not evidence, and the selector does not act on it."
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**How trades ended**")
        _table(perf["by_exit_reason"], [("exit_reason", "Exit reason"), ("trades", "Trades")])
    with right:
        st.markdown("**Selector's performance table**")
        st.caption(
            "What the strategy selector is weighting on right now. It can hold samples "
            "from trades that have aged out of the bounded trade log above."
        )
        _table(
            perf["selector_table"],
            [
                ("strategy", "Strategy"),
                ("regime", "Regime"),
                ("samples", "Samples"),
                ("mean_r_display", "Mean R"),
                ("win_rate_display", "Win rate"),
                ("influence_display", "Effect on weight"),
            ],
        )


def render_research(view: dict[str, Any]) -> None:
    """Validation context for the configuration this session is running."""
    research = view["research"]
    st.subheader("Research and validation")

    if not research["available"]:
        # The whole panel, when there is nothing to show. No tables of zeros.
        message = f"**No validation context: {research['availability']}** — {research['reason']}"
        if research["availability"] == "UNREADABLE":
            st.error(message)
        else:
            st.warning(message)
        st.caption(
            "This is deliberately blank. A research panel showing zeros for a study "
            "that was never run would read as \u201ctested, found nothing\u201d, which is a "
            "much stronger claim than \u201cnever tested\u201d."
        )
        if research["other_experiments"]:
            with st.expander(f"Other studies on disk ({len(research['other_experiments'])})"):
                _table(
                    research["other_experiments"],
                    [
                        ("experiment_id", "Experiment"),
                        ("symbol", "Symbol"),
                        ("timeframe", "TF"),
                        ("verdict", "Verdict"),
                        ("created_at", "Generated"),
                    ],
                )
        _render_evidence_ladder(research)
        return

    _metric_columns(research["metrics"], per_row=3)

    st.markdown(f"**Verdict: {research['verdict']}**")
    for reason in research["verdict_reasons"]:
        st.write(f"- {reason}")
    st.caption(
        "There is no verdict meaning \u201cprofitable\u201d and there never will be. The best "
        "available conclusion is that a configuration survived one round of testing."
    )

    if research["evidence_summary"]:
        st.divider()
        st.markdown("**What has and has not been established**")
        st.caption(
            "These fail independently. Merging them is the most common way a research "
            "result gets overstated, so each is stated separately with its own limitation."
        )
        _table(
            research["evidence_summary"],
            [
                ("category", "Category"),
                ("status", "Status"),
                ("detail", "What was found"),
                ("limitation", "What it still cannot tell you"),
            ],
        )

    st.divider()
    st.markdown("**In-sample vs out-of-sample**")
    st.caption(
        "The same arithmetic on different bars. Only the label distinguishes them, which "
        "is why they are never summed or headlined together."
    )
    _table(
        research["segments"],
        [
            ("segment", "Window"),
            ("supports", "What it supports"),
            ("bars", "Bars"),
            ("trades", "Trades"),
            ("return_display", "Return"),
            ("expectancy_display", "Expectancy"),
            ("drawdown_display", "Max DD"),
            ("win_rate_display", "Win rate"),
            ("data_quality", "Data quality"),
        ],
    )

    wf = research["walk_forward"]
    if wf:
        st.markdown("**Walk-forward**")
        cols = st.columns(3)
        cols[0].metric("Folds", wf["folds"])
        cols[1].metric("Profitable folds", wf["profitable_display"])
        cols[2].metric(
            "Efficiency",
            "n/a" if wf["efficiency"] is None else f"{wf['efficiency']:.2f}",
            help="Mean test objective over mean training objective.",
        )
        for warning in wf["warnings"]:
            st.warning(warning)
    else:
        st.caption("Walk-forward analysis was not run for this study.")

    robustness = research["robustness"]
    if robustness:
        st.markdown("**Parameter robustness**")
        st.caption(
            f"Swept on the {robustness['measured_on']} window against "
            f"{robustness['objective']}. A flat neighbourhood is the good outcome, even "
            "at a mediocre level; a sharp peak at the shipped value describes the sample."
        )
        _table(
            robustness["rows"],
            [
                ("parameter", "Parameter"),
                ("spread_display", "Spread (CV)"),
                ("baseline_is_peak", "Shipped value is the peak"),
                ("verdict", "Verdict"),
            ],
        )
        if robustness["fragile_parameters"]:
            st.error(
                "Fragile parameters: " + ", ".join(robustness["fragile_parameters"])
            )
    else:
        st.caption("No parameter sensitivity sweep is recorded for this study.")

    cost_stress = research["cost_stress"]
    if cost_stress:
        st.markdown("**Execution-cost stress**")
        st.caption(
            f"Same configuration on the {cost_stress['measured_on']} window under fixed, "
            "declared adverse cost scenarios. The baseline is the headline; adverse "
            "scenarios are only ever degradation from it. Surviving cost stress is not "
            "evidence of profitability."
        )
        cols = st.columns(3)
        cols[0].metric("Baseline expectancy", cost_stress["baseline_display"])
        cols[1].metric("Cost drag range", cost_stress["cost_drag_display"])
        cols[2].metric("Survival", cost_stress["survival_status"])
        detail = cost_stress["survival_detail"]
        tone = cost_stress["tone"]
        if tone == "error":
            st.error(detail)
        elif tone == "ok":
            st.success(detail)
        else:
            st.warning(detail)
        _table(
            cost_stress["rows"],
            [
                ("scenario", "Scenario"),
                ("trades", "Trades"),
                ("return_display", "Return"),
                ("expectancy_display", "Expectancy"),
                ("cost_drag_display", "Cost drag"),
            ],
        )
    else:
        st.caption("No execution-cost stress study is recorded for this study.")

    correlation = research["correlation"]
    if correlation:
        st.markdown("**Strategy correlation and redundancy**")
        st.caption(
            f"Measured on the {correlation['measured_on']} window at threshold "
            f"{correlation['threshold']}. A correlation is a hypothesis about redundancy, "
            "not a finding about it — no strategy is disabled on the strength of one."
        )
        if correlation["findings"]:
            _table(
                correlation["findings"],
                [
                    ("pair", "Pair"),
                    ("correlation_display", "Return correlation"),
                    ("agreement_display", "Signal agreement"),
                    ("observations", "Observations"),
                ],
            )
        else:
            st.caption("No pair exceeded the threshold.")

    overfitting = research["overfitting"]
    if overfitting:
        st.markdown("**Overfitting diagnostics**")
        cols = st.columns(2)
        cols[0].metric("Configurations tried", f"{overfitting.get('trials', 0):,}")
        deflated = overfitting.get("deflated_sharpe")
        cols[1].metric(
            "Deflated Sharpe",
            "n/a" if deflated is None else f"{deflated:.2f}",
            help="Sharpe adjusted for how many configurations were searched to find it.",
        )
        for finding in overfitting["findings"]:
            text = f"**{finding['severity']}** — {finding['message']}"
            if finding["tone"] == "error":
                st.error(text)
            elif finding["tone"] == "warn":
                st.warning(text)
            else:
                st.info(text)

    st.divider()
    st.markdown("**Recommendations**")
    st.caption(research["recommendation_note"])
    if research["recommendations"]:
        _table(
            research["recommendations"],
            [
                ("subject", "Finding"),
                ("action", "Action"),
                ("evidence_tier", "Evidence"),
                ("strategy", "Strategy"),
                ("proposed_weight", "Proposed weight"),
                ("applicable", "Meets the bar"),
            ],
        )
        with st.expander("Why each recommendation says what it says"):
            for rec in research["recommendations"]:
                st.markdown(f"**{rec['subject']}** — {rec['action']} ({rec['evidence_tier']})")
                for line in rec["rationale"]:
                    st.write(f"- {line}")
                for blocker in rec["blockers"]:
                    st.write(f"- _not applicable:_ {blocker}")
    else:
        st.caption("Nothing was flagged, so there is nothing to recommend.")

    st.divider()
    st.markdown("**Experiment provenance**")
    st.caption(
        "What this study was run against. A result without its data checksum, code "
        "revision and seed is not reproducible, and an irreproducible result is an "
        "anecdote."
    )
    provenance = research["provenance"]
    _table(
        # Values are stringified: the column mixes ints (bars, seed) with strings
        # (checksums, dates), and a mixed object column fails Arrow conversion,
        # which drops the table silently rather than raising.
        [
            {"field": k.replace("_", " "), "value": "n/a" if v is None else str(v)}
            for k, v in provenance.items()
        ],
        [("field", "Field"), ("value", "Value")],
    )
    if research["report_path"]:
        st.caption(f"Full report: `{research['report_path']}`")

    _render_evidence_ladder(research)


def _render_evidence_ladder(research: dict[str, Any]) -> None:
    """What each rung of evidence does and does not support."""
    ladder = research.get("evidence_ladder") or []
    if not ladder:
        return
    with st.expander("What counts as evidence here"):
        st.caption(
            "A green test suite says the implementation does what it was written to do. "
            "It says nothing about whether the strategy has an edge. These are separate "
            "claims and this page keeps them separate."
        )
        _table(
            ladder,
            [
                ("rung", "Rung"),
                ("supports", "Supports"),
                ("does_not_support", "Does not support"),
            ],
        )


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

    st.plotly_chart(build_graph_figure(spec), width="stretch")

    legend = spec["legend"]
    if legend:
        st.caption(
            " · ".join(
                f"<span style='color:{colour}'>■</span> {status}"
                for status, colour in legend.items()
            ),
            unsafe_allow_html=True,
        )

    cols = st.columns(5)
    cols[0].metric("Stages reached", len(spec["path"]))
    cols[1].metric(
        "Stopped at",
        str(spec.get("stopped_at") or "completed"),
        help=spec.get("stopped_reason") or "The bar ran through every stage.",
    )
    cols[2].metric("Suppressed", len(graph.get("suppressed") or []))
    cols[3].metric("Rejected", len(graph.get("rejected") or []))
    cols[4].metric("Failed nodes", len(graph.get("failed") or []))

    if graph.get("risk_blocks"):
        st.error("Risk blocked this bar: " + "; ".join(str(b) for b in graph["risk_blocks"]))

    with st.expander("Node-by-node status"):
        st.caption(
            "\u201cReached\u201d says the stage ran and recorded a result; \u201cstatus\u201d "
            "says what it concluded. A stage can be reached and still be amber — a feed "
            "the quality gate caveated, or a strategy that declined to act."
        )
        _table(
            graph.get("nodes") or [],
            [
                ("label", "Node"),
                ("stage", "Stage"),
                ("status", "Status"),
                ("reached", "Reached"),
                ("terminal", "Ended the bar"),
                ("detail", "Detail"),
            ],
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
         "Performance", "Research", "Graph", "Activity"]
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
        render_performance(view)
    with tabs[5]:
        render_research(view)
    with tabs[6]:
        render_graph(view)
    with tabs[7]:
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
