"""Dashboard shaping, with no rendering runtime attached.

Everything the monitoring page displays is computed here, as plain data. The
Streamlit module is then a thin translation of this into widgets.

Two reasons for the split. A test can assert that the equity figure on the page
is the equity the engine computed, and that a suppressed strategy is shown as
suppressed, without starting a browser or a Streamlit session. And the panel
definitions stay readable: what a panel means is decided in one place, not
spread through ``st.metric`` calls.

Each metric carries both its raw ``value`` and a formatted ``display``. Tests
read the value; the page shows the display. That way a formatting change cannot
quietly alter what is being asserted, and an assertion cannot pass on a
correctly-formatted wrong number.

Nothing here invents a value. Where the engine has no answer, the panel says so
in words - ``"not yet run"``, ``"unverified"`` - rather than showing a zero that
reads like a measurement.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "STATUS_TONES",
    "build_view",
    "decision_panel",
    "execution_panel",
    "graph_figure_spec",
    "header_panel",
    "market_panel",
    "portfolio_panel",
    "regime_panel",
    "safety_banners",
    "strategy_rows",
    "trade_rows",
]

#: How each status word should read. ``error`` and ``warn`` exist so a degraded
#: state cannot be rendered in the same tone as a healthy one.
STATUS_TONES: dict[str, str] = {
    "HEALTHY": "ok",
    "IDLE": "info",
    "DEGRADED": "warn",
    "FAILED": "error",
    "FRESH": "ok",
    "DELAYED": "warn",
    "STALE": "error",
    "UNKNOWN": "warn",
    "PASS": "ok",
    "WARNING": "warn",
    "FAIL": "error",
    "OK": "ok",
    "NO_DATA": "warn",
    "EMPTY": "warn",
    "ERROR": "error",
}


def _metric(
    label: str,
    value: Any,
    display: str | None = None,
    *,
    tone: str = "neutral",
    help_text: str | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "display": display if display is not None else _default_display(value),
        "tone": tone,
        "help": help_text,
    }


def _default_display(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def _money(value: float | None, currency: str = "USD") -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    symbol = "$" if currency == "USD" else ""
    return f"{sign}{symbol}{abs(value):,.2f}" + ("" if symbol else f" {currency}")


def _pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _tone(status: Any) -> str:
    return STATUS_TONES.get(str(status).upper(), "neutral")


def _price(value: Any, digits: int = 5) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


# ------------------------------------------------------------------- banners


def safety_banners(data: dict[str, Any]) -> list[dict[str, str]]:
    """Warnings that must be visible before any number on the page is read.

    Order matters: the data caveat comes first, because a reader who has not yet
    been told that "XAUUSD" is actually front-month futures will misread every
    price below it. The paper-mode notice is always present - it is the answer to
    "is this real money?", and that question should never require scrolling.
    """
    banners: list[dict[str, str]] = []
    instrument = data.get("instrument") or {}
    market = data.get("market") or {}
    safety = data.get("safety") or {}
    health = data.get("system_health") or {}

    if instrument.get("is_proxy"):
        banners.append(
            {
                "tone": "warn",
                "title": f"{market.get('symbol', 'This instrument')} is a proxy series",
                "body": str(
                    instrument.get("data_caveat")
                    or "The configured provider series stands in for this instrument "
                    "rather than being it."
                ),
            }
        )
    elif instrument.get("data_caveat"):
        banners.append(
            {
                "tone": "info",
                "title": f"Data caveat for {market.get('symbol', 'this instrument')}",
                "body": str(instrument["data_caveat"]),
            }
        )

    quality = str(market.get("quality_status", "UNKNOWN")).upper()
    if quality == "FAIL":
        banners.append(
            {
                "tone": "error",
                "title": "Data quality FAIL",
                "body": "The quality gate rejected this feed. The risk engine refuses to "
                "size a position against it, so no order will be created.",
            }
        )
    elif quality == "UNKNOWN":
        banners.append(
            {
                "tone": "warn",
                "title": "Data quality unverified",
                "body": "No quality grade has been recorded for this feed. An unstated "
                "grade is treated as a refusal, not as a pass.",
            }
        )
    elif quality == "WARNING":
        banners.append(
            {
                "tone": "warn",
                "title": "Data quality WARNING",
                "body": "The quality gate found defects in this feed. Whether that blocks "
                "trading depends on block_on_data_quality_warning in configs/risk.yaml.",
            }
        )

    freshness = market.get("freshness") or {}
    if str(freshness.get("status", "")).upper() in {"STALE", "UNKNOWN"}:
        banners.append(
            {
                "tone": "error" if freshness.get("status") == "STALE" else "warn",
                "title": f"Market data {str(freshness.get('status', '')).lower()}",
                "body": str(freshness.get("detail") or "The feed's age could not be established."),
            }
        )

    if not safety.get("trading_enabled", False):
        banners.append(
            {
                "tone": "info",
                "title": "Kill switch on (trading_enabled = false)",
                "body": "Signals are analysed and risk is evaluated, but no paper order is "
                "created. Set platform.trading_enabled in configs/platform.yaml to "
                "let the simulator place paper orders.",
            }
        )

    banners.append(
        {
            "tone": "info",
            "title": f"Paper trading only - mode {safety.get('mode', 'RESEARCH')}",
            "body": str(
                safety.get("note")
                or "Fills come from the execution simulator. No broker is contacted."
            ),
        }
    )

    if health.get("status") == "IDLE":
        banners.append(
            {
                "tone": "warn",
                "title": "Nothing has run yet",
                "body": "No bar has been processed, so every figure below is a starting "
                "value rather than a measurement. Run a catch-up or a live tick.",
            }
        )
    return banners


# -------------------------------------------------------------------- panels


def header_panel(data: dict[str, Any]) -> dict[str, Any]:
    """The status line: what instrument, how healthy, how current."""
    health = data.get("system_health") or {}
    market = data.get("market") or {}
    freshness = market.get("freshness") or {}
    instrument = data.get("instrument") or {}

    return {
        "symbol": market.get("symbol") or data.get("symbol"),
        "timeframe": market.get("timeframe") or data.get("timeframe"),
        "instrument_name": instrument.get("name"),
        "generated_at": data.get("generated_at"),
        "status": health.get("status", "UNKNOWN"),
        "status_tone": _tone(health.get("status")),
        "reasons": list(health.get("reasons") or []),
        "metrics": [
            _metric(
                "System",
                health.get("status", "UNKNOWN"),
                tone=_tone(health.get("status")),
                help_text="HEALTHY only when bars have been processed, data is current, "
                "quality is graded and nothing has failed.",
            ),
            _metric(
                "Data freshness",
                freshness.get("status", "UNKNOWN"),
                tone=_tone(freshness.get("status")),
                help_text=freshness.get("detail"),
            ),
            _metric(
                "Data quality",
                market.get("quality_status", "UNKNOWN"),
                tone=_tone(market.get("quality_status")),
                help_text="The quality gate's grade for this feed.",
            ),
            _metric(
                "Bars processed",
                int(health.get("bars_processed") or 0),
                f"{int(health.get('bars_processed') or 0):,}",
                help_text=f"{int(health.get('bars_decided') or 0):,} of them reached the "
                "decision graph; the rest were inside the feature warm-up.",
            ),
            _metric(
                "Final decision",
                data.get("final_decision") or "none",
                help_text="The decision graph's verdict on the most recent bar.",
            ),
        ],
    }


def market_panel(data: dict[str, Any]) -> dict[str, Any]:
    """Instrument, price, timestamp, source, freshness and quality."""
    market = data.get("market") or {}
    instrument = data.get("instrument") or {}
    freshness = market.get("freshness") or {}
    quality = market.get("quality") or {}

    price = market.get("last_close")
    return {
        "status": market.get("status", "NO_DATA"),
        "status_tone": _tone(market.get("status")),
        "caveat": instrument.get("data_caveat"),
        "is_proxy": bool(instrument.get("is_proxy")),
        "metrics": [
            _metric(
                "Instrument",
                market.get("symbol"),
                f"{market.get('symbol', '?')} ({instrument.get('name') or 'unnamed'})",
            ),
            _metric(
                "Last close",
                None if price is None else float(price),
                _price(price),
                help_text="The close of the most recently processed bar, not a live tick.",
            ),
            _metric(
                "Last bar",
                market.get("last_bar"),
                str(market.get("last_bar") or "none"),
                tone=_tone(freshness.get("status")),
            ),
            _metric(
                "Bars loaded",
                int(market.get("rows") or 0),
                f"{int(market.get('rows') or 0):,}",
            ),
            _metric(
                "Provider",
                market.get("provider"),
                (
                    f"{market.get('provider') or 'none'}"
                    + (
                        f" ({instrument['provider_symbol']})"
                        if instrument.get("provider_symbol")
                        else ""
                    )
                ),
                help_text="The vendor series the platform symbol maps to.",
            ),
            _metric(
                "Source",
                market.get("source"),
                {
                    "live_refresh": "live refresh (delayed vendor feed)",
                    "historical_replay": "historical replay (deterministic)",
                }.get(str(market.get("source")), str(market.get("source") or "none")),
            ),
            _metric(
                "Freshness",
                freshness.get("status", "UNKNOWN"),
                (
                    f"{freshness.get('status', 'UNKNOWN')}"
                    + (
                        f" - {freshness['intervals']:.1f} intervals old"
                        if freshness.get("intervals") is not None
                        else ""
                    )
                ),
                tone=_tone(freshness.get("status")),
                help_text=freshness.get("detail"),
            ),
            _metric(
                "Quality",
                market.get("quality_status", "UNKNOWN"),
                str(market.get("quality_status", "UNKNOWN")),
                tone=_tone(market.get("quality_status")),
                help_text=(
                    "; ".join(str(i) for i in (quality.get("issues") or [])[:3]) or None
                ),
            ),
            _metric(
                "Volume usable",
                bool(instrument.get("has_reliable_volume")),
                "yes" if instrument.get("has_reliable_volume") else "no - suppressed",
                help_text="Volume features are suppressed where the feed's volume is not a "
                "real traded quantity.",
            ),
        ],
        "quality_detail": quality,
        "freshness_detail": freshness,
    }


def regime_panel(data: dict[str, Any]) -> dict[str, Any]:
    """The detected regime and what supported it."""
    regime = data.get("regime")
    if not regime:
        return {
            "detected": False,
            "regime": "not yet detected",
            "tone": "warn",
            "message": "No bar has reached regime detection yet.",
            "metrics": [],
            "reasoning": [],
            "indicators": {},
        }

    confidence = regime.get("confidence")
    return {
        "detected": True,
        "regime": regime.get("regime", "UNKNOWN"),
        "tone": "warn" if regime.get("is_uncertain") else "ok",
        "message": None,
        "metrics": [
            _metric("Regime", regime.get("regime", "UNKNOWN")),
            _metric(
                "Confidence",
                None if confidence is None else float(confidence),
                _pct(confidence, 0),
                tone="warn" if (confidence or 0) < 0.5 else "ok",
                help_text="How strongly the detector's evidence agreed. Low confidence is "
                "reported, not smoothed away.",
            ),
            _metric("Volatility", regime.get("volatility", "UNKNOWN")),
            _metric(
                "Trend strength",
                regime.get("trend_strength"),
                _price(regime.get("trend_strength"), 3),
            ),
            _metric(
                "Uncertain",
                bool(regime.get("is_uncertain")),
                "yes" if regime.get("is_uncertain") else "no",
                tone="warn" if regime.get("is_uncertain") else "ok",
            ),
        ],
        "reasoning": list(regime.get("reasoning") or []),
        "indicators": dict(regime.get("metrics") or {}),
    }


def strategy_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per strategy, with why it did or did not contribute.

    ``state`` is the single word that explains the row, derived from the
    signal and its weight together. A strategy that signalled LONG but was
    weighted to zero reads ``SUPPRESSED``, not ``LONG`` - it did not contribute,
    and a table that showed only its direction would imply that it had.
    """
    rows: list[dict[str, Any]] = []
    for row in data.get("strategies") or []:
        weight = row.get("weight")
        if row.get("direction") == "NO_SIGNAL":
            state, tone = "NO_SIGNAL", "error"
        elif row.get("suppressed"):
            state, tone = "SUPPRESSED", "warn"
        elif not row.get("actionable"):
            state, tone = "DECLINED", "neutral"
        else:
            state, tone = "CONTRIBUTING", "ok"

        rows.append(
            {
                "strategy": row.get("strategy"),
                "state": state,
                "tone": tone,
                "direction": row.get("direction"),
                "confidence": row.get("confidence"),
                "confidence_display": _pct(row.get("confidence"), 0),
                "weight": weight,
                "weight_display": "n/a" if weight is None else f"{weight:.2f}",
                "base_weight": row.get("base_weight"),
                "regime_factor": row.get("regime_factor"),
                "performance_factor": row.get("performance_factor"),
                "entry_price": row.get("entry_price"),
                "stop_loss": row.get("stop_loss"),
                "take_profit": row.get("take_profit"),
                "reason": row.get("reason") or "",
                "reasoning": list(row.get("reasoning") or []),
                "weight_reasoning": list(row.get("weight_reasoning") or []),
            }
        )
    return sorted(rows, key=lambda r: (r["state"] != "CONTRIBUTING", str(r["strategy"])))


def decision_panel(data: dict[str, Any]) -> dict[str, Any]:
    """Selection, aggregation, the risk verdict and the final decision."""
    decision = data.get("decision") or {}
    aggregation = data.get("aggregation")
    risk = data.get("risk")
    selection = data.get("selection") or {}
    order = data.get("order")

    steps: list[dict[str, Any]] = []

    suppressed = list(selection.get("suppressed") or [])
    declined = [s for s in (selection.get("rejected") or []) if s not in suppressed]
    steps.append(
        {
            "step": "Selection",
            "outcome": f"{len(data.get('strategies') or []) - len(suppressed)} weighted",
            "tone": "warn" if suppressed else "ok",
            "detail": (
                (f"Suppressed by the selector: {', '.join(suppressed)}. " if suppressed else "")
                + (f"Declined to signal: {', '.join(declined)}." if declined else "")
            ).strip()
            or "Every strategy carried weight and signalled.",
        }
    )

    if aggregation is None:
        steps.append(
            {
                "step": "Aggregation",
                "outcome": "not reached",
                "tone": "neutral",
                "detail": "No bar has reached signal aggregation.",
            }
        )
    else:
        steps.append(
            {
                "step": "Aggregation",
                "outcome": f"{aggregation.get('direction')} "
                f"@ {_pct(aggregation.get('confidence'), 0)}",
                "tone": "ok" if aggregation.get("actionable") else "warn",
                "detail": " ".join(aggregation.get("reasoning") or [])
                or f"Method {aggregation.get('method')}.",
                "contributing": list(aggregation.get("contributing") or []),
                "opposing": list(aggregation.get("opposing") or []),
            }
        )

    if risk is None:
        steps.append(
            {
                "step": "Risk",
                "outcome": "not reached",
                "tone": "neutral",
                "detail": "Risk evaluates only an actionable aggregated signal.",
            }
        )
    else:
        steps.append(
            {
                "step": "Risk",
                "outcome": str(risk.get("verdict")),
                "tone": "ok" if risk.get("approved") else "error",
                "detail": (
                    f"{risk.get('quantity'):.6g} units risking "
                    f"{_money(risk.get('risk_amount'))}."
                    if risk.get("approved")
                    else f"Blocked: {risk.get('block_reason') or risk.get('verdict')}."
                )
                + (
                    " " + " ".join(risk.get("reasoning") or [])
                    if risk.get("reasoning")
                    else ""
                ),
                "block_reason": risk.get("block_reason"),
            }
        )

    steps.append(
        {
            "step": "Execution",
            "outcome": "paper order queued" if order else "no order",
            "tone": "ok" if order else "neutral",
            "detail": (
                f"A paper {order.get('side')} order for {order.get('quantity'):.6g} units "
                "was queued to fill at the next bar's open. No broker is contacted."
                if order
                else "Nothing was queued for the next bar."
            ),
        }
    )

    return {
        "final_decision": data.get("final_decision") or "none",
        "timestamp": decision.get("timestamp"),
        "warm": decision.get("warm"),
        "skipped_reason": decision.get("skipped_reason"),
        "trading_enabled": decision.get("trading_enabled"),
        "steps": steps,
        "risk_blocks": [
            s["block_reason"] for s in steps if s.get("block_reason")
        ],
        "trace": list(decision.get("trace") or []),
        "errors": list(decision.get("errors") or []),
    }


def portfolio_panel(data: dict[str, Any]) -> dict[str, Any]:
    """Cash, equity, P&L, drawdown, exposure and open positions."""
    p = data.get("portfolio") or {}
    currency = p.get("currency", "USD")
    total = p.get("total_pnl")

    return {
        "marked_to_market": bool(p.get("marked_to_market", False)),
        "mark_price": p.get("mark_price"),
        "metrics": [
            _metric("Equity", p.get("equity"), _money(p.get("equity"), currency)),
            _metric("Cash", p.get("cash"), _money(p.get("cash"), currency)),
            _metric(
                "Total P&L",
                total,
                f"{_money(total, currency)} ({_pct(p.get('return_pct'))})",
                tone="ok" if (total or 0) > 0 else ("error" if (total or 0) < 0 else "neutral"),
            ),
            _metric(
                "Realised",
                p.get("realised_pnl"),
                _money(p.get("realised_pnl"), currency),
                help_text="From closed trades only.",
            ),
            _metric(
                "Unrealised",
                p.get("unrealised_pnl"),
                _money(p.get("unrealised_pnl"), currency),
                help_text="Open positions marked at the latest close."
                if p.get("marked_to_market")
                else "No mark price is available, so this is not a measurement.",
            ),
            _metric(
                "Drawdown",
                p.get("drawdown"),
                _pct(p.get("drawdown")),
                tone="error" if (p.get("drawdown") or 0) > 0.1 else "neutral",
                help_text=f"From the equity peak of {_money(p.get('peak_equity'), currency)}. "
                "The peak survives a restart, so a restart cannot widen the limit.",
            ),
            _metric(
                "Daily loss",
                p.get("daily_loss"),
                _pct(p.get("daily_loss")),
                tone="warn" if (p.get("daily_loss") or 0) > 0 else "neutral",
            ),
            _metric(
                "Open risk",
                p.get("open_risk"),
                _money(p.get("open_risk"), currency),
                help_text="What would be lost if every open position hit its stop.",
            ),
            _metric("Exposure", p.get("exposure"), _money(p.get("exposure"), currency)),
            _metric(
                "Costs paid",
                p.get("total_costs"),
                _money(p.get("total_costs"), currency),
                help_text="Spread, slippage and commission on every simulated fill.",
            ),
            _metric("Open positions", p.get("open_positions")),
            _metric("Closed trades", p.get("closed_trades")),
        ],
        "positions": [
            {
                "symbol": pos.get("symbol"),
                "direction": pos.get("direction"),
                "strategy": pos.get("strategy"),
                "quantity": pos.get("quantity"),
                "entry_price": pos.get("entry_price"),
                "entry_time": pos.get("entry_time"),
                "stop_loss": pos.get("stop_loss"),
                "take_profit": pos.get("take_profit"),
                "mark_price": pos.get("mark_price"),
                "unrealised_pnl": pos.get("unrealised_pnl"),
                "unrealised_display": _money(pos.get("unrealised_pnl"), currency),
                "r_multiple": pos.get("r_multiple"),
                "risk_amount": pos.get("risk_amount"),
                "entry_regime": pos.get("entry_regime"),
            }
            for pos in data.get("positions") or []
        ],
        "equity_series": [
            {"timestamp": row.get("timestamp"), "equity": row.get("equity"),
             "drawdown": row.get("drawdown")}
            for row in data.get("equity_curve") or []
        ],
    }


def execution_panel(data: dict[str, Any]) -> dict[str, Any]:
    """Pending orders and recent fills, with each cost component separate.

    Spread, slippage and commission stay in their own columns. Showing only a
    total cost makes it impossible to see which execution assumption is driving
    a result, which is the question a simulated fill exists to answer.
    """
    pending = []
    for order in data.get("pending_orders") or []:
        pending.append(
            {
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "quantity": order.get("quantity"),
                "strategy": order.get("strategy"),
                "stop_loss": order.get("stop_loss"),
                "take_profit": order.get("take_profit"),
                "risk_amount": order.get("risk_amount"),
                "queued_on_bar": order.get("queued_on_bar"),
                "reference_price": order.get("reference_price"),
                "note": "Fills at the next bar's open, at the spread and slippage the "
                "simulator applies.",
            }
        )

    fills = []
    for fill in reversed(data.get("fills") or []):
        reference = fill.get("reference_price")
        price = fill.get("price")
        slip = None
        if reference not in (None, 0) and price is not None:
            slip = float(price) - float(reference)
        fills.append(
            {
                "timestamp": fill.get("timestamp"),
                "symbol": fill.get("symbol"),
                "side": fill.get("side"),
                "status": fill.get("status"),
                "quantity": fill.get("quantity"),
                "price": price,
                "reference_price": reference,
                "price_display": _price(price),
                "difference": slip,
                "difference_display": "n/a" if slip is None else f"{slip:+,.5g}",
                "spread_cost": fill.get("spread_cost"),
                "slippage_cost": fill.get("slippage_cost"),
                "commission": fill.get("commission"),
                "total_cost": fill.get("total_cost"),
                "closes_position": bool(fill.get("closes_position")),
                "exit_reason": fill.get("exit_reason"),
                "strategy": fill.get("strategy"),
                "rejection_reason": fill.get("rejection_reason"),
            }
        )

    return {
        "pending_orders": pending,
        "fills": fills,
        "cost_note": (
            "Every fill is simulated: next-bar open, half the configured spread against "
            "the trader, slippage from the configured model, plus commission. Gaps "
            "through a stop fill at the open, not at the stop."
        ),
    }


def trade_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Recent closed trades, newest first."""
    rows = []
    for trade in reversed(data.get("trades") or []):
        rows.append(
            {
                "exit_time": trade.get("exit_time"),
                "symbol": trade.get("symbol"),
                "strategy": trade.get("strategy"),
                "direction": trade.get("direction"),
                "entry_price": trade.get("entry_price"),
                "exit_price": trade.get("exit_price"),
                "exit_reason": trade.get("exit_reason"),
                "net_pnl": trade.get("net_pnl"),
                "net_display": _money(trade.get("net_pnl")),
                "r_multiple": trade.get("r_multiple"),
                "r_display": (
                    "n/a" if trade.get("r_multiple") is None
                    else f"{trade['r_multiple']:+.2f}R"
                ),
                "bars_held": trade.get("bars_held"),
                "costs": trade.get("costs"),
                "entry_regime": trade.get("entry_regime"),
                "tone": "ok" if (trade.get("net_pnl") or 0) > 0 else "error",
            }
        )
    return rows


# ---------------------------------------------------------------------- graph


def graph_figure_spec(graph: dict[str, Any]) -> dict[str, Any]:
    """Plot-ready coordinates for the decision graph.

    Reads the ``column``/``row``/``colour`` the graph view already assigned, so
    the picture on the page and the structure a test asserts are the same data.
    The y axis is inverted here only - node rows stay as the graph view set them.
    """
    nodes = list(graph.get("nodes") or [])
    positions = {n["id"]: (float(n.get("column", 0)), -float(n.get("row", 0))) for n in nodes}

    plotted = [
        {
            "id": node["id"],
            "x": positions[node["id"]][0],
            "y": positions[node["id"]][1],
            "label": node.get("label", node["id"]),
            "status": node.get("status", "PENDING"),
            "colour": node.get("colour"),
            "detail": node.get("detail", ""),
            "hover": f"{node.get('label', node['id'])}<br>{node.get('status')}<br>"
            f"{node.get('detail', '')}",
        }
        for node in nodes
    ]

    edges = []
    for source, target in graph.get("edges") or []:
        if source in positions and target in positions:
            (x0, y0), (x1, y1) = positions[source], positions[target]
            on_path = source in (graph.get("path") or []) and target in (graph.get("path") or [])
            edges.append(
                {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                 "source": source, "target": target, "on_path": on_path}
            )

    xs = [n["x"] for n in plotted] or [0.0]
    ys = [n["y"] for n in plotted] or [0.0]
    return {
        "nodes": plotted,
        "edges": edges,
        "x_range": [min(xs) - 0.6, max(xs) + 0.6],
        "y_range": [min(ys) - 0.8, max(ys) + 0.8],
        "stage_ticks": [
            {"x": float(index), "label": label} for index, label in _stage_pairs(nodes)
        ],
        "legend": dict(graph.get("legend") or {}),
        "path": list(graph.get("path") or []),
        "summary": _graph_summary(graph),
    }


def _stage_pairs(nodes: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Stage column/label pairs for the stages actually present.

    The column is the stage's index in ``STAGES``, which is what the graph view
    assigns to each node. Re-numbering the present stages from zero would slide
    the axis labels out from under the nodes whenever a stage is missing.
    """
    from app.paper_trading.graph_view import STAGES

    present = {n.get("stage") for n in nodes}
    return [
        (index, label) for index, (stage, label) in enumerate(STAGES) if stage in present
    ]


def _graph_summary(graph: dict[str, Any]) -> str:
    """One sentence describing how far the bar got and what stopped it."""
    path = list(graph.get("path") or [])
    nodes = {n["id"]: n for n in graph.get("nodes") or []}
    if not path:
        first = nodes.get("market_data")
        if first is None:
            return "The graph has no recorded state."
        return f"Nothing progressed past market data: {first.get('detail', '')}"

    last = path[-1]
    blocks = graph.get("risk_blocks") or []
    suppressed = graph.get("suppressed") or []
    failed = graph.get("failed") or []

    parts = [f"The bar reached {nodes.get(last, {}).get('label', last)}."]
    if failed:
        parts.append(f"Failed node(s): {', '.join(failed)}.")
    if suppressed:
        parts.append(f"Suppressed strategies: {', '.join(suppressed)}.")
    if blocks:
        parts.append(f"Risk blocked the trade: {'; '.join(str(b) for b in blocks)}.")
    if graph.get("final_decision"):
        parts.append(f"Final decision: {graph['final_decision']}.")
    return " ".join(parts)


# ----------------------------------------------------------------- whole page


def build_view(data: dict[str, Any]) -> dict[str, Any]:
    """Shape a whole dashboard payload into everything the page renders."""
    graph = data.get("graph") or {}
    return {
        "banners": safety_banners(data),
        "header": header_panel(data),
        "market": market_panel(data),
        "regime": regime_panel(data),
        "strategies": strategy_rows(data),
        "decision": decision_panel(data),
        "portfolio": portfolio_panel(data),
        "execution": execution_panel(data),
        "trades": trade_rows(data),
        "graph": graph,
        "graph_figure": graph_figure_spec(graph),
        "events": list(reversed(data.get("events") or [])),
        "errors": list(reversed(data.get("errors") or [])),
        "safety": data.get("safety") or {},
    }
