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
    "performance_panel",
    "portfolio_panel",
    "quality_rows",
    "regime_panel",
    "research_panel",
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


def _repaired_share(quality: dict[str, Any]) -> float | None:
    """Share of bars cleaning had to repair, from the gate's own stats.

    Read from ``stats`` rather than recomputed, so the number on the page is the
    number the gate graded against.
    """
    stats = quality.get("stats") or {}
    repaired, rows = stats.get("bars_repaired"), quality.get("rows")
    if repaired is None or not rows:
        return None
    try:
        return float(repaired) / float(rows)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _repaired_display(quality: dict[str, Any]) -> str:
    share = _repaired_share(quality)
    if share is None:
        return "not measured"
    repaired = (quality.get("stats") or {}).get("bars_repaired") or 0
    return f"{int(repaired):,} bars ({_pct(share)})"


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
                    "; ".join(
                        str(i.get("message", i)) if isinstance(i, dict) else str(i)
                        for i in (quality.get("issues") or [])[:3]
                    )
                    or "The quality gate recorded no findings for this series."
                ),
            ),
            _metric(
                "Bars repaired",
                _repaired_share(quality),
                _repaired_display(quality),
                tone=(
                    "neutral"
                    if _repaired_share(quality) is None
                    else ("warn" if _repaired_share(quality) > 0 else "ok")
                ),
                help_text="Bars whose high/low bounds were invalid in the vendor feed and "
                "had to be repaired before use. Yahoo's OTC FX composites carry a "
                "measurable share of these; the quality gate's thresholds in "
                "configs/data.yaml decide whether that warns or fails.",
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
        "quality_findings": quality_rows(data),
        "quality_stats": dict(quality.get("stats") or {}),
        "freshness_detail": freshness,
    }


def quality_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The data-quality gate's findings, worst first, as displayable rows.

    The gate's report is the authority on whether a feed is usable, so its
    findings get a table of their own rather than a stringified dict in an
    expander. Each row keeps the finding's severity, how many bars it touched
    and what fraction of the series that is - the FX feeds' OHLC-bound defects
    are a percentage-of-bars problem, and a message without its ratio cannot be
    weighed against the thresholds in configs/data.yaml.
    """
    quality = (data.get("market") or {}).get("quality") or {}
    order = {"FAIL": 0, "WARNING": 1, "PASS": 2}
    rows = []
    for issue in quality.get("issues") or []:
        if not isinstance(issue, dict):
            rows.append({"code": "unknown", "severity": "UNKNOWN", "message": str(issue),
                         "bars": None, "share": None, "share_display": "n/a",
                         "tone": "warn", "samples": []})
            continue
        severity = str(issue.get("severity", "UNKNOWN")).upper()
        ratio = issue.get("ratio")
        rows.append(
            {
                "code": str(issue.get("code", "unknown")),
                "severity": severity,
                "message": str(issue.get("message", "")),
                "bars": issue.get("count"),
                "share": ratio,
                "share_display": "n/a" if ratio is None else _pct(ratio),
                "tone": _tone(severity),
                "samples": list(issue.get("samples") or []),
            }
        )
    return sorted(rows, key=lambda r: (order.get(r["severity"], 3), r["code"]))


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


def _profit_factor_display(factor: float | None, trades: int | None) -> str:
    """Format a profit factor, distinguishing "no losses" from "no trades".

    ``build_performance`` returns ``None`` for both, because dividing by zero
    losses has no answer either way. Rendering both as "no losses in sample"
    told a reader with an empty book that trades had happened and none had
    lost. An unmeasured quantity has to read as unmeasured.
    """
    if factor is not None:
        return f"{factor:.2f}"
    return "n/a - no closed trades" if not trades else "no losses in sample"


def performance_panel(data: dict[str, Any]) -> dict[str, Any]:
    """Realised performance by strategy and by regime.

    Rows come straight from :func:`app.paper_trading.performance.build_performance`,
    which computes them from closed trades only. The panel's job is formatting
    and one piece of emphasis: a row whose sample is below the selector's
    evidence threshold is marked, so a mean R from three trades cannot be read
    beside one from ninety as though they carried the same weight.
    """
    perf = data.get("performance") or {}
    overall = perf.get("overall") or {}
    currency = (data.get("portfolio") or {}).get("currency", "USD")

    def row(source: dict[str, Any], name_key: str) -> dict[str, Any]:
        expectancy = source.get("expectancy_r")
        factor = source.get("profit_factor")
        return {
            name_key: source.get(name_key),
            "trades": source.get("trades"),
            "wins": source.get("wins"),
            "losses": source.get("losses"),
            "win_rate": source.get("win_rate"),
            "win_rate_display": _pct(source.get("win_rate"), 0),
            "expectancy_r": expectancy,
            "expectancy_display": "n/a" if expectancy is None else f"{expectancy:+.2f}R",
            "total_r": source.get("total_r"),
            "net_pnl": source.get("net_pnl"),
            "net_display": _money(source.get("net_pnl"), currency),
            "profit_factor": factor,
            "profit_factor_display": _profit_factor_display(factor, source.get("trades")),
            "costs": source.get("costs"),
            "avg_bars_held": source.get("avg_bars_held"),
            "ambiguous_exits": source.get("ambiguous_exits"),
            "insufficient_evidence": bool(source.get("insufficient_evidence")),
            "evidence_display": (
                f"sample only (<{source.get('min_samples')})"
                if source.get("insufficient_evidence")
                else "meets threshold"
            ),
            "verdict": source.get("verdict"),
            "tone": "warn" if source.get("insufficient_evidence") else "neutral",
        }

    pair_rows = []
    for item in perf.get("by_strategy_regime") or []:
        merged = row(item, "strategy")
        merged["regime"] = item.get("regime")
        pair_rows.append(merged)

    selector_rows = []
    for item in perf.get("selector_table") or []:
        mean_r = item.get("mean_r")
        selector_rows.append(
            {
                "strategy": item.get("strategy"),
                "regime": item.get("regime"),
                "samples": item.get("samples"),
                "mean_r": mean_r,
                "mean_r_display": "n/a" if mean_r is None else f"{mean_r:+.2f}R",
                "win_rate_display": _pct(item.get("win_rate"), 0),
                "influences_weight": bool(item.get("influences_weight")),
                "influence_display": (
                    "moves the weight"
                    if item.get("influences_weight")
                    else f"below {item.get('min_samples')} samples - weight unaffected"
                ),
                "tone": "ok" if item.get("influences_weight") else "neutral",
            }
        )

    has_trades = bool(overall.get("trades"))
    return {
        "has_trades": has_trades,
        "basis": perf.get("basis"),
        "verdict": overall.get("verdict")
        or "No closed trade yet, so there is nothing measured.",
        "metrics": [
            _metric("Closed trades", overall.get("trades") or 0),
            _metric(
                "Win rate",
                overall.get("win_rate"),
                _pct(overall.get("win_rate"), 0),
                help_text="Share of closed trades with positive net P&L. A high win rate "
                "does not imply profitability and is not a target.",
            ),
            _metric(
                "Expectancy",
                overall.get("expectancy_r"),
                "n/a"
                if overall.get("expectancy_r") is None
                else f"{overall['expectancy_r']:+.2f}R",
                tone="warn" if overall.get("insufficient_evidence") else "neutral",
                help_text="Mean R multiple across closed trades, the only per-trade figure "
                "comparable across instruments and position sizes.",
            ),
            _metric(
                "Profit factor",
                overall.get("profit_factor"),
                _profit_factor_display(
                    overall.get("profit_factor"), overall.get("trades")
                ),
            ),
            _metric(
                "Net realised",
                overall.get("net_pnl"),
                _money(overall.get("net_pnl"), currency),
                help_text="From closed trades only, net of simulated costs.",
            ),
            _metric(
                "Costs paid",
                overall.get("costs"),
                _money(overall.get("costs"), currency),
            ),
            _metric(
                "Still open",
                perf.get("open_positions") or 0,
                help_text="Excluded from every figure here: an open position has a mark, "
                "not an outcome.",
            ),
            _metric(
                "Ambiguous exits",
                overall.get("ambiguous_exits") or 0,
                help_text="Trades closed on a bar whose range held both the stop and the "
                "target. OHLC cannot say which came first; execution."
                "same_bar_resolution decided.",
            ),
        ],
        "by_strategy": [row(item, "strategy") for item in perf.get("by_strategy") or []],
        "by_contributor": [
            row(item, "strategy") for item in perf.get("by_contributor") or []
        ],
        "contributor_note": perf.get("contributor_note"),
        "by_regime": [row(item, "regime") for item in perf.get("by_regime") or []],
        "by_strategy_regime": pair_rows,
        "by_exit_reason": list(perf.get("by_exit_reason") or []),
        "selector_table": selector_rows,
        "min_samples": perf.get("min_samples"),
    }


def research_panel(data: dict[str, Any]) -> dict[str, Any]:
    """What validation says about the configuration this session is running.

    The hardest requirement here is the negative one: when there is no study, or
    the study describes a different configuration, the panel must say so in
    words and show nothing else. A research section that silently rendered
    zeros would read as "tested, found nothing", which is a different and much
    stronger claim than "never tested".

    The second is that in-sample, validation and out-of-sample numbers are kept
    in labelled, separate rows and never summed, averaged or headlined together.
    They are the same arithmetic on different bars, and only the label
    distinguishes them.
    """
    context = data.get("research") or {}
    availability = str(context.get("availability", "MISSING")).upper()
    available = availability == "AVAILABLE"
    report = context.get("report") or {}

    panel: dict[str, Any] = {
        "available": available,
        "availability": availability,
        "reason": context.get("reason", "No research context was supplied."),
        "tone": {"AVAILABLE": "ok", "MISSING": "warn", "STALE": "warn", "UNREADABLE": "error"}
        .get(availability, "warn"),
        "experiment_id": context.get("experiment_id"),
        "report_path": context.get("report_path"),
        "other_experiments": list(context.get("other_experiments") or []),
        "evidence_ladder": list(context.get("evidence_ladder") or []),
        "segments": [],
        "verdict": None,
        "verdict_reasons": [],
        "walk_forward": None,
        "monte_carlo": None,
        "robustness": None,
        "cost_stress": None,
        "correlation": None,
        "overfitting": None,
        "recommendations": [],
        "recommendation_note": None,
        "evidence_summary": [],
        "provenance": {},
        "metrics": [],
    }

    if not available:
        # Deliberately returns here. Every field above is empty, and the caller
        # renders the reason instead of a table of nothing.
        return panel

    panel["evidence_summary"] = [
        {
            "category": row.get("category"),
            "status": row.get("status"),
            "detail": row.get("detail"),
            "limitation": row.get("limitation"),
            "tone": {
                "NOT_ESTABLISHED": "error",
                "NOT_ASSESSED": "warn",
                "PARTIAL": "warn",
                "SEPARATE QUESTION": "info",
            }.get(str(row.get("status")), "neutral"),
        }
        for row in report.get("evidence_summary") or []
    ]

    verdict = report.get("verdict")
    panel["verdict"] = verdict
    panel["verdict_reasons"] = list(report.get("verdict_reasons") or [])
    panel["verdict_tone"] = {
        "SURVIVED THIS ROUND OF TESTING": "ok",
        "INCONCLUSIVE": "warn",
        "INSUFFICIENT EVIDENCE": "warn",
        "EVIDENCE AGAINST": "error",
    }.get(str(verdict), "neutral")

    # --- windows, each labelled with what it can support ---------------------
    supports = {
        "in_sample": "describes the bars the configuration was built from - no evidence",
        "validation": "one held-out window, consulted during development - weak",
        "out_of_sample": "seen once, never used for a decision - the evidence that counts",
    }
    for row in report.get("segments") or []:
        name = str(row.get("segment"))
        panel["segments"].append(
            {
                "segment": name,
                "role": row.get("role"),
                "supports": supports.get(name, "an additional window"),
                "is_out_of_sample": name == "out_of_sample",
                "bars": row.get("bars"),
                "trades": row.get("trades"),
                "total_return": row.get("total_return"),
                "return_display": _pct(row.get("total_return")),
                "expectancy_r": row.get("expectancy_r"),
                "expectancy_display": (
                    "n/a"
                    if row.get("expectancy_r") is None
                    else f"{row['expectancy_r']:+.3f}R"
                ),
                "max_drawdown": row.get("max_drawdown"),
                "drawdown_display": _pct(row.get("max_drawdown")),
                "sharpe": row.get("sharpe"),
                "win_rate_display": _pct(row.get("win_rate"), 0),
                "data_quality": row.get("data_quality"),
                "quality_tone": _tone(row.get("data_quality")),
            }
        )

    oos = next((s for s in panel["segments"] if s["is_out_of_sample"]), None)

    wf = report.get("walk_forward")
    if wf:
        folds = wf.get("folds") or 0
        panel["walk_forward"] = {
            "folds": folds,
            "profitable_folds": wf.get("profitable_folds"),
            "profitable_display": (
                f"{wf.get('profitable_folds', 0)} of {folds}" if folds else "no folds"
            ),
            "efficiency": wf.get("efficiency"),
            "stitched": wf.get("stitched") or {},
            "warnings": list(wf.get("warnings") or []),
        }

    mc = report.get("monte_carlo")
    if mc:
        panel["monte_carlo"] = mc

    robustness = report.get("robustness")
    if robustness:
        fragile = list(robustness.get("fragile_parameters") or [])
        panel["robustness"] = {
            "objective": robustness.get("objective_name"),
            "measured_on": robustness.get("segment"),
            "fragile_parameters": fragile,
            "rows": [
                {
                    "parameter": s.get("parameter"),
                    "fragile": bool(s.get("fragile")),
                    "baseline_is_peak": bool(s.get("baseline_is_peak")),
                    "spread": s.get("coefficient_of_variation"),
                    "spread_display": (
                        "n/a"
                        if s.get("coefficient_of_variation") is None
                        else f"{s['coefficient_of_variation']:.2f}"
                    ),
                    "verdict": "FRAGILE" if s.get("fragile") else "stable",
                    "tone": "error" if s.get("fragile") else "ok",
                }
                for s in robustness.get("sensitivities") or []
            ],
        }

    cost_stress = report.get("cost_stress")
    if cost_stress:
        lo, hi = ((cost_stress.get("cost_drag_range") or []) + [None, None])[:2]
        status = str(cost_stress.get("survival_status") or "NOT_ASSESSED")
        panel["cost_stress"] = {
            "measured_on": cost_stress.get("segment"),
            "survival_status": status,
            "survives": bool(cost_stress.get("survives")),
            "survival_detail": cost_stress.get("survival_detail"),
            # A losing baseline and a fragile edge are both bad news, but only
            # one of them is "the edge died". Colour them apart, and never green
            # unless an edge actually held.
            "tone": {
                "SURVIVED": "ok",
                "DID_NOT_SURVIVE": "error",
                "NO_BASELINE_EDGE": "warn",
                "NOT_ASSESSED": "warn",
            }.get(status, "neutral"),
            "baseline_expectancy_r": cost_stress.get("baseline_expectancy_r"),
            "baseline_display": (
                "n/a"
                if cost_stress.get("baseline_expectancy_r") is None
                else f"{cost_stress['baseline_expectancy_r']:+.3f}R"
            ),
            "cost_drag_range": [lo, hi],
            "cost_drag_display": (
                "n/a" if lo is None or hi is None else f"{_pct(lo)} to {_pct(hi)}"
            ),
            "rows": [
                {
                    "scenario": s.get("scenario"),
                    "is_baseline": bool(s.get("is_baseline")),
                    "trades": s.get("trades"),
                    "expectancy_r": s.get("expectancy_r"),
                    "expectancy_display": (
                        "error"
                        if s.get("error")
                        else "n/a"
                        if s.get("expectancy_r") is None
                        else f"{s['expectancy_r']:+.3f}R"
                    ),
                    "return_display": _pct(s.get("total_return")),
                    "cost_drag_display": _pct(s.get("cost_drag")),
                    "error": s.get("error"),
                }
                for s in cost_stress.get("scenarios") or []
            ],
        }

    correlation = report.get("correlation")
    if correlation:
        panel["correlation"] = {
            "threshold": correlation.get("threshold"),
            "measured_on": correlation.get("segment"),
            "trades_per_strategy": dict(correlation.get("trades_per_strategy") or {}),
            "findings": [
                {
                    "pair": f"{f.get('strategy_a')} / {f.get('strategy_b')}",
                    "return_correlation": f.get("return_correlation"),
                    "correlation_display": (
                        "n/a"
                        if f.get("return_correlation") is None
                        else f"{f['return_correlation']:+.2f}"
                    ),
                    "signal_agreement": f.get("signal_agreement"),
                    "agreement_display": _pct(f.get("signal_agreement"), 0),
                    "observations": f.get("observations"),
                    "hypothesis": f.get("hypothesis"),
                }
                for f in correlation.get("findings") or []
            ],
        }

    overfitting = report.get("overfitting")
    if overfitting:
        findings = overfitting.get("findings") or []
        panel["overfitting"] = {
            "trials": overfitting.get("trials"),
            "deflated_sharpe": overfitting.get("deflated_sharpe"),
            "findings": [
                {
                    "code": f.get("code"),
                    "severity": str(f.get("severity", "")).upper(),
                    "message": f.get("message"),
                    "tone": {"SEVERE": "error", "WARNING": "warn"}.get(
                        str(f.get("severity", "")).upper(), "neutral"
                    ),
                }
                for f in findings
            ],
            "severe_count": sum(
                1 for f in findings if str(f.get("severity", "")).upper() == "SEVERE"
            ),
        }

    recommendations = report.get("recommendations") or {}
    panel["recommendations"] = [
        {
            "subject": r.get("subject"),
            "action": r.get("action"),
            "evidence_tier": r.get("evidence_tier"),
            "strategy": r.get("strategy"),
            "proposed_weight": r.get("proposed_weight"),
            "applicable": bool(r.get("applicable")),
            "rationale": list(r.get("rationale") or []),
            "blockers": list(r.get("blockers") or []),
            "tone": "ok" if r.get("applicable") else "neutral",
        }
        for r in recommendations.get("recommendations") or []
    ]
    panel["recommendation_note"] = (
        "A recommendation is a reading of what was measured. None of these has been "
        "applied to the running configuration: research.feedback.enabled governs that "
        "and ships false."
    )

    panel["provenance"] = {
        "experiment_id": report.get("experiment_id"),
        "created_at": report.get("created_at"),
        "period": f"{report.get('period_start')} -> {report.get('period_end')}",
        "bars": report.get("bars"),
        "data_provider": report.get("data_provider"),
        "data_checksum": report.get("data_checksum"),
        "data_quality": report.get("data_quality"),
        "git_revision": report.get("git_revision"),
        "random_seed": report.get("random_seed"),
        "config_fingerprint": (report.get("spec") or {}).get("config_fingerprint"),
        "objective": (report.get("spec") or {}).get("objective"),
    }

    panel["metrics"] = [
        _metric(
            "Verdict",
            verdict,
            str(verdict or "unknown"),
            tone=panel["verdict_tone"],
            help_text="No verdict means profitable. The best available is 'survived this "
            "round of testing'.",
        ),
        _metric(
            "OOS trades",
            (oos or {}).get("trades"),
            f"{(oos or {}).get('trades', 0):,}",
            tone="warn" if ((oos or {}).get("trades") or 0) < 30 else "ok",
            help_text="Below roughly 30 the out-of-sample statistics are not stable "
            "enough to support a conclusion.",
        ),
        _metric(
            "OOS expectancy",
            (oos or {}).get("expectancy_r"),
            (oos or {}).get("expectancy_display", "n/a"),
            tone="error" if ((oos or {}).get("expectancy_r") or 0) <= 0 else "neutral",
            help_text="Mean R per trade on bars the configuration had never seen.",
        ),
        _metric(
            "Walk-forward",
            (panel["walk_forward"] or {}).get("profitable_folds"),
            (panel["walk_forward"] or {}).get("profitable_display", "not run"),
            help_text="Folds with positive expectancy, out of folds run.",
        ),
        _metric(
            "Fragile parameters",
            len((panel["robustness"] or {}).get("fragile_parameters") or []),
            ", ".join((panel["robustness"] or {}).get("fragile_parameters") or []) or "none",
            tone="error"
            if (panel["robustness"] or {}).get("fragile_parameters")
            else "ok",
            help_text="Parameters whose neighbourhood is not flat - a sharp peak at the "
            "shipped value describes the sample rather than the market.",
        ),
        _metric(
            "Severe overfitting findings",
            (panel["overfitting"] or {}).get("severe_count", 0),
            str((panel["overfitting"] or {}).get("severe_count", 0)),
            tone="error" if (panel["overfitting"] or {}).get("severe_count") else "ok",
        ),
    ]
    return panel


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
        "stopped_at": graph.get("stopped_at"),
        "stopped_reason": graph.get("stopped_reason"),
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
    """One sentence describing how far the bar got and what stopped it.

    Reads ``path``/``stopped_at`` rather than node colours. A bar can complete
    every stage on data the quality gate has caveated; saying it "did not
    progress" because the feed is stale would be false, and the caveat has its
    own banner.
    """
    nodes = {n["id"]: n for n in graph.get("nodes") or []}
    if not nodes:
        return "The graph has no recorded state."

    path = list(graph.get("path") or [])
    stopped_at = graph.get("stopped_at")

    def label(node_id: Any) -> str:
        return str(nodes.get(str(node_id), {}).get("label", node_id)).replace("\n", " ")

    parts: list[str] = []
    if not path:
        detail = graph.get("stopped_reason") or nodes.get("market_data", {}).get("detail", "")
        parts.append(f"The pipeline did not start: {detail}".strip())
    elif stopped_at:
        parts.append(f"The bar reached {label(stopped_at)} and stopped there.")
        if graph.get("stopped_reason"):
            parts.append(str(graph["stopped_reason"]))
    else:
        parts.append(f"The bar completed every stage through to {label(path[-1])}.")

    if graph.get("failed"):
        parts.append(f"Failed node(s): {', '.join(graph['failed'])}.")
    if graph.get("suppressed"):
        parts.append(f"Suppressed strategies: {', '.join(graph['suppressed'])}.")
    if graph.get("rejected"):
        parts.append(f"Strategies that declined: {', '.join(graph['rejected'])}.")
    if graph.get("risk_blocks"):
        parts.append(
            f"Risk blocked the trade: {'; '.join(str(b) for b in graph['risk_blocks'])}."
        )
    if graph.get("final_decision"):
        parts.append(f"Final decision: {graph['final_decision']}.")
    return " ".join(p for p in parts if p)


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
        "performance": performance_panel(data),
        "research": research_panel(data),
        "quality_findings": quality_rows(data),
        "trades": trade_rows(data),
        "graph": graph,
        "graph_figure": graph_figure_spec(graph),
        "events": list(reversed(data.get("events") or [])),
        "errors": list(reversed(data.get("errors") or [])),
        "safety": data.get("safety") or {},
    }
