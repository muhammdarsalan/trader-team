"""Decision-graph visualisation data.

Turns one bar's decision record into a node/edge graph a reader can look at and
answer: which stages ran, which strategies were suppressed or rejected, whether
risk blocked the trade, and what the final decision was.

Two rules shape this module.

First, **status is derived, never assumed.** The previous version of this code
labelled every node ``"ok"`` before looking at any state, so a dashboard opened
against an engine that had never processed a bar displayed five green active
nodes. A node here is ``PENDING`` until something recorded says otherwise.

Second, **layout is data.** Node coordinates are computed here rather than in
the Streamlit layer, so the arrangement of the graph can be asserted in a test
without a browser or a rendering runtime.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "NODE_STATUS_COLOURS",
    "STAGES",
    "build_graph_visualization",
]

#: Pipeline stages, in order. Strategy nodes fan out inside ``strategies``.
STAGES: tuple[tuple[str, str], ...] = (
    ("market_data", "Market Data"),
    ("features", "Features"),
    ("regime", "Regime"),
    ("strategies", "Strategies"),
    ("selection", "Selection"),
    ("aggregation", "Aggregation"),
    ("risk", "Risk"),
    ("execution", "Execution"),
    ("paper_position", "Paper Position"),
)

#: Status vocabulary and the colour each maps to. Kept next to the statuses so a
#: new status cannot be introduced without deciding how it reads.
NODE_STATUS_COLOURS: dict[str, str] = {
    "ACTIVE": "#1a7f37",  # on the path to a paper trade
    "OK": "#0969da",  # ran and produced a result
    "WAIT": "#9a6700",  # ran, declined to act
    "WARMUP": "#57606a",  # not enough history yet
    "SUPPRESSED": "#8250df",  # weighted to zero by the selector
    "REJECTED": "#57606a",  # produced no actionable signal
    "BLOCKED": "#cf222e",  # refused by risk
    "ERROR": "#a40e26",  # the node failed
    "PENDING": "#8c959f",  # never ran
}

_ACTIONABLE = {"LONG", "SHORT"}


def build_graph_visualization(
    source: Any = None,
    *,
    market: dict[str, Any] | None = None,
    open_positions: int | None = None,
    trading_enabled: bool | None = None,
) -> dict[str, Any]:
    """Build the graph view for one decision.

    Args:
        source: a decision view from
            :func:`app.paper_trading.records.build_decision_view`, or a raw
            :class:`~app.graph.state.TradingState`-shaped mapping. Both are
            accepted so a caller holding live graph output does not have to
            convert first. ``None`` yields an all-``PENDING`` graph.
        market: the market summary, used to grade the ``market_data`` node on
            row count, quality and freshness rather than on assumption.
        open_positions: number of open paper positions, for the terminal node.
        trading_enabled: the kill-switch position, recorded on the execution
            node so a reader can tell "no signal" apart from "execution off".

    Returns:
        A JSON-safe dict with ``nodes``, ``edges``, ``active_nodes``,
        ``suppressed``, ``rejected``, ``risk_blocks``, ``failed``, ``path``,
        ``final_decision`` and ``legend``.
    """
    view = _coerce(source)

    strategies = view["strategies"]
    failed = set(view["failed"])
    risk = view["risk"]
    aggregation = view["aggregation"]
    regime = view["regime"]
    order = view["order"]
    warm = view["warm"]

    nodes: list[dict[str, Any]] = []
    nodes.append(_market_node(market, failed))
    nodes.append(_features_node(warm, view.get("skipped_reason"), failed))
    nodes.append(_regime_node(regime, failed))
    nodes.extend(_strategy_nodes(strategies, failed))
    nodes.append(_selection_node(strategies, failed))
    nodes.append(_aggregation_node(aggregation, failed))
    nodes.append(_risk_node(risk, failed))
    nodes.append(_execution_node(order, risk, trading_enabled, failed))
    nodes.append(_position_node(open_positions, order))

    edges = _edges(nodes)
    _assign_layout(nodes)

    by_id = {node["id"]: node for node in nodes}
    active = sorted(nid for nid, node in by_id.items() if node["status"] == "ACTIVE")

    risk_blocks: list[str] = []
    if risk is not None and not risk["approved"]:
        risk_blocks.append(risk["block_reason"] or risk["verdict"])

    return {
        "nodes": nodes,
        "edges": edges,
        "active_nodes": active,
        "suppressed": sorted(s["strategy"] for s in strategies if s["suppressed"]),
        "rejected": sorted(s["strategy"] for s in strategies if not s["actionable"]),
        "risk_blocks": risk_blocks,
        "failed": sorted(failed),
        "path": _path(by_id),
        "final_decision": view["decision"],
        "timestamp": view["timestamp"],
        "legend": dict(NODE_STATUS_COLOURS),
    }


# ------------------------------------------------------------------- coercion


def _coerce(source: Any) -> dict[str, Any]:
    """Normalise any accepted input into the fields this module reads.

    Handles the canonical decision view, a live ``TradingState`` holding real
    objects, and the older flattened summary whose signals are bare direction
    strings. Being permissive here is deliberate: the alternative is a
    visualisation that raises ``AttributeError`` on a shape it was not expecting
    and takes the monitoring page down with it.
    """
    empty = {
        "strategies": [],
        "failed": [],
        "risk": None,
        "aggregation": None,
        "regime": None,
        "order": None,
        "decision": None,
        "timestamp": None,
        "warm": None,
        "skipped_reason": None,
    }
    if not source or not isinstance(source, dict):
        return empty

    # Already canonical.
    if isinstance(source.get("strategies"), list):
        return {**empty, **{k: source.get(k, empty[k]) for k in empty}}

    signals = source.get("strategy_signals") or source.get("signals") or {}
    weights = source.get("strategy_weights") or {}

    strategies = []
    for name in sorted(set(signals) | set(weights)):
        strategies.append(_coerce_strategy(name, signals.get(name), weights.get(name)))

    errors = source.get("errors") or []
    failed = sorted(
        {str(e.get("node")) for e in errors if isinstance(e, dict) and e.get("node")}
    )

    return {
        "strategies": strategies,
        "failed": failed,
        "risk": _coerce_risk(source.get("risk_decision")),
        "aggregation": _coerce_aggregation(source.get("aggregated")),
        "regime": _coerce_regime(source.get("regime"), source.get("regime_confidence")),
        "order": source.get("order"),
        "decision": source.get("decision"),
        "timestamp": _as_str(source.get("timestamp")),
        "warm": source.get("warm"),
        "skipped_reason": source.get("skipped_reason"),
    }


def _coerce_strategy(name: str, signal: Any, weight: Any) -> dict[str, Any]:
    if signal is None:
        direction, confidence, actionable, reason = "NO_SIGNAL", 0.0, False, ""
    elif isinstance(signal, str):
        # Flattened summary: only the direction survived.
        direction = signal.upper()
        confidence = 0.0
        actionable = direction in _ACTIONABLE
        reason = ""
    else:
        direction = str(getattr(signal, "direction", "UNKNOWN"))
        confidence = _as_float(getattr(signal, "confidence", 0.0)) or 0.0
        actionable = bool(getattr(signal, "is_actionable", direction in _ACTIONABLE))
        reasoning = list(getattr(signal, "reasoning", ()) or ())
        reason = reasoning[0] if reasoning else ""

    weight_value = None
    if weight is not None:
        weight_value = _as_float(getattr(weight, "weight", weight))

    return {
        "strategy": name,
        "direction": direction,
        "confidence": confidence,
        "actionable": actionable,
        "reason": reason,
        "weight": weight_value,
        "suppressed": weight_value is not None and weight_value <= 0.0,
    }


def _coerce_risk(risk: Any) -> dict[str, Any] | None:
    if risk is None:
        return None
    if isinstance(risk, dict):
        return {
            "verdict": _as_str(risk.get("verdict")) or "UNKNOWN",
            "approved": bool(risk.get("approved")),
            "block_reason": _as_str(risk.get("block_reason")),
            "quantity": _as_float(risk.get("quantity")),
            "risk_amount": _as_float(risk.get("risk_amount")),
            "reasoning": list(risk.get("reasoning") or ()),
        }
    if isinstance(risk, str):
        # The flattened summary carried only the verdict string.
        return {
            "verdict": risk,
            "approved": risk.upper() == "APPROVED",
            "block_reason": None,
            "quantity": None,
            "risk_amount": None,
            "reasoning": [],
        }
    block = getattr(risk, "block_reason", None)
    return {
        "verdict": _as_str(getattr(risk, "verdict", None)) or "UNKNOWN",
        "approved": bool(getattr(risk, "approved", False)),
        "block_reason": None if block is None else str(block),
        "quantity": _as_float(getattr(risk, "quantity", None)),
        "risk_amount": _as_float(getattr(risk, "risk_amount", None)),
        "reasoning": list(getattr(risk, "reasoning", ()) or ()),
    }


def _coerce_aggregation(aggregated: Any) -> dict[str, Any] | None:
    if aggregated is None:
        return None
    if isinstance(aggregated, dict):
        direction = _as_str(aggregated.get("direction")) or "UNKNOWN"
        return {
            "direction": direction,
            "confidence": _as_float(aggregated.get("confidence")) or 0.0,
            "actionable": bool(aggregated.get("actionable", direction in _ACTIONABLE)),
            "contributing": list(aggregated.get("contributing") or ()),
            "opposing": list(aggregated.get("opposing") or ()),
        }
    if isinstance(aggregated, str):
        return {
            "direction": aggregated.upper(),
            "confidence": 0.0,
            "actionable": aggregated.upper() in _ACTIONABLE,
            "contributing": [],
            "opposing": [],
        }
    direction = str(getattr(aggregated, "direction", "UNKNOWN"))
    return {
        "direction": direction,
        "confidence": _as_float(getattr(aggregated, "confidence", 0.0)) or 0.0,
        "actionable": bool(getattr(aggregated, "is_actionable", direction in _ACTIONABLE)),
        "contributing": list(getattr(aggregated, "contributing", ()) or ()),
        "opposing": list(getattr(aggregated, "opposing", ()) or ()),
    }


def _coerce_regime(regime: Any, confidence: Any = None) -> dict[str, Any] | None:
    if regime is None:
        return None
    if isinstance(regime, dict):
        return {
            "regime": _as_str(regime.get("regime")) or "UNKNOWN",
            "confidence": _as_float(regime.get("confidence")),
            "reasoning": list(regime.get("reasoning") or ()),
        }
    if isinstance(regime, str):
        return {"regime": regime, "confidence": _as_float(confidence), "reasoning": []}
    return {
        "regime": str(getattr(regime, "regime", "UNKNOWN")),
        "confidence": _as_float(getattr(regime, "confidence", confidence)),
        "reasoning": list(getattr(regime, "reasoning", ()) or ()),
    }


def _as_float(value: Any) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)


# ----------------------------------------------------------------------- nodes


def _node(
    node_id: str,
    label: str,
    stage: str,
    status: str,
    detail: str,
    **metrics: Any,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "stage": stage,
        "status": status,
        "detail": detail,
        "colour": NODE_STATUS_COLOURS.get(status, NODE_STATUS_COLOURS["PENDING"]),
        "metrics": {k: v for k, v in metrics.items() if v is not None},
    }


def _market_node(market: dict[str, Any] | None, failed: set[str]) -> dict[str, Any]:
    if "market_data" in failed:
        return _node("market_data", "Market Data", "market_data", "ERROR",
                     "The market-data node failed for this bar.")
    if not market or market.get("status") == "NO_DATA":
        return _node(
            "market_data", "Market Data", "market_data", "PENDING",
            "No market data has been loaded, so nothing downstream has run.",
        )
    if market.get("status") == "ERROR":
        return _node(
            "market_data", "Market Data", "market_data", "ERROR",
            str(market.get("error") or "The market-data refresh failed."),
        )

    quality = str(market.get("quality_status", "UNKNOWN"))
    freshness = str((market.get("freshness") or {}).get("status", "UNKNOWN"))
    rows = market.get("rows") or 0
    status = "OK"
    if quality in {"FAIL", "UNKNOWN"} or freshness in {"STALE", "UNKNOWN"}:
        status = "WAIT"
    detail = (
        f"{rows:,} bars from {market.get('provider', 'unknown')}, "
        f"quality {quality}, freshness {freshness}."
    )
    if market.get("is_proxy"):
        detail += " Series is a labelled proxy, not the instrument itself."
    return _node(
        "market_data", "Market Data", "market_data", status, detail,
        rows=rows, quality=quality, freshness=freshness,
        provider=market.get("provider"), last_bar=market.get("last_bar"),
    )


def _features_node(
    warm: bool | None, skipped_reason: str | None, failed: set[str]
) -> dict[str, Any]:
    if "features" in failed:
        return _node("features", "Features", "features", "ERROR",
                     "Feature computation failed for this bar.")
    if warm is None:
        return _node("features", "Features", "features", "PENDING",
                     "No bar has been analysed yet.")
    if not warm:
        return _node(
            "features", "Features", "features", "WARMUP",
            skipped_reason
            or "The bar is inside the feature warm-up period, so no decision was taken.",
        )
    return _node("features", "Features", "features", "OK",
                 "Causal features computed and past warm-up.")


def _regime_node(regime: dict[str, Any] | None, failed: set[str]) -> dict[str, Any]:
    if "regime" in failed:
        return _node("regime", "Regime", "regime", "ERROR",
                     "Regime detection failed for this bar.")
    if regime is None:
        return _node("regime", "Regime", "regime", "PENDING", "Regime not yet detected.")
    confidence = regime.get("confidence")
    suffix = "" if confidence is None else f" at {confidence:.0%} confidence"
    label = regime["regime"]
    status = "WAIT" if label.upper().endswith("UNCERTAIN") else "OK"
    return _node(
        "regime", f"Regime\n{label}", "regime", status,
        f"Detected {label}{suffix}.",
        regime=label, confidence=confidence,
    )


def _strategy_nodes(
    strategies: list[dict[str, Any]], failed: set[str]
) -> list[dict[str, Any]]:
    if not strategies:
        return [
            _node("strategies", "Strategies", "strategies", "PENDING",
                  "No strategy has reported yet.")
        ]

    nodes = []
    for row in strategies:
        name = row["strategy"]
        node_id = f"strategy_{name}"
        weight = row.get("weight")
        weight_text = "unweighted" if weight is None else f"weight {weight:.2f}"

        if name in failed or node_id in failed or row["direction"] == "NO_SIGNAL":
            status = "ERROR"
            detail = f"{name} produced no signal for this bar; the node did not report."
        elif row["suppressed"]:
            status = "SUPPRESSED"
            detail = (
                f"{name} said {row['direction']} but the selector weighted it to zero, "
                "so it cannot contribute."
            )
        elif not row["actionable"]:
            status = "REJECTED"
            detail = f"{name} declined to act ({row['direction']}). {row['reason']}".strip()
        else:
            status = "ACTIVE"
            detail = (
                f"{name} signalled {row['direction']} at {row['confidence']:.0%} "
                f"confidence, {weight_text}."
            )
        nodes.append(
            _node(
                node_id, f"{name}\n{row['direction']}", "strategies", status, detail,
                direction=row["direction"], confidence=row["confidence"], weight=weight,
            )
        )
    return nodes


def _selection_node(
    strategies: list[dict[str, Any]], failed: set[str]
) -> dict[str, Any]:
    if "selection" in failed:
        return _node("selection", "Selection", "selection", "ERROR",
                     "Strategy selection failed for this bar.")
    weighted = [s for s in strategies if s.get("weight") is not None]
    if not weighted:
        return _node("selection", "Selection", "selection", "PENDING",
                     "The selector has not weighted any strategy yet.")
    live = [s for s in weighted if not s["suppressed"]]
    status = "OK" if live else "WAIT"
    detail = (
        f"{len(live)} of {len(weighted)} strategies carry weight; "
        f"{len(weighted) - len(live)} suppressed."
    )
    return _node("selection", "Selection", "selection", status, detail,
                 weighted=len(weighted), live=len(live))


def _aggregation_node(
    aggregation: dict[str, Any] | None, failed: set[str]
) -> dict[str, Any]:
    if "aggregation" in failed or "strategy_aggregation" in failed:
        return _node("aggregation", "Aggregation", "aggregation", "ERROR",
                     "Signal aggregation failed for this bar.")
    if aggregation is None:
        return _node("aggregation", "Aggregation", "aggregation", "PENDING",
                     "No aggregated decision yet.")
    status = "ACTIVE" if aggregation["actionable"] else "WAIT"
    contributing = ", ".join(aggregation["contributing"]) or "none"
    detail = (
        f"Aggregated to {aggregation['direction']} at "
        f"{aggregation['confidence']:.0%} confidence. Contributing: {contributing}."
    )
    if aggregation["opposing"]:
        detail += f" Opposing: {', '.join(aggregation['opposing'])}."
    return _node(
        "aggregation", f"Aggregation\n{aggregation['direction']}", "aggregation",
        status, detail,
        direction=aggregation["direction"], confidence=aggregation["confidence"],
    )


def _risk_node(risk: dict[str, Any] | None, failed: set[str]) -> dict[str, Any]:
    if "risk" in failed:
        return _node("risk", "Risk", "risk", "ERROR",
                     "The risk node failed, which is treated as a refusal.")
    if risk is None:
        return _node("risk", "Risk", "risk", "PENDING", "Risk has not evaluated a signal.")
    if not risk["approved"]:
        reason = risk["block_reason"] or risk["verdict"]
        detail = f"Refused: {reason}."
        if risk["reasoning"]:
            detail += f" {risk['reasoning'][0]}"
        return _node("risk", "Risk\nBLOCKED", "risk", "BLOCKED", detail,
                     block_reason=reason, verdict=risk["verdict"])
    quantity = risk["quantity"]
    amount = risk["risk_amount"]
    detail = "Approved"
    if quantity is not None:
        detail += f" for {quantity:.6g} units"
    if amount is not None:
        detail += f" risking ${amount:,.2f}"
    return _node("risk", "Risk\nAPPROVED", "risk", "ACTIVE", detail + ".",
                 quantity=quantity, risk_amount=amount)


def _execution_node(
    order: Any,
    risk: dict[str, Any] | None,
    trading_enabled: bool | None,
    failed: set[str],
) -> dict[str, Any]:
    if "order" in failed or "execution" in failed:
        return _node("execution", "Execution", "execution", "ERROR",
                     "Order construction failed for this bar.")
    # A refusal outranks the presence of an order. The graph only routes to order
    # construction after an approval, so the two together mean something is
    # inconsistent - and in that case the safe reading is the refusal.
    if risk is not None and not risk["approved"]:
        detail = "No order was created because risk refused the signal."
        if order is not None:
            detail = (
                "Risk refused the signal, yet an order is present in the record. "
                "The refusal is what is reported; treat the order as suspect."
            )
        return _node("execution", "Execution", "execution", "BLOCKED", detail)
    if order is not None:
        return _node(
            "execution", "Execution\nPAPER ORDER", "execution", "ACTIVE",
            "A paper order was queued for the next bar's open. No broker is "
            "contacted and no real money is at risk.",
        )
    if trading_enabled is False:
        return _node(
            "execution", "Execution", "execution", "BLOCKED",
            "trading_enabled is false: the kill switch is on, so no order is "
            "created even when a signal and risk would allow one.",
        )
    if risk is None:
        return _node("execution", "Execution", "execution", "PENDING",
                     "Nothing has reached execution yet.")
    return _node("execution", "Execution", "execution", "WAIT",
                 "No actionable signal, so there was nothing to execute.")


def _position_node(open_positions: int | None, order: Any) -> dict[str, Any]:
    if open_positions is None:
        return _node("paper_position", "Paper Position", "paper_position", "PENDING",
                     "Position state unknown.")
    if open_positions > 0:
        return _node(
            "paper_position", f"Paper Position\n{open_positions} open", "paper_position",
            "ACTIVE", f"{open_positions} simulated position(s) open.",
            open_positions=open_positions,
        )
    detail = "No simulated position is open."
    if order is not None:
        detail += " The queued order fills at the next bar's open."
    return _node("paper_position", "Paper Position\nflat", "paper_position", "WAIT",
                 detail, open_positions=0)


# ----------------------------------------------------------------------- edges


def _edges(nodes: list[dict[str, Any]]) -> list[list[str]]:
    """Pipeline edges, fanning out to and back in from the strategy nodes."""
    strategy_ids = [n["id"] for n in nodes if n["stage"] == "strategies"]
    edges: list[list[str]] = [
        ["market_data", "features"],
        ["features", "regime"],
    ]
    for sid in strategy_ids:
        edges.append(["regime", sid])
        edges.append([sid, "selection"])
    edges.extend(
        [
            ["selection", "aggregation"],
            ["aggregation", "risk"],
            ["risk", "execution"],
            ["execution", "paper_position"],
        ]
    )
    return edges


def _assign_layout(nodes: list[dict[str, Any]]) -> None:
    """Give each node a column (stage order) and a row (fan-out position).

    Computed here so the arrangement is assertable without a rendering runtime.
    """
    columns = {stage: index for index, (stage, _) in enumerate(STAGES)}
    per_stage: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        per_stage.setdefault(node["stage"], []).append(node)

    for stage, stage_nodes in per_stage.items():
        count = len(stage_nodes)
        for offset, node in enumerate(stage_nodes):
            node["column"] = columns.get(stage, 0)
            # Centre each stage's nodes on row 0 so the trunk stays level and
            # only the strategy fan-out spreads vertically.
            node["row"] = float(offset) - (count - 1) / 2.0


def _path(by_id: dict[str, dict[str, Any]]) -> list[str]:
    """The trunk stages that are on the live path, in pipeline order.

    Stops at the first stage that is not ACTIVE or OK, which is what makes the
    view answer "how far did this bar get" at a glance.
    """
    trunk = [
        "market_data", "features", "regime", "selection",
        "aggregation", "risk", "execution", "paper_position",
    ]
    path = []
    for node_id in trunk:
        node = by_id.get(node_id)
        if node is None or node["status"] not in {"ACTIVE", "OK"}:
            break
        path.append(node_id)
    return path
