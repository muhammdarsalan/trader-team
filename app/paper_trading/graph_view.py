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

Third, **"how far did the bar get" is separate from "was the answer good".** A
node carries both a ``status`` - the amber/red reading a human needs - and a
``reached`` flag saying whether the stage actually ran. Collapsing the two used
to make a fully-completed decision on stale data report that "nothing
progressed past market data", which was simply false: every stage had run. The
path now follows ``reached``, and the caveat stays visible in the status.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "NODE_STATUS_COLOURS",
    "STAGES",
    "TRUNK",
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
        ``stopped_at``, ``stopped_reason``, ``final_decision`` and ``legend``.
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

    path, stopped_at, stopped_reason = _path(by_id)

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
        "path": path,
        "stopped_at": stopped_at,
        "stopped_reason": stopped_reason,
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

    # A raw TradingState has no "warm" field - that is added by
    # build_decision_view. Inferring it from downstream evidence is not a guess:
    # a regime or a strategy signal exists only if the feature stage ran, and
    # reporting features as PENDING while a strategy node has reported would
    # truncate the path at a stage that demonstrably completed.
    warm = source.get("warm")
    if warm is None and (strategies or source.get("regime") is not None):
        warm = True

    return {
        "strategies": strategies,
        "failed": failed,
        "risk": _coerce_risk(source.get("risk_decision")),
        "aggregation": _coerce_aggregation(source.get("aggregated")),
        "regime": _coerce_regime(source.get("regime"), source.get("regime_confidence")),
        "order": source.get("order"),
        "decision": source.get("decision"),
        "timestamp": _as_str(source.get("timestamp")),
        "warm": warm,
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
    *,
    reached: bool | None = None,
    terminal: bool = False,
    **metrics: Any,
) -> dict[str, Any]:
    """One graph node.

    Args:
        reached: whether this stage actually ran and recorded a result. Defaults
            to "any status other than PENDING or ERROR", which is right for
            every node that does not need to say otherwise. It is deliberately
            *not* derived from whether the answer was favourable: a stage that
            ran and declined, or ran on caveated data, was still reached.
        terminal: whether the flow stopped here. A warm-up skip, a risk refusal
            and a kill switch all reach their stage and end the bar there.
    """
    return {
        "id": node_id,
        "label": label,
        "stage": stage,
        "status": status,
        "detail": detail,
        "colour": NODE_STATUS_COLOURS.get(status, NODE_STATUS_COLOURS["PENDING"]),
        "reached": (status not in {"PENDING", "ERROR"}) if reached is None else bool(reached),
        "terminal": bool(terminal),
        "metrics": {k: v for k, v in metrics.items() if v is not None},
    }


def _market_node(market: dict[str, Any] | None, failed: set[str]) -> dict[str, Any]:
    if "market_data" in failed:
        return _node("market_data", "Market Data", "market_data", "ERROR",
                     "The market-data node failed for this bar.", terminal=True)
    if not market or market.get("status") == "NO_DATA":
        return _node(
            "market_data", "Market Data", "market_data", "PENDING",
            "No market data has been loaded, so nothing downstream has run.",
            terminal=True,
        )
    if market.get("status") in {"ERROR", "EMPTY"}:
        return _node(
            "market_data", "Market Data", "market_data", "ERROR",
            str(
                market.get("error")
                or (
                    "The provider returned no bars."
                    if market.get("status") == "EMPTY"
                    else "The market-data refresh failed."
                )
            ),
            terminal=True,
        )

    quality = str(market.get("quality_status", "UNKNOWN"))
    freshness = str((market.get("freshness") or {}).get("status", "UNKNOWN"))
    rows = market.get("rows") or 0

    # Bars were delivered, so the stage ran: `reached` stays True. The status
    # still carries the caveat, because a feed graded FAIL or gone stale must
    # not read the same as a clean current one - it is the reason risk will
    # refuse, and it belongs in amber on the picture.
    caveats: list[str] = []
    if quality == "FAIL":
        caveats.append("the quality gate graded this feed FAIL")
    elif quality == "UNKNOWN":
        caveats.append("no quality grade was recorded, which risk treats as a refusal")
    elif quality == "WARNING":
        caveats.append("the quality gate graded this feed WARNING")
    if freshness == "STALE":
        caveats.append("the newest bar is stale")
    elif freshness == "UNKNOWN":
        caveats.append("the feed's age could not be established")
    if market.get("is_proxy"):
        caveats.append("the series is a labelled proxy, not the instrument itself")

    status = "WAIT" if quality in {"FAIL", "UNKNOWN"} or freshness in {"STALE", "UNKNOWN"} else "OK"
    detail = (
        f"{rows:,} bars from {market.get('provider', 'unknown')}, "
        f"quality {quality}, freshness {freshness}."
    )
    if caveats:
        detail += " Caveats: " + "; ".join(caveats) + "."
    return _node(
        "market_data", "Market Data", "market_data", status, detail,
        reached=True,
        rows=rows, quality=quality, freshness=freshness,
        provider=market.get("provider"), last_bar=market.get("last_bar"),
        caveats=caveats or None,
    )


def _features_node(
    warm: bool | None, skipped_reason: str | None, failed: set[str]
) -> dict[str, Any]:
    if "features" in failed:
        return _node("features", "Features", "features", "ERROR",
                     "Feature computation failed for this bar.", terminal=True)
    if warm is None:
        return _node("features", "Features", "features", "PENDING",
                     "No bar has been analysed yet.", terminal=True)
    if not warm:
        return _node(
            "features", "Features", "features", "WARMUP",
            skipped_reason
            or "The bar is inside the feature warm-up period, so no decision was taken.",
            terminal=True,
        )
    return _node("features", "Features", "features", "OK",
                 "Causal features computed and past warm-up.")


def _regime_node(regime: dict[str, Any] | None, failed: set[str]) -> dict[str, Any]:
    if "regime" in failed:
        return _node("regime", "Regime", "regime", "ERROR",
                     "Regime detection failed for this bar.", terminal=True)
    if regime is None:
        return _node("regime", "Regime", "regime", "PENDING", "Regime not yet detected.",
                     terminal=True)
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
                  "No strategy has reported yet.", terminal=True)
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
                     "Strategy selection failed for this bar.", terminal=True)
    weighted = [s for s in strategies if s.get("weight") is not None]
    if not weighted:
        return _node("selection", "Selection", "selection", "PENDING",
                     "The selector has not weighted any strategy yet.", terminal=True)
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
                     "Signal aggregation failed for this bar.", terminal=True)
    if aggregation is None:
        return _node("aggregation", "Aggregation", "aggregation", "PENDING",
                     "No aggregated decision yet.", terminal=True)
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
        terminal=not aggregation["actionable"],
        direction=aggregation["direction"], confidence=aggregation["confidence"],
    )


def _risk_node(risk: dict[str, Any] | None, failed: set[str]) -> dict[str, Any]:
    if "risk" in failed:
        return _node("risk", "Risk", "risk", "ERROR",
                     "The risk node failed, which is treated as a refusal.", terminal=True)
    if risk is None:
        return _node("risk", "Risk", "risk", "PENDING", "Risk has not evaluated a signal.",
                     terminal=True)
    if not risk["approved"]:
        reason = risk["block_reason"] or risk["verdict"]
        detail = f"Refused: {reason}."
        if risk["reasoning"]:
            detail += f" {risk['reasoning'][0]}"
        return _node("risk", "Risk\nBLOCKED", "risk", "BLOCKED", detail,
                     terminal=True,
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
                     "Order construction failed for this bar.", terminal=True)
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
        return _node("execution", "Execution", "execution", "BLOCKED", detail, terminal=True)
    if order is not None:
        return _node(
            "execution", "Execution\nPAPER ORDER", "execution", "ACTIVE",
            "A paper order was queued for the next bar's open. No broker is "
            "contacted and no real money is at risk.",
        )
    # Nothing reached risk, so nothing reached execution either. The kill switch
    # is worth stating here because it will matter on a bar that does get this
    # far, but it did not stop *this* bar - aggregation did - and drawing the
    # stage as BLOCKED would claim an execution attempt that never happened.
    if risk is None:
        detail = "Nothing has reached execution yet."
        if trading_enabled is False:
            detail += (
                " Separately, trading_enabled is false, so no order would be "
                "created even on a bar that did reach this stage."
            )
        return _node("execution", "Execution", "execution", "PENDING", detail, terminal=True)
    if trading_enabled is False:
        return _node(
            "execution", "Execution", "execution", "BLOCKED",
            "trading_enabled is false: the kill switch is on, so no order is "
            "created even when a signal and risk would allow one.",
            terminal=True,
        )
    return _node("execution", "Execution", "execution", "WAIT",
                 "No actionable signal, so there was nothing to execute.", terminal=True)


def _position_node(open_positions: int | None, order: Any) -> dict[str, Any]:
    if open_positions is None:
        return _node("paper_position", "Paper Position", "paper_position", "PENDING",
                     "Position state unknown.", terminal=True)
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


#: Trunk stages, in pipeline order. The strategy fan-out sits between ``regime``
#: and ``selection`` and is not part of the trunk: individual strategies
#: declining is normal and does not stop the bar.
TRUNK: tuple[str, ...] = (
    "market_data", "features", "regime", "selection",
    "aggregation", "risk", "execution", "paper_position",
)


def _path(by_id: dict[str, dict[str, Any]]) -> tuple[list[str], str | None, str | None]:
    """How far the bar got, where it stopped, and why.

    A stage is on the path when it was *reached* - it ran and recorded a result.
    That is not the same as the result being favourable, and conflating the two
    is what used to make a completed decision over stale data report that
    nothing had progressed past market data.

    A terminal stage is included and then ends the walk: risk refusing a signal
    is a stage that ran, reached its conclusion, and stopped the bar there.
    """
    path: list[str] = []
    for node_id in TRUNK:
        node = by_id.get(node_id)
        if node is None or not node.get("reached"):
            return path, node_id if node is not None else None, (
                None if node is None else str(node.get("detail") or "")
            )
        path.append(node_id)
        if node.get("terminal"):
            return path, node_id, str(node.get("detail") or "")
    return path, None, None
