"""JSON-safe records for the paper-trading loop.

Everything the paper engine persists or hands to the dashboard passes through
this module first. That boundary exists because of a concrete failure: the
engine used to drop live ``Order`` dataclasses straight into its event log,
which meant the first fill of any paper session raised
``TypeError: Object of type Order is not JSON serializable`` inside
``json.dumps`` and took the whole loop down. Serialising at one recorded
boundary - rather than hoping every call site remembers - is what stops that
class of bug returning.

The second reason this module exists is that the dashboard needs a *richer*
record than logging does. :func:`app.graph.state.state_summary` flattens each
strategy signal to its direction string, which is enough for an experiment log
and nowhere near enough to explain a decision: it loses confidence, reasoning,
weights, suppression, and the risk engine's block reason. The paper loop
therefore builds :func:`build_decision_view`, a complete but JSON-safe account
of one bar, and every consumer - persistence, dashboard, graph view - reads
that single shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from app.utils.timeutils import normalize_timeframe, utcnow

#: Bumped when the persisted state layout changes incompatibly. A state file
#: from an older schema is reloaded on a best-effort basis and the mismatch is
#: recorded, rather than being assumed compatible.
STATE_SCHEMA_VERSION = 2

__all__ = [
    "STATE_SCHEMA_VERSION",
    "DataFreshness",
    "assess_freshness",
    "build_decision_view",
    "jsonable",
    "serialise_fill",
    "serialise_order",
    "serialise_position",
    "serialise_snapshot",
]


# --------------------------------------------------------------------- jsonable


def jsonable(value: Any, *, _depth: int = 0) -> Any:
    """Coerce ``value`` into something :func:`json.dumps` accepts.

    Containers are walked recursively. Objects exposing ``to_dict`` use it, so
    the platform's own dataclasses serialise the way their authors intended.
    Anything else degrades to ``str`` rather than raising: a log entry that
    reads ``"<Foo object at 0x...>"`` is a cosmetic problem, while an exception
    here would abort a trading loop over a diagnostic detail.

    NaN and infinity become ``None``. They are legal Python floats but not legal
    JSON, and ``json.dumps`` would otherwise emit bare ``NaN`` tokens that a
    strict reader rejects.
    """
    if _depth > 6:
        return str(value)

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Enum):
        return str(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.Index)):
        return [jsonable(v, _depth=_depth + 1) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        return [jsonable(v, _depth=_depth + 1) for v in value.tolist()]

    if isinstance(value, dict):
        return {str(k): jsonable(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v, _depth=_depth + 1) for v in value]

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return jsonable(to_dict(), _depth=_depth + 1)
        except Exception:  # noqa: BLE001 - a failed render must not stop the loop
            return str(value)

    return str(value)


def iso(value: pd.Timestamp | None) -> str | None:
    """ISO-8601 rendering of a timestamp, or None."""
    return None if value is None else pd.Timestamp(value).isoformat()


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    """Read a timestamp back from persisted state, always UTC-aware.

    A naive timestamp is interpreted as UTC rather than as local time: the
    platform's canonical index is UTC everywhere, and guessing a local zone on
    reload would shift every restored bar by the operator's offset.
    """
    if value is None or value == "":
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# -------------------------------------------------------------- execution rows


def serialise_order(order: Any) -> dict[str, Any]:
    """Persisted form of an :class:`~app.execution.models.Order`."""
    return {
        "symbol": order.symbol,
        "side": str(order.side),
        "quantity": float(order.quantity),
        "order_type": str(order.order_type),
        "created_at": iso(order.created_at),
        "strategy": order.strategy,
        "reference_price": order.reference_price,
        "metadata": jsonable(order.metadata),
    }


def serialise_fill(fill: Any, timestamp: pd.Timestamp) -> dict[str, Any]:
    """Persisted form of a :class:`~app.execution.models.Fill`.

    Keeps spread, slippage and commission apart. Collapsing them into one
    "cost" number is what makes it impossible to answer, later, whether a
    result was eaten by the spread or by the slippage model.
    """
    return {
        "symbol": fill.order.symbol,
        "side": str(fill.order.side),
        "status": str(fill.status),
        "quantity": float(fill.filled_quantity),
        "price": float(fill.price),
        "reference_price": float(fill.reference_price),
        "spread_cost": float(fill.spread_cost),
        "slippage_cost": float(fill.slippage_cost),
        "commission": float(fill.commission),
        "total_cost": float(fill.total_cost),
        "timestamp": iso(timestamp),
        "strategy": fill.order.strategy,
        "rejection_reason": fill.rejection_reason,
        "metadata": jsonable(fill.metadata),
    }


def serialise_position(position: Any, price: float | None = None) -> dict[str, Any]:
    """Persisted form of an open position, marked at ``price`` when supplied."""
    row: dict[str, Any] = {
        "symbol": position.symbol,
        "direction": str(position.direction),
        "quantity": float(position.quantity),
        "entry_price": float(position.entry_price),
        "entry_time": iso(position.entry_time),
        "stop_loss": float(position.stop_loss),
        "take_profit": None if position.take_profit is None else float(position.take_profit),
        "strategy": position.strategy,
        "entry_regime": position.entry_regime,
        "entry_costs": float(position.entry_costs),
        "risk_amount": float(position.risk_amount),
        "metadata": jsonable(position.metadata),
    }
    if price is not None and math.isfinite(price) and price > 0:
        row["mark_price"] = float(price)
        row["unrealised_pnl"] = float(position.unrealised_pnl(price))
        row["r_multiple"] = position.r_multiple(price)
        row["notional"] = float(abs(position.quantity * price))
    return row


def serialise_snapshot(snapshot: Any) -> dict[str, Any]:
    """Persisted form of one equity-curve point."""
    return {
        "timestamp": iso(snapshot.timestamp),
        "cash": float(snapshot.cash),
        "equity": float(snapshot.equity),
        "unrealised_pnl": float(snapshot.unrealised_pnl),
        "realised_pnl": float(snapshot.realised_pnl),
        "open_positions": int(snapshot.open_positions),
        "exposure": float(snapshot.exposure),
        "drawdown": float(snapshot.drawdown),
    }


# ----------------------------------------------------------------- freshness


@dataclass(frozen=True)
class DataFreshness:
    """How old the newest bar is, relative to how often bars should arrive.

    ``status`` is one of ``FRESH``, ``DELAYED``, ``STALE`` or ``UNKNOWN``.
    ``UNKNOWN`` is deliberately not a synonym for fine: it means the age could
    not be established, which for a monitoring surface is a problem in itself.
    """

    status: str
    last_bar: str | None
    age_seconds: float | None
    intervals: float | None
    detail: str

    @property
    def is_usable(self) -> bool:
        """Whether a decision made on this data would be acting on current prices."""
        return self.status in {"FRESH", "DELAYED"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "last_bar": self.last_bar,
            "age_seconds": self.age_seconds,
            "intervals": self.intervals,
            "detail": self.detail,
        }


def assess_freshness(
    last_bar: pd.Timestamp | str | None,
    timeframe: str,
    *,
    now: pd.Timestamp | None = None,
    fresh_intervals: float = 1.5,
    delayed_intervals: float = 3.0,
    weekday_only: bool = True,
) -> DataFreshness:
    """Grade the age of ``last_bar`` for a ``timeframe`` feed.

    Age is measured in bar intervals rather than in absolute time, because
    "four hours old" means something entirely different on a 1M feed than on a
    daily one.

    For daily and coarser timeframes the elapsed span is counted in business
    days when ``weekday_only`` is set, so a Monday-morning check of a Friday
    close is not reported as three intervals stale. Public holidays are *not*
    modelled - an exchange holiday will show as one interval of extra delay,
    which is stated in ``detail`` rather than silently smoothed away.
    """
    timestamp = parse_timestamp(last_bar)
    if timestamp is None:
        return DataFreshness(
            status="UNKNOWN",
            last_bar=None,
            age_seconds=None,
            intervals=None,
            detail=(
                "No bar timestamp is available, so the feed's age is unknown. An "
                "unknown age is not treated as fresh."
            ),
        )

    reference = pd.Timestamp(now) if now is not None else utcnow()
    if reference.tzinfo is None:
        reference = reference.tz_localize("UTC")

    tf = normalize_timeframe(timeframe)
    interval_seconds = max(1, tf.minutes * 60)
    age_seconds = float((reference - timestamp).total_seconds())

    if age_seconds < 0:
        return DataFreshness(
            status="UNKNOWN",
            last_bar=iso(timestamp),
            age_seconds=age_seconds,
            intervals=None,
            detail=(
                f"The newest bar is stamped {iso(timestamp)}, which is in the future "
                "relative to now. The feed's clock or timezone handling is wrong; the age "
                "cannot be trusted."
            ),
        )

    note = ""
    if weekday_only and tf.minutes >= 1440:
        # Count only business days, so weekends do not masquerade as staleness.
        elapsed_days = float(
            np.busday_count(timestamp.date(), reference.date())
        )
        intervals = elapsed_days / max(1.0, tf.minutes / 1440.0)
        note = " Weekends are excluded; exchange holidays are not modelled."
    else:
        intervals = age_seconds / interval_seconds

    if intervals <= fresh_intervals:
        status = "FRESH"
    elif intervals <= delayed_intervals:
        status = "DELAYED"
    else:
        status = "STALE"

    detail = (
        f"Newest bar {iso(timestamp)} is {intervals:.1f} {tf.code} intervals old "
        f"({age_seconds / 3600:.1f}h). FRESH at or below {fresh_intervals:g}, "
        f"DELAYED to {delayed_intervals:g}, STALE beyond.{note}"
    )
    return DataFreshness(
        status=status,
        last_bar=iso(timestamp),
        age_seconds=age_seconds,
        intervals=float(intervals),
        detail=detail,
    )


# -------------------------------------------------------------- decision view


def build_decision_view(
    state: Any,
    *,
    trading_enabled: bool | None = None,
    data_quality: str | None = None,
    warm: bool | None = None,
    skipped_reason: str | None = None,
) -> dict[str, Any]:
    """A complete, JSON-safe account of one bar's decision.

    This is the canonical shape the dashboard and graph view read. It is built
    from recorded state only - every field traces to something a node actually
    produced - so the explanation cannot drift from what the graph did.

    Args:
        state: the :class:`~app.graph.state.TradingState` returned by a graph
            run, or a mapping shaped like one.
        trading_enabled: the kill-switch position at the time of the run,
            recorded so a dashboard can distinguish "no signal" from "signals
            were analysed but execution is switched off".
        data_quality: the grade the run was told to size against.
        warm: whether the bar was past the feature warm-up period.
        skipped_reason: why no decision was taken, when none was.
    """
    state = state or {}

    regime = state.get("regime")
    aggregated = state.get("aggregated")
    risk = state.get("risk_decision")
    order = state.get("order")
    signals = state.get("strategy_signals") or {}
    weights = state.get("strategy_weights") or {}

    strategies = []
    for name in sorted(signals):
        signal = signals[name]
        weight = weights.get(name)
        strategies.append(_strategy_row(name, signal, weight))

    # A strategy that was weighted but whose node never reported is worth
    # showing: silence there means the node failed, not that it said WAIT.
    for name in sorted(set(weights) - set(signals)):
        strategies.append(_strategy_row(name, None, weights[name]))

    view: dict[str, Any] = {
        "timestamp": iso(state.get("timestamp")),
        "symbol": state.get("symbol"),
        "timeframe": state.get("timeframe"),
        "equity_at_decision": jsonable(state.get("equity")),
        "data_quality": None if data_quality is None else str(data_quality),
        "trading_enabled": trading_enabled,
        "warm": warm,
        "skipped_reason": skipped_reason,
        "regime": _regime_row(regime),
        "strategies": strategies,
        "aggregation": _aggregation_row(aggregated),
        "risk": _risk_row(risk),
        "order": None if order is None else serialise_order(order),
        "decision": state.get("decision"),
        "errors": jsonable(state.get("errors") or []),
        "trace": jsonable(state.get("trace") or []),
    }
    view["suppressed"] = [s["strategy"] for s in strategies if s["suppressed"]]
    view["rejected"] = [s["strategy"] for s in strategies if not s["actionable"]]
    view["failed"] = sorted({str(e.get("node")) for e in view["errors"] if e.get("node")})
    return view


def _regime_row(regime: Any) -> dict[str, Any] | None:
    if regime is None:
        return None
    return {
        "regime": str(regime.regime),
        "confidence": float(regime.confidence),
        "volatility": str(regime.volatility),
        "trend_strength": float(regime.trend_strength),
        "is_uncertain": bool(getattr(regime, "is_uncertain", False)),
        "reasoning": list(regime.reasoning),
        "metrics": jsonable(dict(regime.metrics)),
    }


def _strategy_row(name: str, signal: Any, weight: Any) -> dict[str, Any]:
    """One row of the strategy table, merging the signal with its weight.

    A missing signal is reported as ``NO_SIGNAL`` rather than as ``WAIT``.
    Those are different events - the first is a node that failed, the second is
    a strategy that looked and declined - and a monitoring surface that renders
    them identically hides exactly the failure it exists to catch.
    """
    if signal is None:
        direction, confidence, actionable = "NO_SIGNAL", 0.0, False
        reasoning: list[str] = ["The strategy node produced no signal for this bar."]
        entry = stop = target = None
        metadata: dict[str, Any] = {}
    else:
        direction = str(signal.direction)
        confidence = float(signal.confidence)
        actionable = bool(signal.is_actionable)
        reasoning = list(signal.reasoning)
        entry = signal.entry_price
        stop = signal.stop_loss
        target = signal.take_profit
        metadata = jsonable(dict(signal.metadata))

    row: dict[str, Any] = {
        "strategy": name,
        "direction": direction,
        "confidence": confidence,
        "actionable": actionable,
        "reasoning": reasoning,
        "reason": reasoning[0] if reasoning else "",
        "entry_price": entry,
        "stop_loss": stop,
        "take_profit": target,
        "metadata": metadata,
        "weight": None,
        "base_weight": None,
        "regime_factor": None,
        "performance_factor": None,
        "weight_reasoning": [],
        "suppressed": False,
    }

    if weight is not None:
        row.update(
            {
                "weight": float(getattr(weight, "weight", 0.0)),
                "base_weight": _opt_float(getattr(weight, "base_weight", None)),
                "regime_factor": _opt_float(getattr(weight, "regime_factor", None)),
                "performance_factor": _opt_float(getattr(weight, "performance_factor", None)),
                "weight_reasoning": list(getattr(weight, "reasoning", ()) or ()),
                "suppressed": float(getattr(weight, "weight", 0.0)) <= 0.0,
            }
        )
    return row


def _aggregation_row(aggregated: Any) -> dict[str, Any] | None:
    if aggregated is None:
        return None
    return {
        "direction": str(aggregated.direction),
        "confidence": float(aggregated.confidence),
        "actionable": bool(aggregated.is_actionable),
        "method": str(aggregated.method),
        "contributing": list(aggregated.contributing),
        "opposing": list(aggregated.opposing),
        "scores": jsonable(dict(aggregated.scores)),
        "reasoning": list(aggregated.reasoning),
        "entry_price": aggregated.entry_price,
        "stop_loss": aggregated.stop_loss,
        "take_profit": aggregated.take_profit,
    }


def _risk_row(risk: Any) -> dict[str, Any] | None:
    if risk is None:
        return None
    return {
        "verdict": str(risk.verdict),
        "approved": bool(risk.approved),
        "block_reason": None if risk.block_reason is None else str(risk.block_reason),
        "quantity": float(risk.quantity),
        "risk_amount": float(risk.risk_amount),
        "risk_per_unit": float(risk.risk_per_unit),
        "notional": float(risk.notional),
        "reasoning": list(risk.reasoning),
        "metrics": jsonable(dict(risk.metrics)),
    }


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if math.isfinite(as_float) else None
