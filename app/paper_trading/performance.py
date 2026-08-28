"""Realised paper-trading performance, sliced by strategy and by regime.

Everything here is computed from **closed trades only**. That is the whole
discipline of the module: an open position has an opinion about its own P&L and
no result, and letting one into a performance table means the table improves
whenever a losing trade is left running. Unrealised P&L belongs in the
portfolio panel, where it is labelled as a mark rather than as an outcome.

Three things this module refuses to do.

It does not annualise, extrapolate, or project. A paper session that has closed
eleven trades knows eleven outcomes; multiplying them by a calendar produces a
number with the shape of a return and none of the content.

It does not hide a small sample. Every row carries its trade count, and rows
below :attr:`app.signals.selector.RegimePerformanceTracker.min_samples` are
flagged ``insufficient_evidence`` - the same threshold the selector itself uses
before it will let measured performance move a weight. A mean R computed from
three trades is noise, and a table that presents it beside one computed from
ninety invites exactly the wrong reading.

And it does not claim profitability. ``verdict`` is a description of a sample
("11 closed trades, mean -0.31R"), never an assessment of a strategy.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "build_performance",
    "contributors_of",
    "summarise_trades",
]


def _finite(value: Any) -> float | None:
    """``value`` as a float, or None when it is absent or not a real number."""
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if math.isfinite(as_float) else None


def summarise_trades(trades: list[dict[str, Any]], *, min_samples: int = 10) -> dict[str, Any]:
    """Aggregate statistics for one bucket of closed trades.

    ``expectancy_r`` is the mean R multiple, which is the only per-trade figure
    that compares across instruments and position sizes. It is ``None`` when no
    trade in the bucket had a defined R: a trade whose stop distance was zero
    has no R, and averaging over the ones that do while pretending the bucket is
    complete would overstate the sample.

    ``profit_factor`` is ``None`` rather than infinity when there are no losses.
    "Infinite profit factor" is an artefact of a short sample, and rendering it
    as a number invites it to be read as a result.
    """
    count = len(trades)
    nets = [_finite(t.get("net_pnl")) for t in trades]
    nets = [n for n in nets if n is not None]
    rs = [_finite(t.get("r_multiple")) for t in trades]
    rs = [r for r in rs if r is not None]

    wins = [n for n in nets if n > 0]
    losses = [n for n in nets if n < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)

    costs = [c for c in (_finite(t.get("costs")) for t in trades) if c is not None]
    held = [h for h in (_finite(t.get("bars_held")) for t in trades) if h is not None]

    return {
        "trades": count,
        "wins": len(wins),
        "losses": len(losses),
        "scratches": count - len(wins) - len(losses),
        "win_rate": (len(wins) / len(nets)) if nets else None,
        "net_pnl": sum(nets) if nets else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "expectancy_r": (sum(rs) / len(rs)) if rs else None,
        "total_r": sum(rs) if rs else 0.0,
        "best_r": max(rs) if rs else None,
        "worst_r": min(rs) if rs else None,
        "r_samples": len(rs),
        "avg_win": (gross_profit / len(wins)) if wins else None,
        "avg_loss": (gross_loss / len(losses)) if losses else None,
        "costs": sum(costs) if costs else 0.0,
        "avg_bars_held": (sum(held) / len(held)) if held else None,
        "ambiguous_exits": sum(1 for t in trades if t.get("ambiguous_bar")),
        # The selector will not let measured performance move a weight below
        # this many samples, so neither should a reader.
        "min_samples": int(min_samples),
        "insufficient_evidence": len(rs) < int(min_samples),
    }


def _verdict(stats: dict[str, Any]) -> str:
    """A sentence describing the sample. Never an assessment of the strategy."""
    count = stats["trades"]
    if count == 0:
        return "No closed trade yet, so there is nothing measured."

    expectancy = stats["expectancy_r"]
    expectancy_text = (
        "no defined R multiple in this sample"
        if expectancy is None
        else f"mean {expectancy:+.2f}R"
    )
    win_rate = stats["win_rate"]
    win_text = "" if win_rate is None else f", {win_rate:.0%} of them profitable"
    sentence = f"{count} closed trade(s){win_text}, {expectancy_text}."
    if stats["insufficient_evidence"]:
        sentence += (
            f" Below the {stats['min_samples']}-trade threshold the selector requires "
            "before measured performance moves a weight, so this is a sample, not "
            "evidence."
        )
    return sentence


def contributors_of(trade: dict[str, Any]) -> list[str]:
    """The strategies that voted for ``trade``, or its own strategy name.

    The aggregator hands the risk engine one signal named ``ensemble``, so the
    ``strategy`` field on a closed trade names the ensemble rather than any
    strategy that argued for it. The paper engine carries the contributing names
    through as ``meta_contributing``; this reads them back and falls back to the
    trade's own name when a record predates that (or when a single strategy
    traded on its own).
    """
    raw = trade.get("meta_contributing")
    if raw is None:
        raw = (trade.get("metadata") or {}).get("contributing") if isinstance(
            trade.get("metadata"), dict
        ) else None
    names = [str(n) for n in raw if str(n)] if isinstance(raw, (list, tuple)) else []
    return names or [str(trade.get("strategy") or "unknown")]


def _contributor_rows(
    trades: list[dict[str, Any]], *, min_samples: int
) -> list[dict[str, Any]]:
    """Per-contributing-strategy performance.

    One trade appears under every strategy that voted for it, so these rows do
    **not** sum to the portfolio's P&L - two strategies agreeing on one trade
    would double-count it. The intended reading is per-row: "of the trades this
    strategy argued for, this is how they turned out."
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        for name in contributors_of(trade):
            groups.setdefault(name, []).append(trade)

    rows = []
    for name in sorted(groups):
        stats = summarise_trades(groups[name], min_samples=min_samples)
        rows.append({"strategy": name, **stats, "verdict": _verdict(stats)})
    return sorted(rows, key=lambda r: (-r["trades"], r["strategy"]))


def _bucket(
    trades: list[dict[str, Any]],
    key: str,
    *,
    min_samples: int,
    label: str,
) -> list[dict[str, Any]]:
    """Group ``trades`` by ``trades[key]`` and summarise each group."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        name = str(trade.get(key) or "UNKNOWN")
        groups.setdefault(name, []).append(trade)

    rows = []
    for name in sorted(groups):
        stats = summarise_trades(groups[name], min_samples=min_samples)
        rows.append({label: name, **stats, "verdict": _verdict(stats)})
    # Most-traded first: the rows with the most evidence are the ones worth
    # reading, and sorting by P&L would put a one-trade fluke at the top.
    return sorted(rows, key=lambda r: (-r["trades"], str(r[label])))


def build_performance(
    trades: list[dict[str, Any]],
    *,
    selector_records: list[dict[str, Any]] | None = None,
    min_samples: int = 10,
    open_positions: int = 0,
) -> dict[str, Any]:
    """Strategy, regime and strategy x regime performance from closed trades.

    Args:
        trades: closed-trade records, as persisted by the paper engine.
        selector_records: the selector's ``(strategy, regime) -> [R]`` table, so
            the page can show what the *selector* is actually weighting on. It
            can legitimately differ from ``by_strategy_regime`` here: the
            persisted trade log is bounded, while the selector's table is not
            trimmed, so on a long session the selector may hold samples from
            trades that have aged out of the log. Showing both, and saying so,
            is better than showing one and implying it is the other.
        min_samples: the selector's evidence threshold, surfaced per row.
        open_positions: open positions, reported alongside so a reader can see
            how much of the session is still undecided.

    Returns:
        A JSON-safe payload. Every bucket is derived from closed trades; nothing
        is projected, annualised or extrapolated.
    """
    trades = [t for t in (trades or []) if isinstance(t, dict)]
    overall = summarise_trades(trades, min_samples=min_samples)

    by_exit: dict[str, int] = {}
    for trade in trades:
        reason = str(trade.get("exit_reason") or "UNKNOWN")
        by_exit[reason] = by_exit.get(reason, 0) + 1

    pairs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for trade in trades:
        key = (
            str(trade.get("strategy") or "unknown"),
            str(trade.get("entry_regime") or "UNKNOWN"),
        )
        pairs.setdefault(key, []).append(trade)

    by_strategy_regime = []
    for (strategy, regime) in sorted(pairs):
        stats = summarise_trades(pairs[(strategy, regime)], min_samples=min_samples)
        by_strategy_regime.append(
            {"strategy": strategy, "regime": regime, **stats, "verdict": _verdict(stats)}
        )
    by_strategy_regime.sort(key=lambda r: (-r["trades"], r["strategy"], r["regime"]))

    selector_rows = []
    for row in selector_records or []:
        samples = [s for s in (_finite(v) for v in (row.get("r_multiples") or [])) if s is not None]
        selector_rows.append(
            {
                "strategy": str(row.get("strategy") or "unknown"),
                "regime": str(row.get("regime") or "UNKNOWN"),
                "samples": len(samples),
                "mean_r": (sum(samples) / len(samples)) if samples else None,
                "total_r": sum(samples) if samples else 0.0,
                "win_rate": (
                    sum(1 for s in samples if s > 0) / len(samples) if samples else None
                ),
                "min_samples": int(min_samples),
                # The selector returns None below the threshold, so a row under
                # it is genuinely not influencing any weight yet.
                "influences_weight": len(samples) >= int(min_samples),
            }
        )
    selector_rows.sort(key=lambda r: (-r["samples"], r["strategy"], r["regime"]))

    return {
        "overall": {**overall, "verdict": _verdict(overall)},
        "by_strategy": _bucket(trades, "strategy", min_samples=min_samples, label="strategy"),
        "by_contributor": _contributor_rows(trades, min_samples=min_samples),
        "by_regime": _bucket(trades, "entry_regime", min_samples=min_samples, label="regime"),
        "by_strategy_regime": by_strategy_regime,
        "by_exit_reason": [
            {"exit_reason": reason, "trades": count}
            for reason, count in sorted(by_exit.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "selector_table": selector_rows,
        "open_positions": int(open_positions),
        "min_samples": int(min_samples),
        "basis": (
            "Closed trades only. Open positions are excluded because an open position "
            "has a mark, not an outcome. Nothing here is annualised, projected or "
            "presented as an expected return."
        ),
        "contributor_note": (
            "A trade appears under every strategy that voted for it, so contributor rows "
            "do not sum to the portfolio's P&L. Read each row on its own: of the trades "
            "this strategy argued for, this is how they turned out."
        ),
    }
