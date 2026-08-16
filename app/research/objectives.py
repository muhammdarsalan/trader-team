"""Objectives used to compare configurations.

Anything that ranks configurations is a lever for overfitting, so the choice of
what to rank by matters more than it looks. Two rules are baked in here:

**Never rank by return.** Total return rewards whichever configuration happened
to size up before the biggest move in the sample. It has no risk in it at all,
and picking by it reliably selects the variant that took the most risk rather
than the one that earned the most per unit of it.

**Refuse to score a sample too small to score.** A configuration that took four
trades and won three has an excellent everything. Below a floor the objective
returns negative infinity, so a thin sample cannot win a comparison by being
lucky rather than by being good.

The objectives are risk-adjusted and, in the walk-forward selector, applied
only to training windows. None of them is the "correct" one; that is why the
name of the objective is recorded with every result.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from app.backtest.metrics import PerformanceMetrics

Objective = Callable[[PerformanceMetrics], float]

#: Below this many trades no objective will rank a configuration above another.
MIN_TRADES_FOR_RANKING = 20


def _guard(metrics: PerformanceMetrics, minimum: int = MIN_TRADES_FOR_RANKING) -> bool:
    return metrics.total_trades >= minimum


def sortino(metrics: PerformanceMetrics) -> float:
    """Return per unit of downside deviation. The default."""
    if not _guard(metrics):
        return -math.inf
    return metrics.sortino_ratio


def sharpe(metrics: PerformanceMetrics) -> float:
    if not _guard(metrics):
        return -math.inf
    return metrics.sharpe_ratio


def expectancy_r(metrics: PerformanceMetrics) -> float:
    """Mean R per trade: how much is earned per unit deliberately risked."""
    if not _guard(metrics):
        return -math.inf
    return metrics.expectancy_r


def calmar(metrics: PerformanceMetrics) -> float:
    """Return against the worst drawdown, for a survival-weighted view."""
    if not _guard(metrics):
        return -math.inf
    return metrics.calmar_ratio


def risk_adjusted_with_drawdown_penalty(metrics: PerformanceMetrics) -> float:
    """Sortino, penalised for deep drawdowns.

    Two configurations with the same Sortino are not equally attractive if one
    of them got there through a 40% drawdown. This says so explicitly instead
    of leaving it to whoever reads the table.
    """
    if not _guard(metrics):
        return -math.inf
    penalty = 1.0 + max(0.0, metrics.max_drawdown) * 2.0
    return metrics.sortino_ratio / penalty


OBJECTIVES: dict[str, Objective] = {
    "sortino": sortino,
    "sharpe": sharpe,
    "expectancy_r": expectancy_r,
    "calmar": calmar,
    "risk_adjusted": risk_adjusted_with_drawdown_penalty,
}


def get_objective(name: str) -> Objective:
    """Look up an objective by name, refusing return-maximising ones by design."""
    key = name.strip().lower()
    if key in {"return", "total_return", "cagr", "profit"}:
        raise ValueError(
            f"{name!r} is not available as a selection objective. Ranking configurations "
            "by return selects for risk taken rather than for edge; use one of "
            f"{sorted(OBJECTIVES)}."
        )
    if key not in OBJECTIVES:
        raise ValueError(f"Unknown objective {name!r}. Available: {sorted(OBJECTIVES)}")
    return OBJECTIVES[key]
