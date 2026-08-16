"""Strategy correlation and redundancy.

Five strategies are not five independent opinions if three of them are reading
the same moving average. When they agree, the agreement carries no extra
information, and the aggregator's confidence in it is manufactured. When they
lose, they lose together, and the portfolio risk limits - which assume the
positions are separate bets - are sized against a diversification that does not
exist.

So redundancy is worth finding. What this module deliberately does **not** do
is act on it.

A high correlation between two strategies is not evidence that removing one
improves anything. The two may be redundant on this sample and diverge on the
next; the "redundant" one may be the one that survives; and dropping a strategy
because of a correlation measured on the same data used to justify keeping the
others is a decision made entirely in-sample. Deleting components on the
strength of an in-sample correlation matrix is a well-travelled route to a
system that is beautifully diversified against the past.

The output is therefore a set of *hypotheses*, each paired with a candidate
configuration that can be tested out of sample by the same machinery that tests
everything else. If down-weighting a strategy helps, the out-of-sample result
will say so. If it does not, nothing was thrown away.

Two kinds of correlation are measured, because they answer different questions:

**Return correlation** - do these strategies make and lose money at the same
times? This is the one that matters for portfolio risk.

**Signal agreement** - do they say the same thing at the same bars, including
the bars where neither trades? Two strategies can agree on direction almost
always and still have uncorrelated returns because they exit differently. The
first is a redundancy of *opinion*; the second is a redundancy of *outcome*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from app.config.loader import AppConfig, override_config
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RedundancyFinding:
    """Two strategies that behaved alike, and the experiment that would test it."""

    strategy_a: str
    strategy_b: str
    return_correlation: float | None
    signal_agreement: float | None
    overlapping_observations: int
    hypothesis: str

    def __str__(self) -> str:
        parts = [f"{self.strategy_a} / {self.strategy_b}:"]
        if self.return_correlation is not None:
            parts.append(f"return correlation {self.return_correlation:+.2f}")
        if self.signal_agreement is not None:
            parts.append(f"signal agreement {self.signal_agreement:.0%}")
        parts.append(f"over {self.overlapping_observations:,} observations")
        return " ".join(parts)


@dataclass
class CorrelationReport:
    """How alike the strategies were on one window."""

    segment: str
    return_correlations: pd.DataFrame = field(default_factory=pd.DataFrame)
    signal_agreement: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades_per_strategy: dict[str, int] = field(default_factory=dict)
    findings: tuple[RedundancyFinding, ...] = ()
    threshold: float = 0.7
    notes: tuple[str, ...] = ()

    @property
    def redundant_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple((f.strategy_a, f.strategy_b) for f in self.findings)

    def render(self) -> str:
        width = 60
        lines = ["-" * width, f"STRATEGY CORRELATION ({self.segment})", "-" * width]

        if self.trades_per_strategy:
            lines.append("  Trades per strategy:")
            for name, count in sorted(self.trades_per_strategy.items()):
                lines.append(f"    {name:<32}{count:>8,}")

        if not self.return_correlations.empty:
            lines += ["", "  Daily P&L correlation:"]
            lines.append(
                self.return_correlations.to_string(float_format=lambda v: f"{v:+.2f}")
            )

        if not self.signal_agreement.empty:
            lines += ["", "  Directional agreement (share of bars):"]
            lines.append(self.signal_agreement.to_string(float_format=lambda v: f"{v:.0%}"))

        lines += ["", f"  Redundancy findings (threshold {self.threshold:.2f}):"]
        if self.findings:
            for finding in self.findings:
                lines.append(f"    - {finding}")
            lines += [
                "",
                "  No strategy has been removed or re-weighted on the strength of these",
                "  numbers. A correlation measured on one window is a hypothesis about",
                "  redundancy, not a finding about it; the candidate configurations below",
                "  exist so it can be tested out of sample instead of assumed.",
            ]
        else:
            lines.append("    None above the threshold.")

        if self.notes:
            lines += ["", "  Notes:"]
            lines += [f"    - {note}" for note in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment": self.segment,
            "threshold": self.threshold,
            "trades_per_strategy": dict(self.trades_per_strategy),
            "return_correlations": (
                self.return_correlations.to_dict() if not self.return_correlations.empty else {}
            ),
            "signal_agreement": (
                self.signal_agreement.to_dict() if not self.signal_agreement.empty else {}
            ),
            "findings": [
                {
                    "strategy_a": f.strategy_a,
                    "strategy_b": f.strategy_b,
                    "return_correlation": f.return_correlation,
                    "signal_agreement": f.signal_agreement,
                    "observations": f.overlapping_observations,
                    "hypothesis": f.hypothesis,
                }
                for f in self.findings
            ],
            "notes": list(self.notes),
        }


# ------------------------------------------------------------------ measuring


def strategy_pnl_series(trades: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """Per-strategy profit and loss, resampled onto a common calendar.

    P&L is attributed to the *exit* bar, which is when it became real. Bars
    where a strategy had nothing running contribute zero rather than NaN: two
    strategies that trade in different months are genuinely uncorrelated, and
    dropping the quiet bars would hide exactly that.
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    required = {"strategy", "exit_time", "net_pnl"}
    if not required <= set(trades.columns):
        return pd.DataFrame()

    frame = trades[["strategy", "exit_time", "net_pnl"]].copy()
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    frame["net_pnl"] = pd.to_numeric(frame["net_pnl"], errors="coerce")

    pivot = (
        frame.pivot_table(
            index="exit_time", columns="strategy", values="net_pnl", aggfunc="sum"
        )
        .resample(freq)
        .sum()
    )
    return pivot.fillna(0.0)


def signal_agreement_matrix(decisions: pd.DataFrame) -> pd.DataFrame:
    """Share of bars on which each pair of strategies said the same thing.

    Uses the per-bar signal columns the backtester records, so it covers the
    bars where nothing traded too. Agreeing to wait is agreement - and if two
    strategies wait together nine bars in ten, they are one opinion with two
    names whatever their trades look like.
    """
    if decisions is None or decisions.empty:
        return pd.DataFrame()

    columns = [c for c in decisions.columns if c.startswith("signal_")]
    if len(columns) < 2:
        return pd.DataFrame()

    names = [c.removeprefix("signal_") for c in columns]
    values = decisions[columns].astype(str)

    matrix = pd.DataFrame(1.0, index=names, columns=names, dtype=float)
    for (i, left), (j, right) in combinations(list(enumerate(columns)), 2):
        agreement = float((values[left] == values[right]).mean())
        matrix.iloc[i, j] = agreement
        matrix.iloc[j, i] = agreement
    return matrix


def analyse_correlations(
    trades: pd.DataFrame,
    decisions: pd.DataFrame | None = None,
    *,
    segment: str = "in_sample",
    threshold: float = 0.7,
    min_observations: int = 30,
    isolated_equity: dict[str, pd.Series] | None = None,
    isolated_trades: dict[str, int] | None = None,
) -> CorrelationReport:
    """Measure how alike the strategies behaved, and flag the redundant pairs.

    Args:
        trades: the trade log for the window being examined.
        decisions: the per-bar decision log, for signal agreement.
        threshold: correlation above which a pair is flagged.
        min_observations: below this many overlapping observations, a pair is
            reported but not flagged - a correlation of 0.9 over eleven days is
            a coincidence with a decimal point.
        isolated_equity: equity curves from running each strategy on its own
            over the same window. The graph aggregates every strategy into one
            signal before a position is opened, so the trade log records the
            ensemble rather than which strategy earned what - the only way to
            get a per-strategy return series is to run them separately.
        isolated_trades: trade counts from those isolated runs.
    """
    notes: list[str] = []

    if (trades is None or trades.empty) and not isolated_equity:
        return CorrelationReport(
            segment=segment, threshold=threshold,
            notes=("No trades in this window; correlation is undefined.",),
        )

    counts: dict[str, int] = dict(isolated_trades or {})
    if not counts and trades is not None and not trades.empty and "strategy" in trades.columns:
        counts = trades["strategy"].value_counts().to_dict()

    correlations = pd.DataFrame()

    if isolated_equity:
        correlations = correlation_of_equity_curves(isolated_equity)
        if correlations.empty:
            notes.append(
                "Fewer than two strategies produced a varying equity curve when run on "
                "their own, so return correlation could not be computed."
            )
        else:
            notes.append(
                "Return correlation comes from running each strategy alone over this "
                "window. That is not quite the ensemble's behaviour - the aggregator and "
                "the risk engine both change which signals become positions - but it is "
                "the only way to see the strategies apart from each other."
            )
    else:
        pnl = strategy_pnl_series(trades)
        active = pnl.loc[:, pnl.std() > 0] if not pnl.empty else pnl
        if not active.empty and active.shape[1] >= 2:
            correlations = active.corr()
        else:
            notes.append(
                "Every trade in this window is labelled with the aggregated signal rather "
                "than an individual strategy, so per-strategy return correlation cannot be "
                "read from the trade log. Signal agreement below is measured directly."
            )

    agreement = signal_agreement_matrix(decisions) if decisions is not None else pd.DataFrame()

    findings: list[RedundancyFinding] = []
    observations = _observation_count(isolated_equity, decisions, trades)

    names = sorted(set(correlations.columns) | set(agreement.columns))
    for a, b in combinations(names, 2):
        corr = _lookup(correlations, a, b)
        agree = _lookup(agreement, a, b)

        redundant_returns = corr is not None and corr >= threshold
        redundant_signals = agree is not None and agree >= max(threshold, 0.9)
        if not (redundant_returns or redundant_signals):
            continue

        if observations < min_observations:
            notes.append(
                f"{a} and {b} look alike but only {observations} overlapping observations "
                "support it, which is too few to act on or to flag."
            )
            continue

        if redundant_returns and redundant_signals:
            hypothesis = (
                f"{a} and {b} both agree on direction and profit together. If they are one "
                "bet, halving the weight of one should leave out-of-sample risk-adjusted "
                "performance roughly unchanged while reducing correlated exposure."
            )
        elif redundant_returns:
            hypothesis = (
                f"{a} and {b} make and lose money at the same times despite differing on "
                "individual bars. The portfolio risk limits are treating them as separate "
                "bets; test whether down-weighting one reduces drawdown out of sample "
                "without costing expectancy."
            )
        else:
            hypothesis = (
                f"{a} and {b} say the same thing on most bars but their outcomes differ, "
                "so the difference is in exits rather than entries. Test whether the pair "
                "is adding anything beyond one of them alone."
            )

        findings.append(
            RedundancyFinding(
                strategy_a=a,
                strategy_b=b,
                return_correlation=corr,
                signal_agreement=agree,
                overlapping_observations=observations,
                hypothesis=hypothesis,
            )
        )

    if findings:
        notes.append(
            "These pairings were measured on one window. Redundancy on one sample is not "
            "redundancy in general, which is why nothing here changes the configuration."
        )

    return CorrelationReport(
        segment=segment,
        return_correlations=correlations,
        signal_agreement=agreement,
        trades_per_strategy={str(k): int(v) for k, v in counts.items()},
        findings=tuple(findings),
        threshold=threshold,
        notes=tuple(notes),
    )


def _observation_count(
    isolated_equity: dict[str, pd.Series] | None,
    decisions: pd.DataFrame | None,
    trades: pd.DataFrame | None,
) -> int:
    """How many aligned observations the correlation rests on.

    Whichever source produced the numbers is the one that decides whether there
    were enough of them - a 0.9 correlation over eleven observations is a
    coincidence with a decimal point regardless of which table it came from.
    """
    if isolated_equity:
        return max((len(series) for series in isolated_equity.values()), default=0)
    if decisions is not None and not decisions.empty:
        return len(decisions)
    return 0 if trades is None else len(trades)


def _lookup(matrix: pd.DataFrame, a: str, b: str) -> float | None:
    if matrix.empty or a not in matrix.index or b not in matrix.columns:
        return None
    value = matrix.loc[a, b]
    return None if pd.isna(value) else float(value)


# ----------------------------------------------------------------- candidates


def weight_variants(
    config: AppConfig,
    report: CorrelationReport,
    *,
    down_weight: float = 0.5,
) -> dict[str, AppConfig]:
    """Candidate configurations that test each redundancy hypothesis.

    For every flagged pair, one variant halves the standing weight of the
    strategy with fewer trades in the window, and one disables it outright. Both
    keep every strategy registered and every other setting identical, so the
    only thing the out-of-sample comparison varies is the hypothesis being
    tested.

    The strategy with fewer trades is the one tilted away from, because it has
    the thinner evidence base of the two - not because it is worse. Nothing here
    asserts that either variant is an improvement; that is what running them on
    unseen data is for.
    """
    variants: dict[str, AppConfig] = {"baseline": config}

    for finding in report.findings:
        a, b = finding.strategy_a, finding.strategy_b
        counts = report.trades_per_strategy
        target = a if counts.get(a, 0) <= counts.get(b, 0) else b
        if target not in config.strategies.strategies:
            continue

        variants[f"halve_{target}"] = _with_strategy_update(config, target, {"weight": down_weight})
        variants[f"drop_{target}"] = _with_strategy_update(config, target, {"enabled": False})

    return variants


def _with_strategy_update(
    config: AppConfig, strategy: str, update: dict[str, Any]
) -> AppConfig:
    """Copy of ``config`` with one strategy's block modified."""
    strategies = dict(config.strategies.strategies)
    strategies[strategy] = strategies[strategy].model_copy(update=update)
    return override_config(
        config, strategies=config.strategies.model_copy(update={"strategies": strategies})
    )


def summarise_variant_comparison(
    rows: list[dict[str, Any]], baseline_name: str = "baseline"
) -> str:
    """Plain-language comparison of weight variants on unseen data.

    Deliberately reports both the return-side and the risk-side change. A
    variant that improves expectancy while deepening drawdown has not been shown
    to be better; it has been shown to be different, and which of those two
    matters is not a question the arithmetic can settle.
    """
    if not rows:
        return "  No weight variants were compared."

    frame = pd.DataFrame(rows)
    baseline = frame[frame["variant"] == baseline_name]
    if baseline.empty:
        return "  No baseline row to compare the variants against."

    base = baseline.iloc[0]
    lines = [
        "  Variant comparison on unseen data "
        f"(baseline: {base.get('expectancy_r', 0):+.3f}R, "
        f"max drawdown {base.get('max_drawdown', 0):.1%}):"
    ]

    for _, row in frame[frame["variant"] != baseline_name].iterrows():
        d_expectancy = row.get("expectancy_r", 0) - base.get("expectancy_r", 0)
        d_drawdown = row.get("max_drawdown", 0) - base.get("max_drawdown", 0)
        better_return = d_expectancy > 0
        better_risk = d_drawdown < 0

        if better_return and better_risk:
            verdict = "better on both, on this one window"
        elif better_return or better_risk:
            verdict = "a trade-off, not an improvement"
        else:
            verdict = "worse on both"

        lines.append(
            f"    {row['variant']:<28} expectancy {d_expectancy:+.3f}R, "
            f"drawdown {d_drawdown:+.1%}  -> {verdict}"
        )

    lines += [
        "",
        "  One window cannot settle this. A variant that wins here has won once; the",
        "  configuration is left unchanged unless a walk-forward run agrees.",
    ]
    return "\n".join(lines)


def variant_row(variant: str, metrics: Any) -> dict[str, Any]:
    """One comparison row from a metrics object."""
    return {
        "variant": variant,
        "trades": metrics.total_trades,
        "expectancy_r": metrics.expectancy_r,
        "sortino": metrics.sortino_ratio,
        "max_drawdown": metrics.max_drawdown,
        "total_return": metrics.total_return,
    }


def correlation_of_equity_curves(curves: dict[str, pd.Series]) -> pd.DataFrame:
    """Correlation between whole equity curves, on returns rather than levels.

    Two rising equity curves correlate strongly whether or not their movements
    have anything to do with each other, which is why this differences them
    first. Same reasoning as the risk engine's price correlations.
    """
    usable = {
        name: curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        for name, curve in curves.items()
        if curve is not None and len(curve) > 2
    }
    usable = {name: series for name, series in usable.items() if series.std() > 0}
    if len(usable) < 2:
        return pd.DataFrame()
    return pd.DataFrame(usable).corr()
