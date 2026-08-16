"""Performance metrics.

Section 19 of the brief lists what must be computed; section 3 explains why the
list is long. A strategy is not summarised by any single number, and win rate is
the most actively misleading of them: a 90% win rate with a 20:1 loss ratio
loses money, and a 30% win rate with 5:1 winners makes it.

The metrics that actually decide whether a strategy is worth anything are
**expectancy** and **risk-adjusted return**. Everything else is context.

Every figure here is descriptive of a *past* sample. None of it forecasts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Computed statistics for one backtest."""

    # --- returns ---
    initial_equity: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0

    # --- risk ---
    max_drawdown: float = 0.0
    max_drawdown_duration_bars: int = 0
    average_drawdown: float = 0.0
    volatility_annual: float = 0.0
    downside_deviation: float = 0.0

    # --- risk-adjusted ---
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # --- trade statistics ---
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    expectancy_r: float = 0.0
    average_r: float = 0.0
    payoff_ratio: float = 0.0

    # --- exposure and costs ---
    total_costs: float = 0.0
    cost_drag: float = 0.0
    average_bars_held: float = 0.0
    exposure_pct: float = 0.0

    # --- streaks ---
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # --- caveats surfaced with the numbers, never buried ---
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        """Human-readable report."""
        width = 60
        lines = [
            "=" * width,
            "PERFORMANCE METRICS",
            "=" * width,
            "",
            "RETURNS",
            f"  Initial equity      {self.initial_equity:>14,.2f}",
            f"  Final equity        {self.final_equity:>14,.2f}",
            f"  Total return        {self.total_return:>14.2%}",
            f"  CAGR                {self.cagr:>14.2%}",
            "",
            "RISK",
            f"  Max drawdown        {self.max_drawdown:>14.2%}",
            f"  Max DD duration     {self.max_drawdown_duration_bars:>14,} bars",
            f"  Average drawdown    {self.average_drawdown:>14.2%}",
            f"  Volatility (annual) {self.volatility_annual:>14.2%}",
            "",
            "RISK-ADJUSTED",
            f"  Sharpe              {self.sharpe_ratio:>14.2f}",
            f"  Sortino             {self.sortino_ratio:>14.2f}",
            f"  Calmar              {self.calmar_ratio:>14.2f}",
            "",
            "TRADES",
            f"  Total               {self.total_trades:>14,}",
            f"  Winners / losers    {self.winning_trades:>7,} / {self.losing_trades:<6,}",
            f"  Win rate            {self.win_rate:>14.2%}",
            f"  Average win         {self.average_win:>14,.2f}",
            f"  Average loss        {self.average_loss:>14,.2f}",
            f"  Largest win         {self.largest_win:>14,.2f}",
            f"  Largest loss        {self.largest_loss:>14,.2f}",
            f"  Payoff ratio        {self.payoff_ratio:>14.2f}",
            "",
            "EDGE",
            f"  Profit factor       {self.profit_factor:>14.2f}",
            f"  Expectancy          {self.expectancy:>14,.2f} per trade",
            f"  Expectancy (R)      {self.expectancy_r:>14.3f}R",
            f"  Average R           {self.average_r:>14.3f}R",
            "",
            "COSTS AND EXPOSURE",
            f"  Total costs         {self.total_costs:>14,.2f}",
            f"  Cost drag           {self.cost_drag:>14.2%} of starting equity",
            f"  Average hold        {self.average_bars_held:>14.1f} bars",
            f"  Time in market      {self.exposure_pct:>14.2%}",
            "",
            "STREAKS",
            f"  Max consecutive W   {self.max_consecutive_wins:>14,}",
            f"  Max consecutive L   {self.max_consecutive_losses:>14,}",
        ]

        if self.warnings:
            lines += ["", "-" * width, "WARNINGS"]
            lines += [f"  - {w}" for w in self.warnings]

        lines += [
            "",
            "-" * width,
            "These figures describe one historical sample and are not a forecast.",
            "A backtest is the weakest form of evidence that a strategy works;",
            "out-of-sample and walk-forward results (phase 4) carry more weight.",
            "=" * width,
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    initial_equity: float,
    bars_per_year: int = 252,
    risk_free_rate: float = 0.0,
    total_costs: float = 0.0,
) -> PerformanceMetrics:
    """Compute every metric from an equity curve and trade log.

    Args:
        equity_curve: equity marked to market at each bar.
        trades: completed trades, as produced by ``Portfolio.trades_frame()``.
        initial_equity: starting balance.
        bars_per_year: for annualisation. 252 for daily bars.
        risk_free_rate: annual, for Sharpe.
        total_costs: execution costs, for the cost-drag figure.
    """
    metrics = PerformanceMetrics(initial_equity=initial_equity)

    if equity_curve.empty:
        metrics.warnings.append("Equity curve is empty; no metrics could be computed")
        return metrics

    metrics.final_equity = float(equity_curve.iloc[-1])
    metrics.total_return = (
        (metrics.final_equity - initial_equity) / initial_equity if initial_equity else 0.0
    )

    _returns_metrics(metrics, equity_curve, initial_equity, bars_per_year, risk_free_rate)
    _drawdown_metrics(metrics, equity_curve)
    _trade_metrics(metrics, trades)

    metrics.total_costs = float(total_costs)
    # Expressed against starting equity rather than against gross profit.
    # Dividing by gross profit produces figures like "2871%" whenever profit is
    # near zero, which is arithmetically true and tells the reader nothing.
    metrics.cost_drag = float(total_costs / initial_equity) if initial_equity else 0.0

    gross_profit = metrics.final_equity - initial_equity + total_costs
    if total_costs > 0 and gross_profit <= total_costs:
        metrics.warnings.append(
            f"Execution costs ({total_costs:,.2f}) consumed most or all of the gross "
            f"trading profit ({gross_profit:,.2f}). Before concluding anything about the "
            "strategy, check whether the cost assumptions or the trade frequency are "
            "the dominant effect."
        )

    _add_warnings(metrics, equity_curve, trades)
    return metrics


# ------------------------------------------------------------------ internals


def _returns_metrics(
    metrics: PerformanceMetrics,
    equity: pd.Series,
    initial: float,
    bars_per_year: int,
    risk_free_rate: float,
) -> None:
    returns = equity.pct_change().dropna()
    if returns.empty:
        return

    bars = len(equity)
    years = bars / bars_per_year
    if years > 0 and initial > 0 and metrics.final_equity > 0:
        metrics.cagr = float((metrics.final_equity / initial) ** (1 / years) - 1)

    metrics.volatility_annual = float(returns.std(ddof=1) * math.sqrt(bars_per_year))

    # Sharpe on per-bar excess returns, annualised.
    per_bar_rf = risk_free_rate / bars_per_year
    excess = returns - per_bar_rf
    std = excess.std(ddof=1)
    if std > 0:
        metrics.sharpe_ratio = float(excess.mean() / std * math.sqrt(bars_per_year))

    # Sortino penalises only downside deviation, which is what an investor
    # actually minds; upside volatility is not a risk.
    downside = excess[excess < 0]
    if len(downside) > 1:
        downside_std = float(downside.std(ddof=1))
        metrics.downside_deviation = downside_std * math.sqrt(bars_per_year)
        if downside_std > 0:
            metrics.sortino_ratio = float(
                excess.mean() / downside_std * math.sqrt(bars_per_year)
            )

    if metrics.max_drawdown > 0:
        metrics.calmar_ratio = float(metrics.cagr / metrics.max_drawdown)


def _drawdown_metrics(metrics: PerformanceMetrics, equity: pd.Series) -> None:
    running_peak = equity.cummax()
    drawdown = (running_peak - equity) / running_peak.replace(0, np.nan)
    drawdown = drawdown.fillna(0.0)

    metrics.max_drawdown = float(drawdown.max())
    in_drawdown = drawdown > 1e-12
    metrics.average_drawdown = float(drawdown[in_drawdown].mean()) if in_drawdown.any() else 0.0

    # Longest unbroken stretch below a previous peak.
    longest = current = 0
    for flag in in_drawdown:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    metrics.max_drawdown_duration_bars = int(longest)

    if metrics.max_drawdown > 0:
        metrics.calmar_ratio = float(metrics.cagr / metrics.max_drawdown)


def _trade_metrics(metrics: PerformanceMetrics, trades: pd.DataFrame) -> None:
    if trades.empty:
        return

    pnl = trades["net_pnl"].astype(float)
    metrics.total_trades = int(len(trades))

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    metrics.winning_trades = int(len(wins))
    metrics.losing_trades = int(len(losses))
    metrics.win_rate = float(len(wins) / len(pnl)) if len(pnl) else 0.0

    metrics.average_win = float(wins.mean()) if len(wins) else 0.0
    metrics.average_loss = float(losses.mean()) if len(losses) else 0.0
    metrics.largest_win = float(wins.max()) if len(wins) else 0.0
    metrics.largest_loss = float(losses.min()) if len(losses) else 0.0

    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    if gross_loss > 0:
        metrics.profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        # No losing trades at all. Reported as infinity rather than a large
        # finite number, and flagged, because it almost always means a tiny
        # sample rather than a flawless strategy.
        metrics.profit_factor = float("inf")

    metrics.expectancy = float(pnl.mean())

    if len(wins) and len(losses):
        metrics.payoff_ratio = float(wins.mean() / abs(losses.mean()))

    if "r_multiple" in trades.columns:
        r_values = pd.to_numeric(trades["r_multiple"], errors="coerce").dropna()
        if len(r_values):
            metrics.average_r = float(r_values.mean())
            metrics.expectancy_r = float(r_values.mean())

    if "bars_held" in trades.columns:
        metrics.average_bars_held = float(
            pd.to_numeric(trades["bars_held"], errors="coerce").fillna(0).mean()
        )

    metrics.max_consecutive_wins, metrics.max_consecutive_losses = _streaks(pnl)


def _streaks(pnl: pd.Series) -> tuple[int, int]:
    max_wins = max_losses = current_wins = current_losses = 0
    for value in pnl:
        if value > 0:
            current_wins += 1
            current_losses = 0
        else:
            current_losses += 1
            current_wins = 0
        max_wins = max(max_wins, current_wins)
        max_losses = max(max_losses, current_losses)
    return max_wins, max_losses


def _add_warnings(
    metrics: PerformanceMetrics, equity: pd.Series, trades: pd.DataFrame
) -> None:
    """Surface the conditions that make a result untrustworthy.

    These sit alongside the numbers rather than in a footnote, because the
    numbers are what get copied into a decision.
    """
    if metrics.total_trades == 0:
        metrics.warnings.append(
            "No trades were taken. The strategy, its risk limits or the kill switch "
            "prevented every entry - check before concluding anything about the logic."
        )
        return

    if metrics.total_trades < 30:
        metrics.warnings.append(
            f"Only {metrics.total_trades} trades. Below roughly 30, none of these "
            "statistics is stable; a single trade moves them materially."
        )

    if math.isinf(metrics.profit_factor):
        metrics.warnings.append(
            "Profit factor is infinite because no trade lost money. On any real "
            "sample this indicates too few trades, not a flawless strategy."
        )

    if metrics.win_rate > 0.75 and metrics.total_trades < 100:
        metrics.warnings.append(
            f"Win rate of {metrics.win_rate:.0%} on {metrics.total_trades} trades is "
            "high enough to be suspicious. Check for look-ahead bias and unrealistic fills."
        )

    if metrics.max_drawdown < 0.02 and metrics.total_return > 0.5:
        metrics.warnings.append(
            "A large return with almost no drawdown is the signature of a bug, "
            "not of a strategy. Verify execution assumptions before believing it."
        )

    if metrics.sharpe_ratio > 3.0:
        metrics.warnings.append(
            f"Sharpe of {metrics.sharpe_ratio:.1f} is far above what systematic "
            "strategies sustain out of sample. Treat as a red flag until walk-forward "
            "testing agrees."
        )

    if len(equity) < 100:
        metrics.warnings.append(
            f"Equity curve covers only {len(equity)} bars, too short to be informative."
        )

    if not trades.empty and "exit_reason" in trades.columns:
        ambiguous = int((trades["exit_reason"] == "AMBIGUOUS_BAR_SKIPPED").sum())
        if ambiguous:
            metrics.warnings.append(
                f"{ambiguous} trade(s) exited on a bar containing both stop and target; "
                "OHLC data cannot resolve which came first."
            )
