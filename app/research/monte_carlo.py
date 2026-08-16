"""Monte Carlo analysis of the trade sequence.

A backtest reports one path: the trades in the order history happened to
deliver them. That order is not a property of the strategy. Reshuffle the same
trades and the return is identical while the drawdown - the number that decides
whether an account survives - can double. The equity curve everybody looks at
is therefore one sample from a distribution, and the interesting question is
what the rest of that distribution looks like.

Two resampling methods, answering different questions:

**Permutation** keeps exactly the trades that happened and only reorders them.
It answers: *given this set of outcomes, how much of the drawdown was luck of
sequencing?* It cannot say anything about outcomes that did not occur.

**Bootstrap** samples with replacement, so some trades appear twice and others
not at all. It answers: *if these trades were draws from some stable process,
what else could that process have produced?* The premise is doing real work
there, and it is a premise, not a fact.

Both are run on a block basis by default, in blocks of one. Raising the block
size preserves whatever serial dependence exists between consecutive trades -
losing streaks cluster in trending systems, and a resampling that breaks the
clusters understates the drawdown it is supposed to be measuring.

What this cannot do: no resampling of past trades tells you whether the
strategy will keep producing trades like these. It bounds sequencing risk, not
model risk, and the two are routinely confused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Percentiles reported for every simulated distribution.
PERCENTILES: tuple[int, ...] = (5, 25, 50, 75, 95)


@dataclass(frozen=True)
class MonteCarloReport:
    """Distributions produced by resampling the trade sequence."""

    method: str
    simulations: int
    trades_per_path: int
    block_size: int
    initial_equity: float
    seed: int

    #: Observed values from the actual sequence, for comparison.
    observed_return: float = 0.0
    observed_max_drawdown: float = 0.0

    final_return: dict[str, float] = field(default_factory=dict)
    max_drawdown: dict[str, float] = field(default_factory=dict)
    longest_losing_streak: dict[str, float] = field(default_factory=dict)

    probability_of_loss: float = 0.0
    probability_drawdown_exceeds_limit: float = 0.0
    drawdown_limit: float | None = None
    probability_of_ruin: float = 0.0

    warnings: tuple[str, ...] = ()

    def render(self) -> str:
        width = 60
        lines = [
            "-" * width,
            f"MONTE CARLO ({self.method}, {self.simulations:,} paths, "
            f"block size {self.block_size})",
            "-" * width,
            f"  Trades per path     {self.trades_per_path:>14,}",
            f"  Observed return     {self.observed_return:>14.2%}",
            f"  Observed max DD     {self.observed_max_drawdown:>14.2%}",
            "",
            "  Return distribution",
        ]
        lines += [f"    p{p:<3}{self.final_return.get(f'p{p}', 0.0):>22.2%}" for p in PERCENTILES]
        lines += ["", "  Max drawdown distribution"]
        lines += [f"    p{p:<3}{self.max_drawdown.get(f'p{p}', 0.0):>22.2%}" for p in PERCENTILES]
        lines += [
            "",
            f"  P(final equity below start) {self.probability_of_loss:>14.1%}",
        ]
        if self.drawdown_limit is not None:
            lines.append(
                f"  P(drawdown > {self.drawdown_limit:.0%} limit) "
                f"{self.probability_drawdown_exceeds_limit:>15.1%}"
            )
        lines.append(f"  P(ruin, equity <= 0)        {self.probability_of_ruin:>14.1%}")

        if self.warnings:
            lines += ["", "  Caveats:"]
            lines += [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "simulations": self.simulations,
            "trades_per_path": self.trades_per_path,
            "block_size": self.block_size,
            "initial_equity": self.initial_equity,
            "seed": self.seed,
            "observed_return": self.observed_return,
            "observed_max_drawdown": self.observed_max_drawdown,
            "final_return": dict(self.final_return),
            "max_drawdown": dict(self.max_drawdown),
            "longest_losing_streak": dict(self.longest_losing_streak),
            "probability_of_loss": self.probability_of_loss,
            "probability_drawdown_exceeds_limit": self.probability_drawdown_exceeds_limit,
            "drawdown_limit": self.drawdown_limit,
            "probability_of_ruin": self.probability_of_ruin,
            "warnings": list(self.warnings),
        }


def monte_carlo_trade_sequence(
    trades: pd.DataFrame,
    initial_equity: float,
    *,
    method: str = "permutation",
    simulations: int = 2_000,
    block_size: int = 1,
    seed: int = 42,
    drawdown_limit: float | None = None,
    pnl_column: str = "net_pnl",
) -> MonteCarloReport:
    """Resample the trade sequence and report the resulting distributions.

    Args:
        trades: closed trades, as produced by the backtester.
        initial_equity: starting balance the paths are built from.
        method: ``"permutation"`` (reorder the same trades) or ``"bootstrap"``
            (sample with replacement).
        simulations: number of paths.
        block_size: resample in contiguous blocks of this many trades, to keep
            streaks intact. 1 assumes trades are independent, which they are
            not; the report says so.
        seed: fixed so the whole analysis reproduces from the experiment record.
        drawdown_limit: the configured maximum-drawdown limit, so the report can
            state how often a resampled path would have breached it.
        pnl_column: which column holds each trade's profit and loss.

    Notes:
        P&L is applied additively, in the currency it was earned in. Real
        position sizing is a fraction of equity, so a losing streak early would
        shrink later trades; treating the amounts as fixed slightly overstates
        the damage a deep drawdown does and understates a good run. The
        alternative - recompounding from R multiples - needs every trade to have
        one, which trades closed for reasons other than a stop do not always
        have. The simpler and more pessimistic assumption is the one used.
    """
    warnings: list[str] = []

    if trades is None or trades.empty or pnl_column not in trades.columns:
        return MonteCarloReport(
            method=method, simulations=0, trades_per_path=0, block_size=block_size,
            initial_equity=initial_equity, seed=seed,
            warnings=("No trades to resample; the distribution is undefined.",),
        )

    pnl = pd.to_numeric(trades[pnl_column], errors="coerce").dropna().to_numpy(dtype=float)
    n = len(pnl)

    if n < 2:
        return MonteCarloReport(
            method=method, simulations=0, trades_per_path=n, block_size=block_size,
            initial_equity=initial_equity, seed=seed,
            warnings=(f"Only {n} trade(s); resampling would report its own input.",),
        )

    if n < 30:
        warnings.append(
            f"Only {n} trades. Every percentile below is an estimate from a sample too "
            "small to have a stable shape; treat the spread as indicative, not measured."
        )
    if method == "permutation":
        warnings.append(
            "Permutation reorders exactly the trades that occurred. It bounds sequencing "
            "risk and says nothing about outcomes the sample happened not to contain."
        )
    else:
        warnings.append(
            "Bootstrap assumes these trades are independent draws from a stable process. "
            "If the market regime that produced them does not persist, the distribution "
            "describes a world that has stopped existing."
        )
    if block_size <= 1:
        warnings.append(
            "Block size 1 assumes consecutive trades are independent. Where losses cluster, "
            "this understates drawdown; re-run with a larger block size to check."
        )

    rng = np.random.default_rng(seed)
    simulations = max(1, int(simulations))
    block = max(1, int(block_size))

    finals = np.empty(simulations)
    drawdowns = np.empty(simulations)
    streaks = np.empty(simulations)
    ruined = 0

    for i in range(simulations):
        sequence = _resample(pnl, rng, method=method, block_size=block)
        equity = initial_equity + np.cumsum(sequence)

        finals[i] = equity[-1]
        drawdowns[i] = _max_drawdown(np.concatenate([[initial_equity], equity]))
        streaks[i] = _longest_losing_streak(sequence)
        if equity.min() <= 0:
            ruined += 1

    observed_equity = initial_equity + np.cumsum(pnl)
    observed_return = (observed_equity[-1] - initial_equity) / initial_equity
    observed_dd = _max_drawdown(np.concatenate([[initial_equity], observed_equity]))

    returns = (finals - initial_equity) / initial_equity

    breach = 0.0
    if drawdown_limit is not None:
        breach = float((drawdowns > drawdown_limit).mean())
        if breach > 0.10:
            warnings.append(
                f"{breach:.0%} of resampled paths breach the {drawdown_limit:.0%} "
                "maximum-drawdown limit. The observed path staying inside it was partly "
                "the order the trades arrived in."
            )

    return MonteCarloReport(
        method=method,
        simulations=simulations,
        trades_per_path=n,
        block_size=block,
        initial_equity=float(initial_equity),
        seed=seed,
        observed_return=float(observed_return),
        observed_max_drawdown=float(observed_dd),
        final_return=_percentiles(returns),
        max_drawdown=_percentiles(drawdowns),
        longest_losing_streak=_percentiles(streaks),
        probability_of_loss=float((finals < initial_equity).mean()),
        probability_drawdown_exceeds_limit=breach,
        drawdown_limit=drawdown_limit,
        probability_of_ruin=ruined / simulations,
        warnings=tuple(warnings),
    )


# ------------------------------------------------------------------ internals


def _resample(
    pnl: np.ndarray, rng: np.random.Generator, *, method: str, block_size: int
) -> np.ndarray:
    """One resampled trade sequence of the same length as the original."""
    n = len(pnl)

    if block_size <= 1:
        if method == "bootstrap":
            return pnl[rng.integers(0, n, size=n)]
        return rng.permutation(pnl)

    blocks_needed = int(np.ceil(n / block_size))
    if method == "bootstrap":
        starts = rng.integers(0, n, size=blocks_needed)
        # Wrap around the end so late trades are not systematically under-sampled.
        pieces = [pnl.take(range(s, s + block_size), mode="wrap") for s in starts]
        return np.concatenate(pieces)[:n]

    # Permutation in blocks: cut the sequence into blocks and reorder the blocks,
    # which keeps each streak intact while changing when it arrives.
    blocks = [pnl[i : i + block_size] for i in range(0, n, block_size)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])[:n]


def _max_drawdown(equity: np.ndarray) -> float:
    """Deepest peak-to-trough fall, as a positive fraction of the peak."""
    peaks = np.maximum.accumulate(equity)
    safe = np.where(peaks > 0, peaks, np.nan)
    drawdown = (peaks - equity) / safe
    worst = np.nanmax(drawdown) if len(drawdown) else 0.0
    return float(max(0.0, 0.0 if np.isnan(worst) else worst))


def _longest_losing_streak(pnl: np.ndarray) -> int:
    longest = current = 0
    for value in pnl:
        current = current + 1 if value <= 0 else 0
        longest = max(longest, current)
    return longest


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    out = {f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES}
    out["mean"] = float(np.mean(values))
    out["worst"] = float(np.min(values))
    out["best"] = float(np.max(values))
    return out
