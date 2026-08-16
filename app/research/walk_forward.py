"""Walk-forward analysis.

A single in-sample/out-of-sample split answers "did this survive one unseen
period?" once. One good answer to that question is obtainable by luck. Walk
forward asks it repeatedly - decide using only what had happened by date *t*,
trade the window that followed, roll forward, repeat - and a run of good
answers is much harder to get by luck than one.

The selection step is where walk-forward earns its name and where it is most
often quietly broken. When candidate configurations are supplied, the choice
for each fold is made **using that fold's training window only**. Choosing with
any knowledge of the test window - including choosing once, globally, by
looking at the whole series and then "walking forward" with the winner - turns
the entire analysis into an elaborate in-sample result. That failure mode is
invisible in the output, so it is prevented structurally here: the test segment
is not run until after the choice is fixed.

Two numbers summarise the result:

**Walk-forward efficiency** - test performance divided by training performance.
Below 1.0 always; the question is how far below. It measures how much of what
the training window promised the following period delivered.

**Fold consistency** - how many folds were positive. A total carried by one
window is a result about that window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.backtest.metrics import PerformanceMetrics, compute_metrics
from app.config.loader import AppConfig
from app.research.harness import ResearchHarness, SegmentRun, stitch_equity
from app.research.objectives import Objective, sortino
from app.research.splits import WalkForwardFold
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FoldResult:
    """One train/test pair, and what was chosen on the strength of the training half."""

    index: int
    train: SegmentRun
    test: SegmentRun
    selected_variant: str
    selection_score: float
    candidates_considered: int
    selection_note: str

    @property
    def efficiency(self) -> float | None:
        """Test objective as a fraction of the training objective."""
        train_value = self.train.metrics.sortino_ratio
        if abs(train_value) < 1e-9:
            return None
        return self.test.metrics.sortino_ratio / train_value

    def summary_row(self) -> dict[str, Any]:
        return {
            "fold": self.index,
            "variant": self.selected_variant,
            "train_start": self.train.segment.start_time,
            "train_end": self.train.segment.end_time,
            "test_start": self.test.segment.start_time,
            "test_end": self.test.segment.end_time,
            "train_trades": self.train.metrics.total_trades,
            "test_trades": self.test.metrics.total_trades,
            "train_sortino": self.train.metrics.sortino_ratio,
            "test_sortino": self.test.metrics.sortino_ratio,
            "test_return": self.test.metrics.total_return,
            "test_max_dd": self.test.metrics.max_drawdown,
            "test_expectancy_r": self.test.metrics.expectancy_r,
            "test_quality": self.test.data_quality,
        }


@dataclass
class WalkForwardReport:
    """Every fold, plus the stitched out-of-sample record they form."""

    folds: list[FoldResult] = field(default_factory=list)
    stitched_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    stitched_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    objective_name: str = "sortino"
    warnings: tuple[str, ...] = ()

    # ------------------------------------------------------------------ shape

    @property
    def fold_count(self) -> int:
        return len(self.folds)

    @property
    def profitable_folds(self) -> int:
        return sum(1 for f in self.folds if f.test.metrics.expectancy_r > 0)

    @property
    def traded_folds(self) -> int:
        return sum(1 for f in self.folds if f.test.metrics.total_trades > 0)

    @property
    def efficiency(self) -> float | None:
        """Mean test objective over mean training objective.

        Averaging the two sides separately rather than averaging per-fold ratios
        keeps one fold with a near-zero training figure from producing a ratio
        of several hundred and dominating the mean.
        """
        train = [f.train.metrics.sortino_ratio for f in self.folds]
        test = [f.test.metrics.sortino_ratio for f in self.folds]
        if not train or abs(np.mean(train)) < 1e-9:
            return None
        return float(np.mean(test) / np.mean(train))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.summary_row() for f in self.folds])

    # ----------------------------------------------------------------- render

    def render(self) -> str:
        width = 60
        lines = ["-" * width, "WALK-FORWARD", "-" * width]

        if not self.folds:
            lines.append("  No folds were run.")
            return "\n".join(lines)

        lines.append(f"  Folds                    {self.fold_count:>14,}")
        lines.append(f"  Folds that traded        {self.traded_folds:>14,}")
        lines.append(
            f"  Folds with positive R    {self.profitable_folds:>14,}"
            f"  ({self.profitable_folds / self.fold_count:.0%})"
        )
        efficiency = self.efficiency
        if efficiency is not None:
            lines.append(f"  Walk-forward efficiency  {efficiency:>14.2f}")

        lines += ["", "  Stitched out-of-sample record (folds joined end to end):"]
        m = self.stitched_metrics
        lines += [
            f"    Trades              {m.total_trades:>14,}",
            f"    Total return        {m.total_return:>14.2%}",
            f"    Max drawdown        {m.max_drawdown:>14.2%}",
            f"    Sharpe              {m.sharpe_ratio:>14.2f}",
            f"    Sortino             {m.sortino_ratio:>14.2f}",
            f"    Expectancy          {m.expectancy_r:>14.3f}R",
        ]

        lines += ["", "  Per fold:"]
        frame = self.to_frame()
        if not frame.empty:
            display = frame[
                ["fold", "variant", "test_start", "test_trades", "test_return",
                 "test_max_dd", "test_expectancy_r"]
            ].copy()
            display["test_start"] = pd.to_datetime(display["test_start"]).dt.strftime("%Y-%m-%d")
            lines.append(display.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

        if self.warnings:
            lines += ["", "  Caveats:"]
            lines += [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)


class WalkForwardAnalysis:
    """Runs a rolling train/test evaluation, optionally choosing per fold."""

    def __init__(
        self,
        harness: ResearchHarness,
        folds: list[WalkForwardFold],
        *,
        candidates: dict[str, AppConfig] | None = None,
        objective: Objective = sortino,
        objective_name: str = "sortino",
        on_trial: Any = None,
    ) -> None:
        """
        Args:
            harness: the segment runner.
            folds: train/test pairs from :func:`~app.research.splits.walk_forward_folds`.
            candidates: named configurations to choose between on each fold's
                training window. None means run the baseline configuration
                everywhere, which measures stability over time without adding
                any selection.
            objective: how training windows are scored when choosing.
            on_trial: callback ``(variant, config, segment, metrics)`` for every
                training evaluation, so the caller can count the search.
        """
        self.harness = harness
        self.folds = folds
        self.candidates = candidates or {}
        self.objective = objective
        self.objective_name = objective_name
        self.on_trial = on_trial

    def run(self) -> WalkForwardReport:
        """Execute every fold in chronological order."""
        results: list[FoldResult] = []
        warnings: list[str] = []

        for fold in self.folds:
            selected_name, selected_config, score, train_run, note = self._select(fold)

            # Only now, with the choice fixed by training data alone, is the test
            # window allowed to run.
            test_run = self.harness.run(
                fold.test, config=selected_config, variant=selected_name
            )

            results.append(
                FoldResult(
                    index=fold.index,
                    train=train_run,
                    test=test_run,
                    selected_variant=selected_name,
                    selection_score=score,
                    candidates_considered=max(1, len(self.candidates)),
                    selection_note=note,
                )
            )

            if test_run.degraded_data:
                warnings.append(
                    f"Fold {fold.index} tested on {test_run.data_quality}-grade data "
                    f"({fold.test.start_time:%Y-%m-%d} to {fold.test.end_time:%Y-%m-%d})."
                )

        stitched = stitch_equity(
            [r.test for r in results], self.harness.config.backtest.initial_balance
        )
        trades = [r.test.trades for r in results if not r.test.trades.empty]
        all_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()

        metrics = (
            compute_metrics(
                equity_curve=stitched,
                trades=all_trades,
                initial_equity=self.harness.config.backtest.initial_balance,
                bars_per_year=self.harness.config.backtest.bars_per_year,
                risk_free_rate=self.harness.config.backtest.risk_free_rate,
                total_costs=(
                    float(all_trades["costs"].sum())
                    if not all_trades.empty and "costs" in all_trades
                    else 0.0
                ),
            )
            if not stitched.empty
            else PerformanceMetrics()
        )

        warnings.extend(self._structural_warnings(results))

        return WalkForwardReport(
            folds=results,
            stitched_equity=stitched,
            stitched_metrics=metrics,
            objective_name=self.objective_name,
            warnings=tuple(warnings),
        )

    # -------------------------------------------------------------- selection

    def _select(
        self, fold: WalkForwardFold
    ) -> tuple[str, AppConfig | None, float, SegmentRun, str]:
        """Choose a configuration using this fold's training window only."""
        if not self.candidates:
            train_run = self.harness.run(fold.train, variant="baseline")
            return (
                "baseline",
                None,
                self.objective(train_run.metrics),
                train_run,
                "No candidates supplied; the baseline configuration was run unchanged.",
            )

        best_name: str | None = None
        best_score = -np.inf
        best_run: SegmentRun | None = None

        for name, candidate in self.candidates.items():
            run = self.harness.run(fold.train, config=candidate, variant=name)
            score = self.objective(run.metrics)
            if self.on_trial is not None:
                self.on_trial(name, candidate, fold.train, run.metrics)
            if score > best_score:
                best_name, best_score, best_run = name, score, run

        if best_run is None or not np.isfinite(best_score):
            # Every candidate scored -inf, which the objectives return when the
            # sample is too thin to rank. Falling back to the first candidate is
            # honest: nothing was learned, so nothing was chosen.
            name = next(iter(self.candidates))
            run = self.harness.run(fold.train, config=self.candidates[name], variant=name)
            return (
                name,
                self.candidates[name],
                float("-inf"),
                run,
                "No candidate produced enough trades on the training window to be ranked; "
                "the first was used and this fold's result carries no selection information.",
            )

        return (
            best_name,
            self.candidates[best_name],
            float(best_score),
            best_run,
            f"Chosen from {len(self.candidates)} candidates by {self.objective_name} on the "
            f"training window alone ({best_score:.3f}).",
        )

    # --------------------------------------------------------------- warnings

    def _structural_warnings(self, results: list[FoldResult]) -> list[str]:
        warnings: list[str] = []
        if not results:
            return ["No folds produced a result."]

        silent = [r.index for r in results if r.test.metrics.total_trades == 0]
        if len(silent) > len(results) / 2:
            warnings.append(
                f"{len(silent)} of {len(results)} test windows contained no trades. The "
                "walk-forward record mostly measures a system that was not trading."
            )

        thin = [r.index for r in results if 0 < r.test.metrics.total_trades < 10]
        if thin:
            warnings.append(
                f"Folds {thin} produced fewer than ten trades each; their individual "
                "statistics are noise."
            )

        if self.candidates:
            chosen = {r.selected_variant for r in results}
            if len(chosen) > 1:
                warnings.append(
                    f"Selection picked {len(chosen)} different configurations across folds "
                    f"({', '.join(sorted(chosen))}). An unstable choice means the training "
                    "windows disagree about what works, which is itself a finding."
                )
            else:
                warnings.append(
                    f"Selection picked {next(iter(chosen))} on every fold. The choice is "
                    "stable, so these results measure that configuration rather than the "
                    "selection procedure."
                )

        return warnings
