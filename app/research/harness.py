"""Running a configuration over one temporal segment.

The backtester from phase 3 runs a configuration over a series. Validation
needs something slightly different: run it over a *slice*, with a read-only
warm-up prefix in front, and score only the slice. Doing that by hand at every
call site is how a warm-up prefix quietly ends up inside the numbers, so it
happens here once.

Three properties this module is responsible for:

**Warm-up never scores.** The prefix is handed to the backtester as its warm-up
period, so no decision is taken inside it, and the metrics are recomputed from
the evaluation window alone rather than from the whole slice.

**Every segment is graded on its own data.** A quality report for a decade says
nothing about which year inside it was gappy. Each segment is graded
separately and its grade goes to the risk engine, so a degraded window is
refused rather than averaged into a clean-looking whole.

**Segments resolve their own trades.** Open positions are closed at the end of
each window, so a trade cannot leak its outcome across a boundary into a
segment that did not open it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.backtest.engine import Backtester
from app.backtest.metrics import PerformanceMetrics, compute_metrics
from app.backtest.results import BacktestResult
from app.config.loader import AppConfig, override_config
from app.config.models import AssetConfig
from app.data.schema import MarketData
from app.data.validators.quality import DataQualityEngine, QualityStatus
from app.research.splits import Segment
from app.utils.logging import get_logger

logger = get_logger(__name__)


class HarnessError(RuntimeError):
    """Raised when a segment cannot be run honestly."""


@dataclass(frozen=True)
class SegmentRun:
    """One configuration measured over one segment.

    ``metrics`` covers the evaluation window only. ``result`` is the raw
    backtest over window-plus-prefix and is kept so that provenance, decision
    logs and node errors remain inspectable.
    """

    segment: Segment
    variant: str
    metrics: PerformanceMetrics
    trades: pd.DataFrame
    equity_curve: pd.Series
    data_quality: str
    data_quality_source: str
    result: BacktestResult
    notes: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.segment.name

    @property
    def degraded_data(self) -> bool:
        """Whether this window ran on data the quality gate did not pass clean."""
        return self.data_quality != str(QualityStatus.PASS)

    def summary_row(self) -> dict[str, Any]:
        """One flat row, for the comparison tables in the report."""
        m = self.metrics
        return {
            "segment": self.segment.name,
            "role": str(self.segment.role),
            "variant": self.variant,
            "start": self.segment.start_time,
            "end": self.segment.end_time,
            "bars": self.segment.bars,
            "trades": m.total_trades,
            "total_return": m.total_return,
            "cagr": m.cagr,
            "max_drawdown": m.max_drawdown,
            "max_dd_bars": m.max_drawdown_duration_bars,
            "sharpe": m.sharpe_ratio,
            "sortino": m.sortino_ratio,
            "profit_factor": m.profit_factor,
            "expectancy_r": m.expectancy_r,
            "win_rate": m.win_rate,
            "data_quality": self.data_quality,
        }


class ResearchHarness:
    """Runs configurations over temporal segments of one series."""

    def __init__(
        self,
        config: AppConfig,
        data: MarketData,
        asset: AssetConfig | None = None,
        *,
        trading_enabled: bool = True,
        quality_status: str | None = None,
    ) -> None:
        """
        Args:
            config: the baseline configuration.
            data: the full series every segment is cut from.
            asset: asset configuration, for execution and sizing rules.
            trading_enabled: the kill switch. True for research runs, because a
                validation study of a system that opens no positions measures
                nothing.
            quality_status: grade for the whole series. Segments are still
                graded individually; this is only the fallback for a segment
                whose own grading fails.
        """
        self.config = config
        self.data = data
        self.asset = asset
        self.trading_enabled = trading_enabled
        self.series_quality = quality_status
        self._quality_engine = DataQualityEngine(config.data.quality)
        self._quality_cache: dict[tuple[int, int], str] = {}

    # ------------------------------------------------------------------ facts

    @property
    def feature_warmup_bars(self) -> int:
        """Bars before every feature is defined."""
        return self.config.features.warmup_bars()

    def required_warmup(self) -> int:
        """Warm-up a segment must carry for its first scored bar to be honest."""
        configured = self.config.backtest.warmup_bars
        return max(self.feature_warmup_bars, configured or 0)

    # -------------------------------------------------------------------- run

    def run(
        self,
        segment: Segment,
        config: AppConfig | None = None,
        variant: str = "baseline",
        experiment_id: str | None = None,
    ) -> SegmentRun:
        """Run ``config`` over ``segment`` and score the evaluation window.

        Args:
            segment: the window to score, carrying its own warm-up prefix.
            config: configuration to run. Defaults to the harness baseline.
            variant: label recorded with the result, so a comparison table can
                say which configuration produced which row.
            experiment_id: passed through to the backtest provenance record.

        Raises:
            HarnessError: when the segment's warm-up prefix is too short for the
                features to be defined at its first scored bar. Running anyway
                would score decisions taken on half-formed indicators.
        """
        run_config = config or self.config
        required = max(
            run_config.features.warmup_bars(), run_config.backtest.warmup_bars or 0
        )
        if segment.warmup_bars < required:
            raise HarnessError(
                f"Segment {segment.name!r} carries {segment.warmup_bars:,} warm-up bars but "
                f"the feature set needs {required:,}. Running it would score decisions taken "
                "on partially-formed indicators, which are different numbers rather than "
                "rougher versions of the real ones. Widen the warm-up or shorten the split."
            )

        df = segment.slice_frame(self.data.df)
        sliced = self.data.replace(df=df)
        quality = self._grade(segment, sliced)

        # The warm-up is pinned to exactly the prefix, so the first scored bar is
        # the first bar that can produce a decision. Positions are closed at the
        # window's end so the segment reports resolved outcomes only.
        run_config = override_config(
            run_config,
            backtest=run_config.backtest.model_copy(
                update={
                    "warmup_bars": segment.warmup_bars,
                    "close_positions_at_end": True,
                    "max_bars": None,
                }
            ),
        )

        result = Backtester(
            config=run_config, asset=self.asset, experiment_id=experiment_id or segment.name
        ).run(sliced, quality_status=quality, trading_enabled=self.trading_enabled)

        metrics, equity, trades, notes = self._score_window(segment, result, run_config)

        return SegmentRun(
            segment=segment,
            variant=variant,
            metrics=metrics,
            trades=trades,
            equity_curve=equity,
            data_quality=result.provenance.data_quality,
            data_quality_source=result.provenance.data_quality_source,
            result=result,
            notes=tuple(notes),
        )

    # -------------------------------------------------------------- internals

    def _grade(self, segment: Segment, sliced: MarketData) -> str:
        """Quality grade for this window's own data.

        Cached per (data_start, end) because a walk-forward study re-runs the
        same windows for every candidate configuration and grading a long
        window is not free.
        """
        key = (segment.data_start, segment.end)
        if key in self._quality_cache:
            return self._quality_cache[key]

        try:
            report = self._quality_engine.validate(
                sliced, self.config.assets.assets.get(sliced.symbol.upper())
            )
            grade = str(report.status)
        except Exception as exc:  # noqa: BLE001 - a failed grading is not a pass
            logger.warning(
                "Segment quality grading failed; falling back to the series grade",
                extra={"segment": segment.name, "error": str(exc)},
            )
            grade = self.series_quality or "UNKNOWN"

        self._quality_cache[key] = grade
        return grade

    def _score_window(
        self, segment: Segment, result: BacktestResult, config: AppConfig
    ) -> tuple[PerformanceMetrics, pd.Series, pd.DataFrame, list[str]]:
        """Recompute metrics from the evaluation window alone.

        The backtest's own metrics cover the warm-up prefix as well, where
        equity is flat by construction. Flat bars do not change the return but
        they do deflate volatility and inflate every ratio computed from it, so
        the window is scored on its own bars.
        """
        notes: list[str] = []
        equity = result.equity_curve
        if not equity.empty:
            equity = equity.loc[equity.index >= segment.start_time]

        trades = result.trades
        if not trades.empty and "entry_time" in trades.columns:
            entries = pd.to_datetime(trades["entry_time"], utc=True)
            trades = trades[entries >= segment.start_time].reset_index(drop=True)

        if equity.empty:
            notes.append("The evaluation window produced no equity points")
            return PerformanceMetrics(), equity, trades, notes

        initial = float(equity.iloc[0])
        costs = float(trades["costs"].sum()) if not trades.empty and "costs" in trades else 0.0

        metrics = compute_metrics(
            equity_curve=equity,
            trades=trades,
            initial_equity=initial,
            bars_per_year=config.backtest.bars_per_year,
            risk_free_rate=config.backtest.risk_free_rate,
            total_costs=costs,
        )

        blocked = result.blocked_counts or {}
        quality_blocks = sum(
            count for reason, count in blocked.items() if "DATA_QUALITY" in str(reason)
        )
        if quality_blocks:
            notes.append(
                f"{quality_blocks:,} decision(s) were refused by the risk engine because of "
                f"this window's data quality ({result.provenance.data_quality}). The window's "
                "results describe a system that was largely prevented from trading."
            )

        if metrics.total_trades == 0:
            notes.append(
                "No trades in this window. That is a result, not a gap - but it means "
                "every ratio below is undefined rather than zero."
            )

        return metrics, equity, trades, notes


def stitch_equity(runs: list[SegmentRun], initial_equity: float) -> pd.Series:
    """Join consecutive windows into one continuous equity curve.

    Each window restarts from the same balance, so the raw curves cannot simply
    be concatenated. Compounding their period returns instead produces the
    curve an account would have followed if it had been run window after
    window - which is exactly what a walk-forward result is claiming.
    """
    pieces: list[pd.Series] = []
    equity = float(initial_equity)

    for run in sorted(runs, key=lambda r: r.segment.start):
        curve = run.equity_curve
        if curve.empty:
            continue
        start = float(curve.iloc[0])
        if start <= 0:
            continue
        scaled = curve / start * equity
        # Drop the first point of each window; it duplicates the previous
        # window's closing equity and would show as a zero-return bar.
        pieces.append(scaled.iloc[1:] if pieces else scaled)
        equity = float(scaled.iloc[-1])

    if not pieces:
        return pd.Series(dtype="float64", name="equity")

    stitched = pd.concat(pieces)
    stitched.name = "equity"
    return stitched[~stitched.index.duplicated(keep="first")].sort_index()
