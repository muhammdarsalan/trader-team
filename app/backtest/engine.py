"""The event-driven backtester.

The ordering of operations within each bar is the entire correctness argument,
so it is stated explicitly:

    For bar i:
      1. Fill orders created at bar i-1, at bar i's OPEN.
      2. Check exits for every open position against bar i's HIGH and LOW.
         (A position entered at this bar's open can also exit on this bar - a
         gap through the stop is exactly how real trades die.)
      3. Mark to market at bar i's CLOSE and record the equity snapshot.
      4. Run the decision graph using data up to and including bar i.
         Any order it creates is queued for bar i+1.

Step 4 happening *after* step 3, and its order filling at step 1 of the *next*
bar, is what makes the backtest honest. A signal derived from bar i's close
cannot be filled at that close: the price had already traded by the time the
decision existed.

Features are computed once for the whole series and then truncated per bar.
That is safe only because every feature is causal, which the phase-2
``assert_causal`` suite proves independently. The truncation makes it
structural: a strategy cannot read a row that is not in the frame it is given.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.backtest.metrics import compute_metrics
from app.backtest.results import BacktestResult, RunProvenance
from app.config.loader import AppConfig
from app.config.models import AssetConfig
from app.data.cache import frame_checksum
from app.data.schema import MarketData
from app.data.validators.quality import DataQualityEngine
from app.execution.models import ExitReason, Order, Position
from app.execution.simulator import ExecutionSimulator
from app.features.engine import FeatureEngine
from app.graph.state import TradingState
from app.graph.workflow import TradingGraph
from app.portfolio.portfolio import Portfolio
from app.signals.models import SignalDirection
from app.signals.selector import RegimePerformanceTracker
from app.utils.logging import get_logger
from app.utils.paths import experiments_dir
from app.utils.timeutils import utcnow

logger = get_logger(__name__)


@dataclass
class PendingOrder:
    """An order awaiting the next bar's open."""

    order: Order
    stop_loss: float
    take_profit: float | None
    direction: SignalDirection
    risk_amount: float
    regime: str
    created_index: int


def next_experiment_id(prefix: str = "EXP") -> str:
    """Sequential experiment id, e.g. ``EXP-2026-003``."""
    year = utcnow().year
    base = experiments_dir()
    existing = 0
    if base.exists():
        existing = sum(
            1 for p in base.iterdir() if p.is_dir() and p.name.startswith(f"{prefix}-{year}-")
        )
    return f"{prefix}-{year}-{existing + 1:03d}"


class Backtester:
    """Runs the decision graph over history, bar by bar."""

    def __init__(
        self,
        config: AppConfig,
        asset: AssetConfig | None = None,
        experiment_id: str | None = None,
    ) -> None:
        self.config = config
        self.asset = asset
        self.experiment_id = experiment_id or next_experiment_id()

        # Fixed seed so every stochastic component is reproducible from the
        # experiment record alone.
        np.random.seed(config.platform.random_seed)

    # ------------------------------------------------------------------- run

    def run(
        self,
        data: MarketData,
        quality_status: str | None = None,
        trading_enabled: bool = True,
    ) -> BacktestResult:
        """Backtest ``data`` and return the full result.

        Args:
            data: validated market data from the phase-1 pipeline.
            quality_status: the data-quality grade, recorded in provenance so a
                result computed on WARNING-grade data says so, and passed to the
                risk engine so it can refuse to size against bad data. When
                omitted the backtester grades the series itself rather than
                proceeding on an unverified frame - a caller that reached the
                backtester without going through :class:`MarketDataService`
                must not thereby escape the quality gate.
            trading_enabled: the kill switch. Defaults to True here because a
                backtest that opens no positions is not a backtest; the live
                and paper paths take it from configuration instead.
        """
        df = data.df
        backtest_cfg = self.config.backtest
        quality_status, quality_source = self._resolve_quality(data, quality_status)

        portfolio = Portfolio(backtest_cfg.initial_balance, self.config.platform.base_currency)
        performance = RegimePerformanceTracker()

        graph = TradingGraph(
            config=self.config,
            portfolio=portfolio,
            asset=self.asset,
            performance=performance,
            trading_enabled=trading_enabled,
            data_quality=quality_status,
        )
        simulator = ExecutionSimulator(self.config.execution, self.asset)

        features = FeatureEngine(self.config.features).compute(data, self.asset)
        warmup = backtest_cfg.warmup_bars or features.warmup_bars

        if len(df) <= warmup:
            logger.warning(
                "Series is shorter than the warm-up period; no bars are tradable",
                extra={"bars": len(df), "warmup": warmup},
            )

        atr_series = features.df["atr"] if "atr" in features.df.columns else None
        regimes = features.df.index.to_series().map(lambda _: "UNKNOWN")

        pending: PendingOrder | None = None
        decisions: list[dict[str, Any]] = []
        node_errors: list[dict[str, Any]] = []
        blocked: Counter[str] = Counter()
        entry_index: dict[int, int] = {}

        limit = backtest_cfg.max_bars or len(df)
        symbol = data.symbol

        for i, timestamp in enumerate(df.index[:limit]):
            bar = df.iloc[i]
            atr = float(atr_series.iloc[i]) if atr_series is not None else None
            atr = atr if atr is not None and pd.notna(atr) else None

            # --- 1. fill the order queued at the previous bar -----------------
            if pending is not None:
                self._open_position(
                    pending, bar, timestamp, i, portfolio, simulator, atr, entry_index
                )
                pending = None

            # --- 2. exits, checked against this bar's range --------------------
            self._process_exits(
                portfolio, simulator, bar, timestamp, i, atr, performance, entry_index, regimes
            )

            # --- 3. mark to market ----------------------------------------------
            close = float(bar["close"])
            prices = {symbol: close}
            portfolio.track_excursions({symbol: float(bar["high"])})
            portfolio.track_excursions({symbol: float(bar["low"])})
            snapshot = portfolio.record_snapshot(timestamp, prices)

            # --- 4. decide, using data up to and including this bar -------------
            if i < warmup:
                continue

            state = graph.run(
                symbol=symbol,
                timeframe=data.timeframe.code,
                timestamp=timestamp,
                market_data=data,
                equity=snapshot.equity,
                features=features.upto(timestamp),
            )

            regime_label = str(state["regime"].regime) if state.get("regime") else "UNKNOWN"
            regimes.iloc[i] = regime_label

            decision = state.get("decision") or "UNKNOWN"
            blocked[decision] += 1
            decisions.append(self._decision_record(state, timestamp, snapshot.equity, regime_label))
            node_errors.extend(state.get("errors") or [])

            order = state.get("order")
            if order is not None and i + 1 < min(limit, len(df)):
                signal = state["final_signal"]
                risk = state["risk_decision"]
                pending = PendingOrder(
                    order=order,
                    stop_loss=float(signal.stop_loss),
                    take_profit=None if signal.take_profit is None else float(signal.take_profit),
                    direction=signal.direction,
                    risk_amount=risk.risk_amount,
                    regime=regime_label,
                    created_index=i,
                )

        # --- close anything still open ------------------------------------------
        if backtest_cfg.close_positions_at_end and portfolio.positions:
            final_index = min(limit, len(df)) - 1
            self._close_all(
                portfolio, simulator, df, final_index, performance, entry_index, regimes
            )
            # The final snapshot was taken while those positions were still
            # open, so it holds their mark-to-market value rather than their
            # realised value net of exit costs. Restate it, keeping one
            # snapshot per bar.
            if portfolio.snapshots:
                portfolio.snapshots.pop()
            portfolio.record_snapshot(
                df.index[final_index], {symbol: float(df.iloc[final_index]["close"])}
            )

        return self._build_result(
            data, portfolio, performance, decisions, node_errors, blocked,
            quality_status, quality_source, warmup,
        )

    # ------------------------------------------------------------ data quality

    def _resolve_quality(self, data: MarketData, supplied: str | None) -> tuple[str, str]:
        """Determine the grade this run is operating under, and where it came from.

        Grading here rather than trusting a default string is the difference
        between a gate and a label. The normal path arrives with a grade from
        :class:`MarketDataService`; anything else gets graded now, so there is
        no route from a corrupt frame to an open position that does not pass a
        quality check.
        """
        if supplied is not None:
            return str(supplied), "caller"

        try:
            report = DataQualityEngine(self.config.data.quality).validate(
                data, self.config.assets.assets.get(data.symbol.upper())
            )
        except Exception as exc:  # noqa: BLE001 - a failed grading is not a pass
            logger.warning(
                "Data-quality grading failed; the run proceeds as unverified, which the "
                "risk engine treats as a refusal",
                extra={"symbol": data.symbol, "error": str(exc)},
            )
            return "UNKNOWN", "grading-failed"

        return str(report.status), "self-graded"

    # -------------------------------------------------------------- bar steps

    def _open_position(
        self,
        pending: PendingOrder,
        bar: pd.Series,
        timestamp: pd.Timestamp,
        index: int,
        portfolio: Portfolio,
        simulator: ExecutionSimulator,
        atr: float | None,
        entry_index: dict[int, int],
    ) -> None:
        """Fill a queued order at this bar's open."""
        fill = simulator.execute(pending.order, bar, timestamp, atr)
        if not fill.is_filled:
            logger.debug(
                "Queued order not filled",
                extra={"reason": fill.rejection_reason, "symbol": pending.order.symbol},
            )
            return

        # The stop was computed from the signal bar's close, but the fill
        # happened at the next open. The stop level stays where the strategy put
        # it - moving it to preserve the intended risk would be rewriting the
        # trade after seeing the fill.
        if not self._stop_still_valid(pending.direction, fill.price, pending.stop_loss):
            logger.info(
                "Order abandoned: the open gapped through its stop",
                extra={
                    "symbol": pending.order.symbol, "fill": fill.price,
                    "stop": pending.stop_loss,
                },
            )
            return

        position = Position(
            symbol=pending.order.symbol,
            direction=pending.direction,
            quantity=fill.filled_quantity,
            entry_price=fill.price,
            entry_time=timestamp,
            stop_loss=pending.stop_loss,
            take_profit=pending.take_profit,
            strategy=pending.order.strategy,
            entry_regime=pending.regime,
            entry_costs=fill.total_cost,
            risk_amount=abs(fill.price - pending.stop_loss) * fill.filled_quantity,
            metadata={
                "signal_bar": str(pending.created_index),
                "reference_price": fill.reference_price,
                "slippage": fill.slippage_cost,
                "spread": fill.spread_cost,
            },
        )
        portfolio.open_position(position, fill)
        entry_index[id(position)] = index

    @staticmethod
    def _stop_still_valid(
        direction: SignalDirection, fill_price: float, stop: float
    ) -> bool:
        """Whether the fill left the stop on the correct side.

        If the market gapped past the stop before the entry filled, the trade
        is already invalid - taking it would mean opening a position that is
        instantly beyond its own stop, which no risk system would permit and
        which would produce a nonsensical negative risk distance.
        """
        if direction is SignalDirection.LONG:
            return stop < fill_price
        return stop > fill_price

    def _process_exits(
        self,
        portfolio: Portfolio,
        simulator: ExecutionSimulator,
        bar: pd.Series,
        timestamp: pd.Timestamp,
        index: int,
        atr: float | None,
        performance: RegimePerformanceTracker,
        entry_index: dict[int, int],
        regimes: pd.Series,
    ) -> None:
        """Close any position whose stop or target was touched this bar."""
        for position in list(portfolio.positions):
            event = simulator.check_exit(position, bar)
            if event is None:
                continue

            fill = simulator.exit_fill(position, event.price, timestamp, event.reason, atr)
            bars_held = index - entry_index.get(id(position), index)
            exit_regime = str(regimes.iloc[index]) if index < len(regimes) else "UNKNOWN"

            trade = portfolio.close_position(
                position, fill, timestamp, event.reason, exit_regime, bars_held
            )
            # The selector learns only from closed trades, which is what keeps
            # its weighting free of future information.
            performance.record_trade(trade.strategy, trade.entry_regime, trade.r_multiple)

    def _close_all(
        self,
        portfolio: Portfolio,
        simulator: ExecutionSimulator,
        df: pd.DataFrame,
        index: int,
        performance: RegimePerformanceTracker,
        entry_index: dict[int, int],
        regimes: pd.Series,
    ) -> None:
        """Close open positions at the final bar.

        Leaving them open would silently omit their outcome from every metric,
        which flatters a strategy that ends holding a loser.
        """
        timestamp = df.index[index]
        close = float(df.iloc[index]["close"])

        for position in list(portfolio.positions):
            fill = simulator.exit_fill(
                position, close, timestamp, ExitReason.END_OF_BACKTEST, None
            )
            bars_held = index - entry_index.get(id(position), index)
            trade = portfolio.close_position(
                position, fill, timestamp, ExitReason.END_OF_BACKTEST,
                str(regimes.iloc[index]) if index < len(regimes) else "UNKNOWN",
                bars_held,
            )
            performance.record_trade(trade.strategy, trade.entry_regime, trade.r_multiple)

    # ---------------------------------------------------------------- records

    @staticmethod
    def _decision_record(
        state: TradingState, timestamp: pd.Timestamp, equity: float, regime: str
    ) -> dict[str, Any]:
        """One row per analysed bar, including every no-trade decision.

        Section 67 of the brief requires the system to record *why* it did not
        trade. This is that record, and it is usually the majority of rows.
        """
        aggregated = state.get("aggregated")
        risk = state.get("risk_decision")
        signals = state.get("strategy_signals") or {}

        record: dict[str, Any] = {
            "timestamp": timestamp,
            "equity": equity,
            "regime": regime,
            "regime_confidence": state["regime"].confidence if state.get("regime") else None,
            "decision": state.get("decision"),
            "aggregated_direction": str(aggregated.direction) if aggregated else None,
            "aggregated_confidence": aggregated.confidence if aggregated else None,
            "risk_verdict": str(risk.verdict) if risk else None,
            "block_reason": str(risk.block_reason) if risk and risk.block_reason else None,
            "reason": "; ".join(aggregated.reasoning[:3]) if aggregated else None,
        }
        for name, signal in signals.items():
            record[f"signal_{name}"] = str(signal.direction)
        return record

    def _build_result(
        self,
        data: MarketData,
        portfolio: Portfolio,
        performance: RegimePerformanceTracker,
        decisions: list[dict[str, Any]],
        node_errors: list[dict[str, Any]],
        blocked: Counter[str],
        quality_status: str,
        quality_source: str,
        warmup: int,
    ) -> BacktestResult:
        trades = portfolio.trades_frame()
        equity_curve = portfolio.equity_curve()

        metrics = compute_metrics(
            equity_curve=equity_curve,
            trades=trades,
            initial_equity=portfolio.initial_balance,
            bars_per_year=self.config.backtest.bars_per_year,
            risk_free_rate=self.config.backtest.risk_free_rate,
            total_costs=portfolio.total_costs,
        )

        if not trades.empty:
            snapshots = portfolio.snapshots_frame()
            in_market = (snapshots["open_positions"] > 0).mean() if not snapshots.empty else 0.0
            metrics.exposure_pct = float(in_market)

        provenance = RunProvenance(
            experiment_id=self.experiment_id,
            symbol=data.symbol,
            timeframe=data.timeframe.code,
            start=str(data.start) if data.start is not None else None,
            end=str(data.end) if data.end is not None else None,
            bars=len(data),
            data_provider=data.provider,
            data_checksum=frame_checksum(data.df),
            data_quality=quality_status,
            data_quality_source=quality_source,
            random_seed=self.config.platform.random_seed,
            config_snapshot={
                "risk": self.config.risk.model_dump(),
                "execution": self.config.execution.model_dump(),
                "backtest": self.config.backtest.model_dump(),
                "features": self.config.features.model_dump(),
                "regimes": self.config.regimes.model_dump(),
                "strategies": self.config.strategies.model_dump(),
                "warmup_bars_used": warmup,
            },
        )

        return BacktestResult(
            provenance=provenance,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            snapshots=portfolio.snapshots_frame(),
            decisions=pd.DataFrame(decisions),
            regime_performance=performance.to_frame(),
            blocked_counts=dict(blocked),
            node_errors=node_errors,
        )
