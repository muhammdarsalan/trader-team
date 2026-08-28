"""Paper-trading loop and state persistence.

The loop mirrors :class:`app.backtest.engine.Backtester` bar-for-bar, and that is
the point: a paper result that used different sequencing from the backtest would
not be comparable with it, and the difference would be invisible. Each bar runs
the same four steps in the same order:

1. Fill the order queued at the previous bar, at this bar's OPEN.
2. Check exits for every open position against this bar's HIGH and LOW.
3. Track excursions, then mark to market at this bar's CLOSE and snapshot equity.
4. Run the decision graph using data up to and including this bar, and queue any
   resulting order for the *next* bar.

No step reads a value the timestamp being processed would not have had.

Two things are true of this engine that are not true of the backtester. It is
restartable, so state is persisted after every bar and reloaded on construction
- including the drawdown peak and daily-loss baseline, which cannot be derived
from cash and positions and whose loss would silently widen both risk limits.
And it can be driven from a live feed as well as from a historical frame; both
paths run the identical per-bar code, so a bug cannot hide in one of them.

This is simulation. Nothing here contacts a broker, and there is no code path,
flag or configuration that makes it do so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.config.loader import AppConfig, get_config
from app.config.models import AssetConfig
from app.data.schema import MarketData
from app.data.service import MarketDataService
from app.execution.models import ExitReason, Order, OrderSide, Position, Trade
from app.execution.simulator import ExecutionSimulator
from app.features.engine import FeatureEngine, FeatureSet
from app.graph.state import TradingState
from app.graph.workflow import TradingGraph
from app.paper_trading.graph_view import build_graph_visualization
from app.paper_trading.performance import build_performance
from app.paper_trading.records import (
    STATE_SCHEMA_VERSION,
    assess_freshness,
    build_decision_view,
    iso,
    jsonable,
    parse_timestamp,
    serialise_fill,
    serialise_order,
    serialise_position,
    serialise_snapshot,
)
from app.portfolio.portfolio import Portfolio, PortfolioSnapshot
from app.signals.models import SignalDirection
from app.signals.selector import RegimePerformanceTracker
from app.utils.logging import get_logger
from app.utils.timeutils import normalize_timeframe, utcnow

logger = get_logger(__name__)

#: Bounds on the persisted history. The state file is rewritten after every bar,
#: so an unbounded log would make each write slower than the last until a long
#: paper run spends more time serialising than deciding.
MAX_RECENT_BARS = 500
MAX_RECENT_EVENTS = 200
MAX_RECENT_ERRORS = 50
MAX_RECENT_FILLS = 200
MAX_RECENT_ORDERS = 200
MAX_EQUITY_POINTS = 5_000
MAX_CLOSED_TRADES = 1_000


class PaperTradingEngine:
    """Simulated trading over a live or historical feed, with restartable state.

    Args:
        config: resolved application config. Defaults to the process config.
        symbol: canonical platform symbol. Defaults to the first enabled asset.
        timeframe: bar interval, e.g. ``"1D"``.
        state_path: JSON file the engine persists to and reloads from. When
            ``None`` the engine is in-memory only, which is what tests of pure
            loop behaviour want.
        market_data_service: injected for the live path; not used by replay.
        portfolio: injected portfolio, mostly for tests. A reloaded state file
            replaces it, because the file is the authority on what has happened.
        asset: injected asset config.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        symbol: str | None = None,
        timeframe: str = "1D",
        state_path: str | Path | None = None,
        market_data_service: MarketDataService | None = None,
        portfolio: Portfolio | None = None,
        asset: AssetConfig | None = None,
    ) -> None:
        self.config = config or get_config()
        self.symbol = (symbol or self.config.assets.enabled_symbols()[0]).upper()
        self.timeframe = normalize_timeframe(timeframe).code
        self.asset = asset or self.config.assets.get(self.symbol)
        self.state_path = Path(state_path) if state_path is not None else None

        # The live path builds a service lazily: constructing one reaches for
        # provider configuration, and a replay-only or dashboard-only caller
        # should not pay for - or fail on - a provider it never uses.
        self._market_data_service = market_data_service

        self.portfolio = portfolio or Portfolio(
            self.config.platform.starting_balance, self.config.platform.base_currency
        )
        self.performance = RegimePerformanceTracker()
        self.trading_enabled = bool(self.config.platform.trading_enabled)
        self.graph = TradingGraph(
            config=self.config,
            portfolio=self.portfolio,
            asset=self.asset,
            performance=self.performance,
            trading_enabled=self.trading_enabled,
        )
        self.simulator = ExecutionSimulator(self.config.execution, self.asset)
        self.feature_engine = FeatureEngine(self.config.features)

        self._state: dict[str, Any] = self._blank_state()
        self._last_close: float | None = None
        self._features_cache: FeatureSet | None = None

        if self.state_path is not None and self.state_path.exists():
            self.load_state(self.state_path)

    # ------------------------------------------------------------------ state

    def _blank_state(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "mode": "PAPER",
            "live_trading_enabled": False,  # invariant, recorded so it is auditable
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "starting_balance": float(self.config.platform.starting_balance),
            "currency": self.config.platform.base_currency,
            "trading_enabled": self.trading_enabled,
            "cash": float(self.portfolio.cash),
            "realised_pnl": 0.0,
            "total_costs": 0.0,
            "risk_baselines": self.portfolio.risk_baselines(),
            "bar_counter": 0,
            "bars_processed": 0,
            "bars_decided": 0,
            "ambiguous_exits": 0,
            "last_processed_bar": None,
            "processed_bars": [],
            "pending_orders": [],
            "orders": [],
            "fills": [],
            "positions": [],
            "closed_trades": [],
            "equity_curve": [],
            "performance_records": [],
            "recent_events": [],
            "recent_errors": [],
            "market": {},
            "latest_decision": None,
            "latest_state": None,
            "last_bar": None,
            "last_quality": None,
            "last_regime": None,
            "last_close": None,
            "started_at": utcnow().isoformat(),
            "updated_at": None,
        }

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @property
    def market_data_service(self) -> MarketDataService:
        """The data service, built on first use."""
        if self._market_data_service is None:
            self._market_data_service = MarketDataService()
        return self._market_data_service

    # ------------------------------------------------------------ persistence

    def _serialise_state(self) -> dict[str, Any]:
        """The full persisted payload.

        Everything goes through :func:`jsonable` on the way out. That single
        boundary is what makes it impossible to reintroduce the failure this
        engine had: an ``Order`` dataclass placed in the event log used to raise
        ``TypeError`` inside ``json.dumps`` on the very first fill and abort the
        run, and no amount of care at individual call sites reliably prevents
        that recurring.
        """
        mark = self._mark_prices()
        payload = {
            **self._state,
            "schema_version": STATE_SCHEMA_VERSION,
            "cash": float(self.portfolio.cash),
            "realised_pnl": float(self.portfolio.realised_pnl),
            "total_costs": float(self.portfolio.total_costs),
            "risk_baselines": self.portfolio.risk_baselines(),
            "positions": [self._position_record(p, mark) for p in self.portfolio.positions],
            "closed_trades": [
                t.to_dict() for t in self.portfolio.closed_trades[-MAX_CLOSED_TRADES:]
            ],
            "equity_curve": [
                serialise_snapshot(s) for s in self.portfolio.snapshots[-MAX_EQUITY_POINTS:]
            ],
            "performance_records": [
                {"strategy": strategy, "regime": regime, "r_multiples": list(samples)}
                for (strategy, regime), samples in sorted(self.performance.records.items())
            ],
            "trading_enabled": self.trading_enabled,
            "live_trading_enabled": False,
            "last_close": self._last_close,
            "updated_at": utcnow().isoformat(),
        }
        return jsonable(payload)

    def _position_record(self, position: Position, mark: dict[str, float]) -> dict[str, Any]:
        row = serialise_position(position, mark.get(position.symbol))
        bounds = self.portfolio.excursion_bounds(position)
        if bounds is not None:
            row["excursion_low"], row["excursion_high"] = bounds
        return row

    def _persist(self) -> None:
        """Write state atomically.

        The engine rewrites the whole file after every bar. A partial write -
        from a crash, a full disk, or Ctrl-C at the wrong moment - would leave
        truncated JSON that the next start cannot parse, losing every position
        and the drawdown baseline with it. Writing to a temporary file and
        replacing means the visible file is always a complete state.
        """
        if self.state_path is None:
            return
        payload = self._serialise_state()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def save_state(self, path: str | Path | None = None) -> dict[str, Any]:
        """Persist to ``path`` (or the configured path) and return the payload."""
        if path is not None:
            self.state_path = Path(path)
        self._persist()
        return self._serialise_state()

    def load_state(self, path: str | Path | None = None) -> dict[str, Any]:
        """Reload persisted state, or start clean if there is none.

        A state file that cannot be parsed is moved aside rather than ignored or
        partially applied. Continuing from a half-read file would mean trading
        against positions and baselines that may or may not reflect reality,
        which is worse than starting flat and saying so.
        """
        target = Path(path) if path is not None else self.state_path
        if target is None or not target.exists():
            return self._serialise_state()

        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"state file holds {type(payload).__name__}, not an object")
        except (OSError, ValueError) as exc:
            quarantine = target.with_suffix(target.suffix + ".corrupt")
            try:
                target.replace(quarantine)
            except OSError:
                quarantine = None
            self._state = self._blank_state()
            self._record_error(
                f"The saved state at {target.name} could not be read ({exc}); the engine "
                "started flat instead of guessing. "
                + (
                    f"The unreadable file was kept as {quarantine.name} for inspection."
                    if quarantine
                    else "The unreadable file could not be moved aside."
                ),
                node="paper_trading",
                recoverable=False,
            )
            logger.error(
                "Paper state file unreadable; starting flat",
                extra={"path": str(target), "error": str(exc)},
            )
            return {}

        self._load_from_snapshot(payload)
        return payload

    def _load_from_snapshot(self, payload: dict[str, Any]) -> None:
        """Rebuild the engine from a persisted payload."""
        version = payload.get("schema_version")
        self._state = {**self._blank_state(), **payload}
        self._state["schema_version"] = STATE_SCHEMA_VERSION
        self._state["symbol"] = payload.get("symbol", self.symbol)
        self._state["timeframe"] = payload.get("timeframe", self.timeframe)
        # The kill switch is read from live configuration, never from the file:
        # a saved ``true`` must not be able to re-enable order creation that the
        # operator has since switched off.
        self._state["trading_enabled"] = self.trading_enabled
        self._state["live_trading_enabled"] = False

        self.portfolio = Portfolio(
            float(payload.get("starting_balance", self.config.platform.starting_balance)),
            payload.get("currency", self.config.platform.base_currency),
        )
        self.portfolio.cash = float(payload.get("cash", self.portfolio.initial_balance))
        self.portfolio.realised_pnl = float(payload.get("realised_pnl", 0.0))
        self.portfolio.total_costs = float(payload.get("total_costs", 0.0))

        self._restore_positions(payload.get("positions") or [])
        self._restore_trades(payload.get("closed_trades") or [])
        self._restore_equity_curve(payload.get("equity_curve") or [])
        self._restore_performance(payload.get("performance_records") or [])
        self._restore_risk_baselines(payload.get("risk_baselines") or {})

        self._last_close = payload.get("last_close")

        # The graph holds a reference to the portfolio for position and risk
        # checks, so it has to be pointed at the rebuilt one. Missing this would
        # leave risk sizing against an empty portfolio while the real one holds
        # open positions.
        self.graph = TradingGraph(
            config=self.config,
            portfolio=self.portfolio,
            asset=self.asset,
            performance=self.performance,
            trading_enabled=self.trading_enabled,
        )

        if version != STATE_SCHEMA_VERSION:
            self._record_event(
                "state_schema_mismatch",
                f"Loaded state written by schema {version!r}; this build expects "
                f"{STATE_SCHEMA_VERSION}. Fields the older layout did not carry have "
                "started from their defaults.",
                loaded_version=version,
                expected_version=STATE_SCHEMA_VERSION,
            )

        logger.info(
            "Paper state restored",
            extra={
                "symbol": self._state["symbol"],
                "cash": round(self.portfolio.cash, 2),
                "open_positions": len(self.portfolio.positions),
                "closed_trades": len(self.portfolio.closed_trades),
                "bars_processed": self._state.get("bars_processed"),
            },
        )

    def _restore_positions(self, rows: list[dict[str, Any]]) -> None:
        for item in rows:
            try:
                metadata = dict(item.get("metadata") or {})
                position = Position(
                    symbol=item["symbol"],
                    direction=SignalDirection[str(item["direction"]).upper()],
                    quantity=float(item["quantity"]),
                    entry_price=float(item["entry_price"]),
                    entry_time=parse_timestamp(item.get("entry_time")) or utcnow(),
                    stop_loss=float(item["stop_loss"]),
                    take_profit=item.get("take_profit"),
                    strategy=item.get("strategy", "unknown"),
                    entry_regime=item.get("entry_regime", "UNKNOWN"),
                    entry_costs=float(item.get("entry_costs", 0.0)),
                    risk_amount=float(item.get("risk_amount", 0.0)),
                    metadata=metadata,
                )
            except (KeyError, TypeError, ValueError) as exc:
                self._record_error(
                    f"A saved open position could not be restored ({exc}). It is not "
                    "being tracked, so its outcome will be missing from results.",
                    node="paper_trading",
                    position=jsonable(item),
                )
                continue
            self.portfolio.positions.append(position)
            low = item.get("excursion_low", position.entry_price)
            high = item.get("excursion_high", position.entry_price)
            self.portfolio.restore_excursions(position, float(low), float(high))

    def _restore_trades(self, rows: list[dict[str, Any]]) -> None:
        for raw in rows:
            try:
                trade = Trade(
                    symbol=raw["symbol"],
                    strategy=raw["strategy"],
                    direction=SignalDirection[str(raw["direction"]).upper()],
                    quantity=float(raw["quantity"]),
                    entry_time=parse_timestamp(raw["entry_time"]) or utcnow(),
                    entry_price=float(raw["entry_price"]),
                    exit_time=parse_timestamp(raw["exit_time"]) or utcnow(),
                    exit_price=float(raw["exit_price"]),
                    stop_loss=float(raw["stop_loss"]),
                    take_profit=raw.get("take_profit"),
                    exit_reason=self._parse_exit_reason(raw.get("exit_reason")),
                    gross_pnl=float(raw.get("gross_pnl", 0.0)),
                    costs=float(raw.get("costs", 0.0)),
                    net_pnl=float(raw.get("net_pnl", 0.0)),
                    r_multiple=raw.get("r_multiple"),
                    entry_regime=raw.get("entry_regime", "UNKNOWN"),
                    exit_regime=raw.get("exit_regime", "UNKNOWN"),
                    bars_held=int(raw.get("bars_held", 0)),
                    mae=float(raw.get("mae", 0.0)),
                    mfe=float(raw.get("mfe", 0.0)),
                    metadata={k: v for k, v in raw.items() if k.startswith("meta_")},
                )
            except (KeyError, TypeError, ValueError) as exc:
                self._record_error(
                    f"A saved closed trade could not be restored ({exc}); performance "
                    "statistics will understate the number of trades.",
                    node="paper_trading",
                )
                continue
            self.portfolio.closed_trades.append(trade)

    @staticmethod
    def _parse_exit_reason(value: Any) -> ExitReason:
        if isinstance(value, ExitReason):
            return value
        try:
            return ExitReason[str(value).upper()]
        except KeyError:
            return ExitReason.RISK_LIMIT

    def _restore_equity_curve(self, rows: list[dict[str, Any]]) -> None:
        """Reinstate the equity curve so the dashboard chart survives a restart."""
        for row in rows:
            timestamp = parse_timestamp(row.get("timestamp"))
            if timestamp is None:
                continue
            try:
                self.portfolio.snapshots.append(
                    PortfolioSnapshot(
                        timestamp=timestamp,
                        cash=float(row["cash"]),
                        equity=float(row["equity"]),
                        unrealised_pnl=float(row.get("unrealised_pnl", 0.0)),
                        realised_pnl=float(row.get("realised_pnl", 0.0)),
                        open_positions=int(row.get("open_positions", 0)),
                        exposure=float(row.get("exposure", 0.0)),
                        drawdown=float(row.get("drawdown", 0.0)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

    def _restore_performance(self, rows: list[dict[str, Any]]) -> None:
        """Reinstate the selector's realised-performance table.

        The selector weights strategies by measured expectancy per regime. If
        that table reset on every restart, strategy weights would silently jump
        back to their priors and the paper run's behaviour would depend on when
        it was last restarted.
        """
        for row in rows:
            strategy = row.get("strategy")
            regime = row.get("regime")
            samples = row.get("r_multiples") or []
            if not strategy or not regime:
                continue
            for value in samples:
                try:
                    self.performance.records[(str(strategy), str(regime))].append(float(value))
                except (TypeError, ValueError):
                    continue

    def _restore_risk_baselines(self, baselines: dict[str, Any]) -> None:
        peak = baselines.get("peak_equity")
        if peak is None:
            # An older state file, or one written before baselines were tracked.
            # Falling back to the recorded equity peak is closer to the truth
            # than the starting balance, and the gap is recorded either way.
            equities = [s.equity for s in self.portfolio.snapshots]
            peak = max(equities) if equities else 0.0
            if peak > 0:
                self._record_event(
                    "risk_baselines_reconstructed",
                    "The saved state carried no drawdown baseline, so it was "
                    f"reconstructed from the persisted equity curve (peak ${peak:,.2f}).",
                    peak_equity=peak,
                )
        try:
            self.portfolio.restore_risk_baselines(
                peak_equity=float(peak or 0.0),
                day_start_equity=baselines.get("day_start_equity"),
                current_day=parse_timestamp(baselines.get("current_day")),
            )
        except (TypeError, ValueError) as exc:
            self._record_error(
                f"The saved drawdown baseline could not be restored ({exc}). Drawdown and "
                "daily-loss limits are measured from the starting balance until the next "
                "equity peak, which makes them looser than they should be.",
                node="risk",
            )

    # -------------------------------------------------------------- recording

    def _record_event(self, event: str, message: str, **payload: Any) -> None:
        entry = jsonable(
            {"event": event, "message": message, "timestamp": utcnow().isoformat(), **payload}
        )
        recent = list(self._state.get("recent_events") or [])
        recent.append(entry)
        self._state["recent_events"] = recent[-MAX_RECENT_EVENTS:]

    def _record_error(self, message: str, *, node: str = "paper_trading", **payload: Any) -> None:
        entry = jsonable(
            {"node": node, "error": message, "timestamp": utcnow().isoformat(), **payload}
        )
        recent = list(self._state.get("recent_errors") or [])
        recent.append(entry)
        self._state["recent_errors"] = recent[-MAX_RECENT_ERRORS:]
        logger.warning("Paper trading error recorded", extra={"node": node, "detail": message})

    @staticmethod
    def _read_quality(value: Any) -> tuple[str | None, dict[str, Any] | None]:
        """Split a quality argument into its grade and its full report.

        Callers may pass either the grade string or the
        :class:`~app.data.validators.quality.DataQualityReport` itself. Accepting
        the report matters: the grade alone tells a reader that a feed was graded
        WARNING and not *why*, and the why is the whole point for the FX feeds -
        an OHLC-bound defect in 3% of bars is a different fact from a single
        stale bar, and only the findings distinguish them.
        """
        if value is None:
            return None, None
        if isinstance(value, str):
            return value, None
        status = getattr(value, "status", None)
        to_dict = getattr(value, "to_dict", None)
        if status is None:
            return str(value), None
        report = None
        if callable(to_dict):
            try:
                report = jsonable(to_dict())
            except Exception:  # noqa: BLE001 - a report that will not render is not fatal
                report = None
        return str(status), report

    # ------------------------------------------------------------- market data

    def refresh_market_data(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        bars: int = 500,
        *,
        validate: bool = True,
    ) -> MarketData | None:
        """Fetch the newest bars for the live path.

        Returns ``None`` on any failure - a rejected quality grade, an
        unreachable provider, an empty response - after recording what went
        wrong and marking the market lane ``ERROR``. The last good data is left
        in place and untouched. Nothing is fabricated and no stale frame is
        relabelled as current: the freshness field carries the real age, so a
        reader can see that the engine is running on old bars.
        """
        symbol = (symbol or self.symbol).upper()
        timeframe = normalize_timeframe(timeframe or self.timeframe).code

        try:
            result = self.market_data_service.get_latest_data(
                symbol, timeframe, bars=bars, validate=validate
            )
        except Exception as exc:  # noqa: BLE001 - any failure must fail safe, not raise
            self._record_error(
                f"Could not refresh {symbol} {timeframe}: {exc}. The engine is still "
                "showing the last data it successfully loaded; check the freshness "
                "field before reading anything into the current state.",
                node="market_data",
                symbol=symbol,
                timeframe=timeframe,
                error_type=type(exc).__name__,
            )
            # Built on the previous lane, so the last good bar and its grade
            # stay visible - relabelled ERROR rather than blanked, because
            # "we are showing you old data" is the useful statement.
            market = {**self._blank_market(), **(self._state.get("market") or {})}
            market.update(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                    "refresh_failed_at": utcnow().isoformat(),
                }
            )
            self._state["market"] = self._decorate_market(market)
            return None

        if result.data.df.empty:
            self._record_error(
                f"The provider returned no bars for {symbol} {timeframe}. Nothing was "
                "processed; the previous data is unchanged.",
                node="market_data",
                symbol=symbol,
                timeframe=timeframe,
            )
            market = {**self._blank_market(), **(self._state.get("market") or {})}
            market.update(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": "EMPTY",
                    "rows": 0,
                    "error": "The provider returned no bars.",
                    "refresh_failed_at": utcnow().isoformat(),
                }
            )
            self._state["market"] = self._decorate_market(market)
            return None

        df = result.data.df
        market = {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "OK",
            "rows": int(len(df)),
            "first_bar": iso(df.index[0]),
            "last_bar": iso(df.index[-1]),
            "last_close": float(df["close"].iloc[-1]),
            "quality_status": str(result.quality.status),
            "quality": jsonable(result.quality.to_dict()),
            "provider": result.data.provider,
            "fetched_at": utcnow().isoformat(),
            "source": "live_refresh",
        }
        self._state["market"] = self._decorate_market(market)
        self._state["last_quality"] = str(result.quality.status)
        self._record_event(
            "market_data_refreshed",
            f"Loaded {len(df):,} bars of {symbol} {timeframe} from "
            f"{result.data.provider}; quality {result.quality.status}.",
            symbol=symbol,
            timeframe=timeframe,
            rows=int(len(df)),
            quality_status=str(result.quality.status),
            last_bar=iso(df.index[-1]),
        )
        return result.data

    def _blank_market(self, **overrides: Any) -> dict[str, Any]:
        """The market lane with every field a consumer reads, explicitly empty.

        A missing key and a null carry the same fact, but only the null survives
        a caller that indexes rather than gets - and a KeyError in a monitoring
        surface reads as a bug in the engine. ``quality_status`` defaults to
        ``UNKNOWN`` rather than to None because that is the value the risk engine
        will act on, and it refuses on it.
        """
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "status": "NO_DATA",
            "rows": 0,
            "quality_status": "UNKNOWN",
            "quality": None,
            "provider": None,
            "source": None,
            "first_bar": None,
            "last_bar": None,
            "last_close": None,
            "error": None,
            **overrides,
        }

    def _decorate_market(self, market: dict[str, Any]) -> dict[str, Any]:
        """Attach freshness and the instrument's data caveat to a market summary.

        The caveat travels with the data rather than living in the dashboard,
        because every surface that shows a price for a proxy series has to show
        that it is a proxy - and only one of those surfaces is the dashboard.
        """
        market = dict(market)
        timeframe = market.get("timeframe") or self.timeframe
        market["freshness"] = assess_freshness(
            market.get("last_bar"),
            timeframe,
            weekday_only=(self.asset.trading_days == "WEEKDAYS" if self.asset else True),
        ).to_dict()
        if self.asset is not None:
            market["instrument_name"] = self.asset.name
            market["is_proxy"] = bool(self.asset.is_proxy)
            market["data_caveat"] = self.asset.data_caveat
            market["provider_symbol"] = self.asset.provider_symbols.get(
                self.config.data.default_provider
            )
            market["has_reliable_volume"] = bool(self.asset.has_reliable_volume)
        return market

    def market_summary(self) -> dict[str, Any]:
        """The market lane, or an explicit NO_DATA marker when nothing is loaded."""
        market = self._state.get("market") or {}
        if not market:
            return self._decorate_market(self._blank_market())
        if "freshness" not in market:
            return self._decorate_market(market)
        return market

    # ----------------------------------------------------------- bar sequence

    def process_bar(
        self,
        market_data: MarketData,
        *,
        timestamp: pd.Timestamp | None = None,
        features: FeatureSet | None = None,
        data_quality: Any = None,
        quality_report: dict[str, Any] | None = None,
        force: bool = False,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Run one bar through the full paper sequence.

        The four steps run in the order documented at the top of this module,
        matching the backtester exactly. ``force`` re-runs a bar that was already
        processed; without it a repeat is skipped, which is what makes a restart
        that replays overlapping history safe.

        Args:
            market_data: the frame to read this bar from.
            timestamp: the bar to process. Defaults to the frame's last bar.
            features: precomputed causal features, truncated per bar internally.
            data_quality: the grade this bar is sized against - a status string,
                or the :class:`~app.data.validators.quality.DataQualityReport`
                itself, in which case its findings are recorded too. Absent means
                unverified, which the risk engine treats as a refusal.
            quality_report: the findings, when the grade was passed separately.
            force: re-run a bar that has already been processed.
            persist: write state after this bar.
        """
        df = market_data.df
        if df.empty:
            raise ValueError("Cannot process an empty market data frame")

        timestamp = pd.Timestamp(df.index[-1] if timestamp is None else timestamp)
        if timestamp not in df.index:
            raise KeyError(
                f"{timestamp} is not a bar in the supplied {market_data.symbol} frame"
            )

        if not force and self._already_processed(timestamp):
            return {
                "status": "skipped",
                "reason": "This bar has already been processed.",
                "bar": timestamp.isoformat(),
                "latest_decision": self._state.get("latest_decision"),
            }

        symbol = market_data.symbol
        grade, report = self._read_quality(data_quality)
        quality = grade or self._state.get("last_quality")
        report = quality_report if quality_report is not None else report

        if features is None:
            features = self._features_for(market_data)
        atr = self._atr_at(features, timestamp)

        position_in_frame = df.index.get_indexer([timestamp])[0]
        bar = df.iloc[position_in_frame]
        self._state["bar_counter"] = int(self._state.get("bar_counter", 0)) + 1
        bar_index = self._state["bar_counter"]

        # --- 1. fill the order queued at the previous bar, at this OPEN --------
        self._fill_pending_orders(bar, timestamp, atr, bar_index)

        # --- 2. exits, checked against this bar's range ------------------------
        self._process_exits(bar, timestamp, atr, bar_index)

        # --- 3. excursions, then mark to market at this CLOSE ------------------
        close = float(bar["close"])
        self.portfolio.track_excursions({symbol: float(bar["high"])})
        self.portfolio.track_excursions({symbol: float(bar["low"])})
        snapshot = self.portfolio.record_snapshot(timestamp, {symbol: close})
        self._last_close = close
        self._trim_equity_curve()

        # --- 4. decide, using data up to and including this bar ----------------
        warm = self._is_warm(features, timestamp)
        if warm:
            state = self._decide_on_bar(
                market_data,
                timestamp,
                features=features,
                equity=snapshot.equity,
                data_quality=quality,
            )
            if state is not None:
                self._queue_order(state, timestamp)
                self._state["bars_decided"] = int(self._state.get("bars_decided", 0)) + 1
        else:
            reason = (
                f"Bar {timestamp.date()} is inside the {self.warmup_bars(features)}-bar "
                "feature warm-up, so no decision was taken. Indicators are not yet "
                "defined and a signal from them would be meaningless."
            )
            self._state["latest_decision"] = "WARMUP"
            self._state["latest_state"] = build_decision_view(
                {"symbol": symbol, "timeframe": market_data.timeframe.code,
                 "timestamp": timestamp, "equity": snapshot.equity, "decision": "WARMUP"},
                trading_enabled=self.trading_enabled,
                data_quality=quality,
                warm=False,
                skipped_reason=reason,
            )

        self._mark_processed(timestamp)
        self._state["last_bar"] = timestamp.isoformat()
        self._state["last_quality"] = quality or "UNKNOWN"
        self._state["last_close"] = close
        self._state["bars_processed"] = int(self._state.get("bars_processed", 0)) + 1
        self._sync_market_from_frame(market_data, timestamp, quality, report)

        if persist:
            self._persist()
        return self.snapshot()

    def catch_up(
        self,
        market_data: MarketData,
        *,
        features: FeatureSet | None = None,
        data_quality: Any = None,
        persist_every: int = 250,
    ) -> list[dict[str, Any]]:
        """Replay every unprocessed bar in ``market_data``, oldest first.

        Features are computed once over the whole frame and each bar is handed a
        set truncated to that bar, exactly as the backtester does. That is safe
        because every feature is causal, and it is what keeps a 3,000-bar replay
        from recomputing indicators 3,000 times.

        ``persist_every`` batches the state write. Writing after every bar during
        a long historical catch-up dominates the run for no benefit; the final
        state is always written.

        ``data_quality`` takes either the grade or the whole
        :class:`~app.data.validators.quality.DataQualityReport`. Prefer the
        report: the dashboard shows the gate's findings, and a grade on its own
        cannot say what was wrong with the feed.
        """
        if market_data.df.empty:
            self._record_error(
                "catch_up was given an empty frame; there was nothing to replay.",
                node="paper_trading",
            )
            return []

        if features is None:
            features = self._features_for(market_data)

        # Split the grade from the report once, rather than per bar: the report
        # can hold thousands of characters and every bar would re-serialise it.
        grade, report = self._read_quality(data_quality)

        outputs: list[dict[str, Any]] = []
        pending_writes = 0
        for timestamp in market_data.df.index:
            if self._already_processed(timestamp):
                continue
            outputs.append(
                self.process_bar(
                    market_data,
                    timestamp=timestamp,
                    features=features,
                    data_quality=grade,
                    quality_report=report,
                    persist=False,
                )
            )
            pending_writes += 1
            if persist_every > 0 and pending_writes >= persist_every:
                self._persist()
                pending_writes = 0

        if outputs:
            self._persist()
        return outputs

    def run_live_tick(
        self,
        *,
        bars: int = 500,
        validate: bool = True,
    ) -> dict[str, Any]:
        """Fetch the newest bars and process whatever has not been seen yet.

        This is the live-ish path. It shares every line of per-bar logic with
        replay - only the source of the frame differs - so behaviour cannot
        diverge between the two. A failed fetch returns a ``failed`` status and
        changes no portfolio state.
        """
        data = self.refresh_market_data(bars=bars, validate=validate)
        if data is None:
            self._persist()
            return {
                "status": "failed",
                "reason": "Market data could not be refreshed; no bars were processed.",
                "market": self.market_summary(),
                "errors": list(self._state.get("recent_errors") or [])[-3:],
            }

        # refresh_market_data already recorded the full report on the market
        # lane; pass the grade so process_bar does not overwrite it with None.
        quality = self._state.get("last_quality")
        outputs = self.catch_up(data, data_quality=quality, persist_every=0)
        self._persist()
        return {
            "status": "ok" if outputs else "no_new_bars",
            "bars_processed": len(outputs),
            "market": self.market_summary(),
            "latest_decision": self._state.get("latest_decision"),
            "portfolio": self.portfolio_snapshot(),
        }

    # -------------------------------------------------------------- bar steps

    def _features_for(self, market_data: MarketData) -> FeatureSet:
        """Compute features for this frame, reusing the last set when possible."""
        cached = self._features_cache
        if (
            cached is not None
            and cached.symbol == market_data.symbol
            and cached.timeframe == market_data.timeframe.code
            and len(cached) == len(market_data.df)
            and not cached.df.empty
            and not market_data.df.empty
            and cached.df.index[-1] == market_data.df.index[-1]
        ):
            return cached
        computed = self.feature_engine.compute(market_data, self.asset)
        self._features_cache = computed
        return computed

    @staticmethod
    def _atr_at(features: FeatureSet | None, timestamp: pd.Timestamp) -> float | None:
        """This bar's ATR, for the slippage model.

        The simulator's ``atr_fraction`` slippage model needs ATR. Omitting it
        does not disable slippage - it falls back to a flat 0.05%, silently
        replacing the configured model with a different one. Passing the real
        value is the difference between the paper run and the backtest using the
        same cost assumptions.
        """
        df = getattr(features, "df", None)
        if df is None or df.empty or "atr" not in df.columns or timestamp not in df.index:
            return None
        value = df["atr"].loc[timestamp]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        if pd.isna(value):
            return None
        atr = float(value)
        return atr if atr > 0 else None

    def warmup_bars(self, features: FeatureSet | None) -> int:
        """Bars of history required before a decision is taken.

        ``backtest.warmup_bars`` wins when set, exactly as in
        :class:`app.backtest.engine.Backtester`. Reading it here is not an
        aesthetic nicety: if the two engines resolved warm-up differently, an
        operator who lengthened it to stabilise a backtest would find the paper
        run still deciding on bars the backtest had discarded, and the two
        results would stop being comparable without anything looking wrong.
        """
        configured = self.config.backtest.warmup_bars
        if configured:
            return int(configured)
        return int(getattr(features, "warmup_bars", 0) or 0)

    def _is_warm(self, features: FeatureSet | None, timestamp: pd.Timestamp) -> bool:
        if features is None:
            return False
        df = getattr(features, "df", None)
        if df is None or timestamp not in df.index:
            return False
        return int(df.index.get_loc(timestamp)) >= self.warmup_bars(features)

    def _fill_pending_orders(
        self,
        bar: pd.Series,
        timestamp: pd.Timestamp,
        atr: float | None,
        bar_index: int,
    ) -> None:
        """Fill orders queued on the previous bar, at this bar's open.

        Every queued order is resolved here - filled, rejected, or abandoned -
        and then removed. Leaving an unfilled order in the queue would let a
        market order from weeks ago fill on some later bar, which is not
        something any real order would do.
        """
        pending_orders = list(self._state.get("pending_orders") or [])
        if not pending_orders:
            return
        self._state["pending_orders"] = []

        for pending in pending_orders:
            try:
                order = self._rebuild_order(pending)
            except (KeyError, TypeError, ValueError) as exc:
                self._record_error(
                    f"A queued paper order could not be rebuilt ({exc}) and was dropped.",
                    node="execution",
                    order=jsonable(pending),
                )
                continue

            fill = self.simulator.execute(order, bar, timestamp, atr)
            if not fill.is_filled:
                self._record_event(
                    "paper_order_rejected",
                    f"The queued {order.side} order for {order.symbol} did not fill: "
                    f"{fill.rejection_reason}. It was discarded rather than carried "
                    "forward to a later bar.",
                    symbol=order.symbol,
                    reason=fill.rejection_reason,
                    order_key=pending.get("fill_key"),
                )
                continue

            direction = (
                SignalDirection.LONG if order.side is OrderSide.BUY else SignalDirection.SHORT
            )
            stop_loss = pending.get("stop_loss")
            if stop_loss is None:
                self._record_error(
                    f"The queued order for {order.symbol} carried no stop, so it was "
                    "abandoned. An unstopped position has undefined risk and cannot be "
                    "sized or limited.",
                    node="execution",
                    order_key=pending.get("fill_key"),
                )
                continue
            stop_loss = float(stop_loss)

            if not self._stop_still_valid(direction, fill.price, stop_loss):
                self._record_event(
                    "paper_order_abandoned",
                    f"The open gapped through the stop for {order.symbol}: filled at "
                    f"{fill.price:.6g} with the stop at {stop_loss:.6g}. The trade was "
                    "abandoned rather than opened already beyond its own stop.",
                    symbol=order.symbol,
                    fill_price=float(fill.price),
                    stop_loss=stop_loss,
                    order_key=pending.get("fill_key"),
                )
                continue

            take_profit = pending.get("take_profit")
            position = Position(
                symbol=order.symbol,
                direction=direction,
                quantity=fill.filled_quantity,
                entry_price=fill.price,
                entry_time=timestamp,
                stop_loss=stop_loss,
                take_profit=None if take_profit is None else float(take_profit),
                strategy=order.strategy,
                entry_regime=pending.get("regime", "UNKNOWN"),
                entry_costs=fill.total_cost,
                # Risk is recomputed from the actual fill, not from the size the
                # signal assumed. The fill moved with the spread and slippage, so
                # the money genuinely at risk is not what was intended.
                risk_amount=abs(fill.price - stop_loss) * fill.filled_quantity,
                metadata={
                    "entry_bar": bar_index,
                    "order_key": pending.get("fill_key"),
                    "reference_price": float(fill.reference_price),
                    "spread": float(fill.spread_cost),
                    "slippage": float(fill.slippage_cost),
                    "commission": float(fill.commission),
                    "intended_risk": pending.get("risk_amount"),
                    # Survives into Trade.metadata, and from there into the
                    # persisted record as meta_contributing, so the performance
                    # table can attribute an ensemble trade to the strategies
                    # that voted for it.
                    "contributing": list(pending.get("contributing") or []),
                    "aggregation_method": pending.get("aggregation_method"),
                },
            )
            self.portfolio.open_position(position, fill)

            fill_record = serialise_fill(fill, timestamp)
            fill_record["fill_key"] = pending.get("fill_key")
            fill_record["stop_loss"] = stop_loss
            fill_record["take_profit"] = position.take_profit
            self._append_bounded("fills", fill_record, MAX_RECENT_FILLS)
            self._record_event(
                "paper_fill",
                f"Filled {order.symbol} {order.side} {fill.filled_quantity:.6g} at "
                f"{fill.price:.6g} (reference {fill.reference_price:.6g}, costs "
                f"${fill.total_cost:,.2f}).",
                symbol=order.symbol,
                side=str(order.side),
                price=float(fill.price),
                quantity=float(fill.filled_quantity),
                total_cost=float(fill.total_cost),
                order_key=pending.get("fill_key"),
            )

    def _rebuild_order(self, pending: dict[str, Any]) -> Order:
        return Order(
            symbol=pending["symbol"],
            side=OrderSide[str(pending["side"]).upper()],
            quantity=float(pending["quantity"]),
            created_at=parse_timestamp(pending.get("created_at")),
            strategy=pending.get("strategy", "unknown"),
            reference_price=float(pending.get("reference_price") or 0.0),
            metadata=dict(pending.get("metadata") or {}),
        )

    @staticmethod
    def _stop_still_valid(
        direction: SignalDirection, fill_price: float, stop: float
    ) -> bool:
        """Whether the fill left the stop on the correct side.

        Same guard as the backtester. If the market gapped past the stop before
        the entry filled, opening the position would create one that is already
        beyond its own stop and whose risk distance is negative.
        """
        if direction is SignalDirection.LONG:
            return stop < fill_price
        return stop > fill_price

    def _process_exits(
        self,
        bar: pd.Series,
        timestamp: pd.Timestamp,
        atr: float | None,
        bar_index: int,
    ) -> None:
        """Close any position whose stop or target was touched this bar.

        The exit regime is the last regime the graph *detected*, which is from
        the previous bar. This bar's regime is not known yet - detection happens
        in step 4 - and using it would be reading a value from the future of this
        step.
        """
        exit_regime = str(self._state.get("last_regime") or "UNKNOWN")

        for position in list(self.portfolio.positions):
            event = self.simulator.check_exit(position, bar)
            if event is None:
                continue

            fill = self.simulator.exit_fill(position, event.price, timestamp, event.reason, atr)
            entry_bar = position.metadata.get("entry_bar")
            try:
                bars_held = max(0, bar_index - int(entry_bar))
            except (TypeError, ValueError):
                bars_held = 0

            trade = self.portfolio.close_position(
                position, fill, timestamp, event.reason, exit_regime, bars_held
            )
            # The selector learns only from closed trades, which is what keeps
            # its weighting free of future information.
            self.performance.record_trade(trade.strategy, trade.entry_regime, trade.r_multiple)

            ambiguous = bool(getattr(event, "ambiguous", False))
            if ambiguous:
                # The bar's range contained both the stop and the target, and
                # OHLC cannot say which was touched first. The simulator resolves
                # it per execution.same_bar_resolution; recording the fact means
                # a reader can see which trades rest on that assumption rather
                # than on observed sequence.
                self._state["ambiguous_exits"] = (
                    int(self._state.get("ambiguous_exits") or 0) + 1
                )

            record = jsonable(trade.to_dict())
            record["ambiguous_bar"] = ambiguous
            self._append_bounded("closed_trades", record, MAX_CLOSED_TRADES)
            exit_fill = serialise_fill(fill, timestamp)
            exit_fill["exit_reason"] = str(event.reason)
            exit_fill["closes_position"] = True
            exit_fill["ambiguous_bar"] = ambiguous
            self._append_bounded("fills", exit_fill, MAX_RECENT_FILLS)
            self._record_event(
                "paper_exit",
                f"Closed {trade.symbol} {trade.direction} at {trade.exit_price:.6g} "
                f"({event.reason}). Net ${trade.net_pnl:,.2f}"
                + (
                    f" ({trade.r_multiple:+.2f}R)."
                    if trade.r_multiple is not None
                    else " (R multiple undefined)."
                )
                + (
                    " This bar's range contained both the stop and the target, so which "
                    f"came first is unknowable from OHLC; resolved as "
                    f"{self.config.execution.same_bar_resolution}."
                    if ambiguous
                    else ""
                ),
                symbol=trade.symbol,
                strategy=trade.strategy,
                reason=str(event.reason),
                net_pnl=float(trade.net_pnl),
                r_multiple=trade.r_multiple,
                bars_held=bars_held,
                ambiguous_bar=ambiguous,
            )

    def _decide_on_bar(
        self,
        market_data: MarketData,
        timestamp: pd.Timestamp,
        *,
        features: FeatureSet,
        equity: float,
        data_quality: str | None,
    ) -> TradingState | None:
        """Run the decision graph for one bar and record the full decision.

        A graph failure is caught here. The paper loop must keep marking equity
        and managing open positions even when a decision cannot be made; raising
        would abandon live positions mid-run, which is a worse outcome than a
        recorded bar with no decision.
        """
        try:
            state = self.graph.run(
                symbol=market_data.symbol,
                timeframe=market_data.timeframe.code,
                timestamp=timestamp,
                market_data=market_data,
                equity=equity,
                features=features.upto(timestamp),
                data_quality=data_quality,
            )
        except Exception as exc:  # noqa: BLE001 - a failed decision is not a trade
            self._record_error(
                f"The decision graph failed on {timestamp.isoformat()}: {exc}. No order "
                "was created for this bar; open positions are still being managed.",
                node="graph",
                error_type=type(exc).__name__,
                timestamp=timestamp.isoformat(),
            )
            self._state["latest_decision"] = "GRAPH_ERROR"
            self._state["latest_state"] = build_decision_view(
                {
                    "symbol": market_data.symbol,
                    "timeframe": market_data.timeframe.code,
                    "timestamp": timestamp,
                    "equity": equity,
                    "decision": "GRAPH_ERROR",
                    "errors": [{"node": "graph", "error": str(exc)}],
                },
                trading_enabled=self.trading_enabled,
                data_quality=data_quality,
                warm=True,
                skipped_reason=f"The decision graph raised {type(exc).__name__}.",
            )
            return None

        regime = state.get("regime")
        self._state["last_regime"] = str(regime.regime) if regime else "UNKNOWN"
        self._state["latest_decision"] = state.get("decision")
        self._state["latest_state"] = build_decision_view(
            state,
            trading_enabled=self.trading_enabled,
            data_quality=data_quality,
            warm=True,
        )

        for error in state.get("errors") or []:
            self._record_error(
                str(error.get("error", "A graph node reported an error.")),
                node=str(error.get("node", "graph")),
                timestamp=timestamp.isoformat(),
            )
        return state

    def _queue_order(self, state: TradingState, timestamp: pd.Timestamp) -> None:
        """Queue an approved order for the next bar's open.

        The stop and target are captured from the signal now rather than being
        recomputed at fill time. Recomputing them from the fill would move the
        stop after seeing where the entry landed, which quietly rewrites the
        trade's risk in its own favour.
        """
        order = state.get("order")
        if order is None:
            return

        signal = state.get("final_signal")
        risk = state.get("risk_decision")
        if signal is None or signal.stop_loss is None:
            self._record_error(
                "An order was produced without a stop level, so it was not queued. "
                "Position size and risk limits are both defined by the stop distance.",
                node="execution",
                timestamp=timestamp.isoformat(),
            )
            return

        order_key = (
            f"{order.symbol}:{order.side}:{timestamp.isoformat()}:"
            f"{order.quantity:.10g}:{order.strategy}"
        )
        # Which strategies actually voted for this trade. The aggregator hands
        # risk a single signal named "ensemble", so without carrying the
        # contributors through, every paper trade is attributed to "ensemble"
        # and a per-strategy performance table says nothing at all.
        signal_meta = dict(getattr(signal, "metadata", None) or {})
        contributing = [str(name) for name in (signal_meta.get("contributing") or [])]
        pending = {
            "fill_key": order_key,
            "symbol": order.symbol,
            "side": order.side.name,
            "quantity": float(order.quantity),
            "created_at": iso(order.created_at),
            "queued_on_bar": timestamp.isoformat(),
            "strategy": order.strategy,
            "reference_price": float(order.reference_price or 0.0),
            "stop_loss": float(signal.stop_loss),
            "take_profit": None if signal.take_profit is None else float(signal.take_profit),
            "risk_amount": None if risk is None else float(risk.risk_amount),
            "regime": str(self._state.get("last_regime") or "UNKNOWN"),
            "contributing": contributing,
            "aggregation_method": (
                None if signal_meta.get("method") is None else str(signal_meta["method"])
            ),
            "metadata": jsonable(order.metadata),
        }
        self._state["pending_orders"] = [pending]

        record = serialise_order(order)
        record["order_key"] = order_key
        record["queued_on_bar"] = timestamp.isoformat()
        record["stop_loss"] = pending["stop_loss"]
        record["take_profit"] = pending["take_profit"]
        record["risk_amount"] = pending["risk_amount"]
        self._append_bounded("orders", record, MAX_RECENT_ORDERS)

        self._record_event(
            "paper_order_queued",
            f"Queued a paper {order.side} order for {order.quantity:.6g} "
            f"{order.symbol} to fill at the next bar's open, stop "
            f"{pending['stop_loss']:.6g}. No broker is contacted.",
            symbol=order.symbol,
            side=str(order.side),
            quantity=float(order.quantity),
            stop_loss=pending["stop_loss"],
            order_key=order_key,
        )

    # ------------------------------------------------------------ bookkeeping

    def _already_processed(self, timestamp: pd.Timestamp) -> bool:
        """Whether this bar has been handled.

        Uses a high-water mark rather than a growing membership list. A bar at or
        before the newest processed timestamp has been seen; the recent-bar list
        is kept only for inspection, and is bounded so a long run's state file
        does not grow without limit.
        """
        watermark = parse_timestamp(self._state.get("last_processed_bar"))
        if watermark is None:
            return timestamp.isoformat() in (self._state.get("processed_bars") or [])
        candidate = timestamp if timestamp.tzinfo else timestamp.tz_localize("UTC")
        return candidate <= watermark

    def _mark_processed(self, timestamp: pd.Timestamp) -> None:
        watermark = parse_timestamp(self._state.get("last_processed_bar"))
        candidate = timestamp if timestamp.tzinfo else timestamp.tz_localize("UTC")
        if watermark is None or candidate > watermark:
            self._state["last_processed_bar"] = timestamp.isoformat()
        recent = list(self._state.get("processed_bars") or [])
        key = timestamp.isoformat()
        if key not in recent:
            recent.append(key)
        self._state["processed_bars"] = recent[-MAX_RECENT_BARS:]

    def _append_bounded(self, key: str, item: dict[str, Any], limit: int) -> None:
        rows = list(self._state.get(key) or [])
        rows.append(item)
        self._state[key] = rows[-limit:]

    def _trim_equity_curve(self) -> None:
        if len(self.portfolio.snapshots) > MAX_EQUITY_POINTS * 2:
            self.portfolio.snapshots = self.portfolio.snapshots[-MAX_EQUITY_POINTS:]

    def _sync_market_from_frame(
        self,
        market_data: MarketData,
        timestamp: pd.Timestamp,
        quality: str | None,
        quality_report: dict[str, Any] | None = None,
    ) -> None:
        """Record the market lane for a replayed bar.

        Marked ``historical_replay`` and stamped with the bar being replayed, not
        with the wall clock. Labelling replayed history as a live feed would make
        a deterministic backfill look like current market state.
        """
        existing = self._state.get("market") or {}
        if existing.get("source") == "live_refresh":
            # A live refresh already described this frame more completely; keep
            # its provenance and only advance the position within it.
            market = dict(existing)
            market["last_bar"] = timestamp.isoformat()
            market["last_close"] = self._last_close
            if quality_report is not None:
                market["quality"] = quality_report
            self._state["market"] = self._decorate_market(market)
            return

        df = market_data.df
        previous = existing.get("quality") if isinstance(existing, dict) else None
        self._state["market"] = self._decorate_market(
            {
                "symbol": market_data.symbol,
                "timeframe": market_data.timeframe.code,
                "status": "OK",
                "source": "historical_replay",
                "rows": int(len(df)),
                "first_bar": iso(df.index[0]),
                "last_bar": timestamp.isoformat(),
                "frame_last_bar": iso(df.index[-1]),
                "last_close": self._last_close,
                "quality_status": str(quality or "UNKNOWN"),
                # The gate's findings, not just its verdict. Carried forward from
                # the previous bar when this one supplied none, so a mid-replay
                # bar does not blank a report the run already established.
                "quality": quality_report if quality_report is not None else previous,
                "provider": market_data.provider,
            }
        )

    # ---------------------------------------------------------------- outputs

    def _mark_prices(self) -> dict[str, float]:
        """Prices to mark open positions at.

        The latest close, when there is one. Falling back to each position's own
        entry price - as this engine used to - makes unrealised P&L identically
        zero and drawdown blind to every open loss, which is precisely the
        drawdown a risk limit exists to catch.
        """
        if self._last_close is not None and self._last_close > 0:
            return {self.symbol: float(self._last_close)}
        market = self._state.get("market") or {}
        close = market.get("last_close")
        if close:
            return {self.symbol: float(close)}
        return {}

    def portfolio_snapshot(self, prices: dict[str, float] | None = None) -> dict[str, Any]:
        """Cash, equity, P&L, drawdown and exposure, marked at ``prices``."""
        prices = self._mark_prices() if prices is None else prices
        equity = self.portfolio.equity(prices)
        unrealised = self.portfolio.unrealised_pnl(prices)
        marked = bool(prices) and bool(self.portfolio.positions)
        return {
            "cash": float(self.portfolio.cash),
            "balance": float(self.portfolio.initial_balance),
            "equity": float(equity),
            "unrealised_pnl": float(unrealised),
            "realised_pnl": float(self.portfolio.realised_pnl),
            "total_pnl": float(equity - self.portfolio.initial_balance),
            "return_pct": float(
                (equity - self.portfolio.initial_balance) / self.portfolio.initial_balance
            ),
            "open_positions": len(self.portfolio.positions),
            "closed_trades": len(self.portfolio.closed_trades),
            "total_costs": float(self.portfolio.total_costs),
            "drawdown": float(self.portfolio.drawdown(equity)),
            "daily_loss": float(self.portfolio.daily_loss(equity)),
            "peak_equity": float(self.portfolio.risk_baselines()["peak_equity"]),
            "open_risk": float(self.portfolio.open_risk()),
            "exposure": float(self.portfolio.exposure(prices)),
            "currency": self.portfolio.currency,
            # Says whether the numbers above reflect a mark price at all. Without
            # it, an unmarked zero is indistinguishable from a genuine zero.
            "marked_to_market": marked or not self.portfolio.positions,
            "mark_price": prices.get(self.symbol),
        }

    def positions_view(self) -> list[dict[str, Any]]:
        """Open positions, marked at the latest close."""
        mark = self._mark_prices()
        return [self._position_record(p, mark) for p in self.portfolio.positions]

    def equity_curve(self) -> list[dict[str, Any]]:
        """The equity curve as plottable rows."""
        return [serialise_snapshot(s) for s in self.portfolio.snapshots]

    def snapshot(self) -> dict[str, Any]:
        """A compact status record, returned by :meth:`process_bar`."""
        chart = {
            "status": "ok",
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bar": self._state.get("last_bar"),
            "market": self.market_summary(),
            "portfolio": self.portfolio_snapshot(),
            "decision": self._state.get("latest_decision"),
            "regime": self._state.get("last_regime"),
            "positions": self.positions_view(),
            "pending_orders": list(self._state.get("pending_orders") or []),
            "events": list(self._state.get("recent_events") or [])[-10:],
            "errors": list(self._state.get("recent_errors") or [])[-10:],
            "fills": list(self._state.get("fills") or [])[-10:],
        }
        return chart

    def system_health(self) -> dict[str, Any]:
        """An honest health verdict.

        Health is derived from four independent questions - has anything run, did
        the last refresh succeed, is the data current, and did anything fail -
        because they fail independently. An earlier version reported ``HEALTHY``
        whenever the error list happened to be empty, which meant an engine that
        had never processed a bar and held no data at all described itself as
        healthy.

        The severity is the *worst* of the findings rather than whichever check
        ran last, and the "nothing has run yet" note is suppressed when a more
        specific failure explains why: telling an operator to run a catch-up is
        unhelpful when the catch-up is exactly what just failed.
        """
        market = self.market_summary()
        freshness = market.get("freshness") or {}
        quality = str(market.get("quality_status", "UNKNOWN"))
        errors = list(self._state.get("recent_errors") or [])
        bars = int(self._state.get("bars_processed") or 0)
        market_status = str(market.get("status", "NO_DATA"))

        # Ordered worst-first so `max` over this ranking cannot be defeated by
        # the order the checks happen to run in.
        rank = {"HEALTHY": 0, "IDLE": 1, "DEGRADED": 2, "FAILED": 3}
        findings: list[tuple[str, str]] = []

        if market_status == "ERROR":
            findings.append((
                "DEGRADED",
                f"The last market-data refresh failed: {market.get('error')}. Any figures "
                "shown come from the previous successful load, not from a current feed.",
            ))
        elif market_status == "EMPTY":
            findings.append((
                "DEGRADED",
                "The provider returned no bars on the last refresh. Nothing was processed "
                "and the previous data is unchanged.",
            ))

        if market_status == "NO_DATA" or bars == 0:
            findings.append((
                "IDLE",
                "No bar has been processed yet, so there is no state to report on. "
                "Run a catch-up or a live tick first.",
            ))

        if bars:
            if freshness.get("status") == "STALE":
                findings.append(("DEGRADED", f"Market data is stale. {freshness.get('detail')}"))
            elif freshness.get("status") == "UNKNOWN":
                findings.append((
                    "DEGRADED",
                    f"Market-data age could not be established. {freshness.get('detail')}",
                ))

        if quality == "FAIL":
            findings.append((
                "DEGRADED",
                "The data-quality gate graded this feed FAIL. Risk refuses to size against "
                "it, so no new position will be opened.",
            ))
        elif quality == "UNKNOWN" and bars:
            findings.append((
                "DEGRADED",
                "Data quality is unverified. The risk engine treats an unstated grade as a "
                "refusal rather than as a pass.",
            ))
        elif quality == "WARNING":
            findings.append((
                "DEGRADED" if self.config.risk.block_on_data_quality_warning else "HEALTHY",
                "The data-quality gate graded this feed WARNING. "
                + (
                    "block_on_data_quality_warning is set, so no new position is opened."
                    if self.config.risk.block_on_data_quality_warning
                    else "block_on_data_quality_warning is false in configs/risk.yaml, so "
                    "trading continues on caveated data - read the quality findings before "
                    "reading anything into a result."
                ),
            ))

        if errors:
            findings.append((
                "DEGRADED",
                f"{len(errors)} error(s) recorded; most recent: {errors[-1].get('error')}",
            ))

        status = max((f[0] for f in findings), key=lambda s: rank.get(s, 0), default="HEALTHY")

        # A specific failure explains the situation better than "nothing has run".
        specific = any(level != "IDLE" for level, _ in findings)
        reasons = [
            message
            for level, message in findings
            if not (level == "IDLE" and specific) and message
        ]
        if status == "HEALTHY" and not reasons:
            reasons.append(
                f"{bars:,} bars processed, data {freshness.get('status', 'UNKNOWN')}, "
                f"quality {quality}, no errors recorded."
            )

        return {
            "status": status,
            "reasons": reasons,
            "data_freshness": freshness.get("status", "UNKNOWN"),
            "freshness_detail": freshness.get("detail"),
            "quality_status": quality,
            "market_status": market_status,
            "bars_processed": bars,
            "bars_decided": int(self._state.get("bars_decided") or 0),
            "ambiguous_exits": int(self._state.get("ambiguous_exits") or 0),
            "error_count": len(errors),
            "trading_enabled": self.trading_enabled,
            "live_trading_enabled": False,
            "mode": str(self.config.platform.mode),
        }

    def performance_view(self) -> dict[str, Any]:
        """Realised performance by strategy and by regime.

        Built from the persisted closed-trade log and the selector's own
        performance table. Both are shown because they can legitimately differ:
        the trade log is bounded (see :data:`MAX_CLOSED_TRADES`) while the
        selector's samples are not trimmed, so on a long session the selector
        can be weighting on trades that have aged out of the log.
        """
        return build_performance(
            list(self._state.get("closed_trades") or []),
            selector_records=[
                {"strategy": strategy, "regime": regime, "r_multiples": list(samples)}
                for (strategy, regime), samples in sorted(self.performance.records.items())
            ],
            min_samples=int(self.performance.min_samples),
            open_positions=len(self.portfolio.positions),
        )

    def dashboard_data(self) -> dict[str, Any]:
        """Everything the dashboard renders, in one JSON-safe payload.

        Assembled from recorded state only. Where something has not happened,
        the payload says so rather than substituting a placeholder that reads
        like a result.
        """
        market = self.market_summary()
        portfolio = self.portfolio_snapshot()
        decision = self._state.get("latest_state") or {}
        if not isinstance(decision, dict):
            decision = {}

        graph = build_graph_visualization(
            decision or None,
            market=market,
            open_positions=len(self.portfolio.positions),
            trading_enabled=self.trading_enabled,
        )

        return {
            "generated_at": utcnow().isoformat(),
            "symbol": market.get("symbol", self.symbol),
            "timeframe": market.get("timeframe", self.timeframe),
            "instrument": {
                "name": market.get("instrument_name"),
                "is_proxy": bool(market.get("is_proxy")),
                "data_caveat": market.get("data_caveat"),
                "provider_symbol": market.get("provider_symbol"),
                "has_reliable_volume": market.get("has_reliable_volume"),
            },
            "market": market,
            "regime": decision.get("regime"),
            "strategies": decision.get("strategies") or [],
            "aggregation": decision.get("aggregation"),
            "selection": {
                "suppressed": decision.get("suppressed") or [],
                "rejected": decision.get("rejected") or [],
            },
            "risk": decision.get("risk"),
            "order": decision.get("order"),
            "decision": decision,
            "final_decision": self._state.get("latest_decision"),
            "portfolio": portfolio,
            "positions": self.positions_view(),
            "pending_orders": list(self._state.get("pending_orders") or []),
            "orders": list(self._state.get("orders") or [])[-20:],
            "fills": list(self._state.get("fills") or [])[-20:],
            "trades": list(self._state.get("closed_trades") or [])[-20:],
            "equity_curve": self.equity_curve(),
            "performance": self.performance_view(),
            "graph": graph,
            "system_health": self.system_health(),
            "events": list(self._state.get("recent_events") or [])[-25:],
            "errors": list(self._state.get("recent_errors") or [])[-25:],
            "safety": {
                "mode": str(self.config.platform.mode),
                "trading_enabled": self.trading_enabled,
                "live_trading_enabled": False,
                "note": (
                    "Paper trading only. Fills are produced by the execution simulator "
                    "from historical or delayed bars. No broker is contacted and no real "
                    "money is at risk."
                ),
            },
        }


__all__ = ["PaperTradingEngine", "build_graph_visualization"]
