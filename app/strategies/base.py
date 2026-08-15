"""The strategy interface.

Every strategy is independent: it receives market data, features and the
detected regime, and returns a :class:`Signal`. It knows nothing about the
graph, the risk engine, the backtester or the other strategies.

That isolation is the point. Adding a sixth strategy must not require editing
the graph, the database, the dashboard, the backtester or the risk engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.config.models import StrategyConfig
from app.data.schema import MarketData
from app.features.engine import FeatureSet
from app.regimes.models import MarketRegime, RegimeType
from app.signals.models import Signal, SignalDirection
from app.utils.logging import get_logger
from app.utils.timeutils import normalize_timeframe

logger = get_logger(__name__)


class StrategyError(RuntimeError):
    """Raised for a misconfigured strategy. Distinct from a strategy declining to trade."""


@dataclass(frozen=True)
class StrategyMetadata:
    """What a strategy declares about itself.

    Required by section 5 of the project brief: a strategy must state its
    supported timeframes, minimum history and assumptions rather than silently
    running anywhere it is pointed.
    """

    name: str
    description: str
    supported_timeframes: tuple[str, ...]
    min_history_bars: int
    indicators_used: tuple[str, ...] = ()
    # Regimes where the strategy's own logic says it should be favoured. This is
    # a hypothesis to be tested in phase 4, not an established fact - the
    # selector will eventually use measured performance instead.
    preferred_regimes: tuple[RegimeType, ...] = ()
    avoided_regimes: tuple[RegimeType, ...] = ()
    assumptions: tuple[str, ...] = ()

    def supports_timeframe(self, timeframe: str) -> bool:
        return normalize_timeframe(timeframe).code in self.supported_timeframes


@dataclass
class StrategyContext:
    """Everything a strategy needs to decide about one bar.

    Bundled into one object so the signature does not grow every time a new
    input is added, and so a strategy cannot accidentally reach for data it was
    not given.
    """

    data: MarketData
    features: FeatureSet
    regime: MarketRegime
    timestamp: pd.Timestamp
    asset_tick_size: float = 0.01
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def row(self) -> pd.Series:
        """Feature row for this bar."""
        return self.features.at(self.timestamp)

    @property
    def bar(self) -> pd.Series:
        """OHLCV for this bar."""
        return self.data.df.loc[self.timestamp]

    @property
    def close(self) -> float:
        return float(self.bar["close"])


class Strategy(ABC):
    """Base class for all strategies."""

    #: Overridden by each subclass.
    metadata: StrategyMetadata

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()
        self.params = dict(self.config.params)
        self._validate_params()

    # ------------------------------------------------------------- interface

    @abstractmethod
    def _generate(self, ctx: StrategyContext) -> Signal:
        """Produce a signal for one bar. Implemented by each strategy.

        Implementations may assume the context is warm and the timeframe is
        supported - :meth:`generate_signal` checks both first.
        """

    def generate_signal(
        self,
        data: MarketData,
        features: FeatureSet,
        regime: MarketRegime,
        timestamp: pd.Timestamp | None = None,
    ) -> Signal:
        """Public entry point: preconditions, then the strategy's own logic.

        Returns a WAIT signal (never raises) when preconditions are not met, so
        that one strategy declining to trade cannot interrupt the others.
        """
        timestamp = timestamp or (features.df.index[-1] if not features.is_empty else None)

        precondition_failure = self._check_preconditions(data, features, timestamp)
        if precondition_failure is not None:
            return precondition_failure

        ctx = StrategyContext(
            data=data,
            features=features,
            regime=regime,
            timestamp=timestamp,
        )

        signal = self._generate(ctx)

        # A strategy's own confidence floor is applied here rather than inside
        # each strategy, so the rule cannot be forgotten in a new one.
        if signal.is_actionable and signal.confidence < self.config.min_confidence:
            return Signal.wait(
                strategy=self.name,
                symbol=data.symbol,
                timeframe=features.timeframe,
                timestamp=timestamp,
                reasoning=(
                    *signal.reasoning,
                    f"Confidence {signal.confidence:.2f} is below this strategy's "
                    f"{self.config.min_confidence:.2f} minimum",
                ),
                suppressed_direction=str(signal.direction),
            )
        return signal

    # ------------------------------------------------------------ preconditions

    def _check_preconditions(
        self,
        data: MarketData,
        features: FeatureSet,
        timestamp: pd.Timestamp | None,
    ) -> Signal | None:
        """Return a WAIT signal if the strategy must not run, else None."""
        name = self.name
        timeframe = features.timeframe

        if timestamp is None or features.is_empty:
            return Signal.wait(
                name, data.symbol, timeframe, timestamp, ("No feature data available",)
            )

        if not self.metadata.supports_timeframe(timeframe):
            return Signal.wait(
                name,
                data.symbol,
                timeframe,
                timestamp,
                (
                    f"{name} does not support {timeframe} "
                    f"(supported: {', '.join(self.metadata.supported_timeframes)})",
                ),
            )

        if len(data) < self.metadata.min_history_bars:
            return Signal.wait(
                name,
                data.symbol,
                timeframe,
                timestamp,
                (
                    f"Insufficient history: {len(data)} bars available, "
                    f"{self.metadata.min_history_bars} required",
                ),
            )

        if timestamp not in features.df.index:
            return Signal.wait(
                name, data.symbol, timeframe, timestamp, (f"No feature row at {timestamp}",)
            )

        if not features.is_warm_at(timestamp):
            return Signal.wait(
                name,
                data.symbol,
                timeframe,
                timestamp,
                (
                    f"Bar is within the {features.warmup_bars}-bar warm-up; indicators are "
                    "not fully formed and a signal here would not be reproducible live",
                ),
            )

        return None

    # ------------------------------------------------------------------ helpers

    @property
    def name(self) -> str:
        return self.metadata.name

    def param(self, key: str, default: Any = None) -> Any:
        """Read a configured parameter, falling back to a default."""
        return self.params.get(key, default)

    def _validate_params(self) -> None:  # noqa: B027
        """Optional hook: reject nonsensical parameter combinations at construction.

        Deliberately concrete and empty rather than abstract - most strategies
        have nothing to validate, and forcing every one to define an empty
        method would be noise.
        """

    def _wait(self, ctx: StrategyContext, *reasons: str, **metadata: Any) -> Signal:
        """Build a WAIT signal for this strategy."""
        return Signal.wait(
            strategy=self.name,
            symbol=ctx.data.symbol,
            timeframe=ctx.features.timeframe,
            timestamp=ctx.timestamp,
            reasoning=reasons,
            **metadata,
        )

    def _build_signal(
        self,
        ctx: StrategyContext,
        direction: SignalDirection,
        confidence: float,
        stop_loss: float,
        take_profit: float | None,
        reasoning: list[str],
        **metadata: Any,
    ) -> Signal:
        """Assemble an actionable signal from this bar's close.

        The entry price is the signal bar's **close**, which is the last price
        actually knowable when the decision is made. The backtester fills at the
        next bar's open plus costs; it never fills at this price.
        """
        return Signal(
            strategy=self.name,
            symbol=ctx.data.symbol,
            timeframe=ctx.features.timeframe,
            direction=direction,
            confidence=round(float(min(max(confidence, 0.0), 1.0)), 4),
            timestamp=ctx.timestamp,
            entry_price=ctx.close,
            stop_loss=float(stop_loss),
            take_profit=None if take_profit is None else float(take_profit),
            reasoning=tuple(reasoning),
            metadata={"regime": str(ctx.regime.regime), **metadata},
        )

    def _atr_stop(
        self, ctx: StrategyContext, direction: SignalDirection, multiple: float
    ) -> tuple[float, float]:
        """ATR-based stop and target distances.

        Returns:
            ``(stop_price, atr)``.

        Raises:
            StrategyError: if ATR is unavailable or non-positive. A stop derived
                from a missing ATR would be the entry price itself, which the
                Signal validator would reject anyway - failing here gives a
                clearer message.
        """
        atr = ctx.row.get("atr")
        if atr is None or pd.isna(atr) or float(atr) <= 0:
            raise StrategyError(f"{self.name}: ATR unavailable at {ctx.timestamp}")

        atr = float(atr)
        distance = atr * multiple
        stop = ctx.close - distance if direction is SignalDirection.LONG else ctx.close + distance
        return stop, atr

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} name={self.name!r} enabled={self.config.enabled}>"
