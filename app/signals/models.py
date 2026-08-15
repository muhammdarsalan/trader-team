"""The standardized signal.

Every strategy returns this type. No strategy is permitted to invent its own
output shape - the moment two strategies disagree about what a signal looks
like, the aggregator, risk engine and backtester all need special cases, and
adding a sixth strategy means touching five other files.

The object validates itself on construction. A signal with a stop on the wrong
side of the entry is not a slightly-flawed signal; it is a bug that would size
a position off a negative risk distance and blow up the risk engine three
layers downstream, where the cause is unrecognisable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd


class SignalDirection(StrEnum):
    """What a strategy wants to do.

    ``WAIT`` is a real answer and the correct one most of the time. A strategy
    that always has an opinion is not a strategy, it is a random number
    generator with extra steps.
    """

    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"

    @property
    def is_actionable(self) -> bool:
        return self in {SignalDirection.LONG, SignalDirection.SHORT}

    @property
    def sign(self) -> int:
        """+1 long, -1 short, 0 wait."""
        return {SignalDirection.LONG: 1, SignalDirection.SHORT: -1}.get(self, 0)


@dataclass(frozen=True)
class Signal:
    """A strategy's opinion about one bar.

    Attributes:
        strategy: registered strategy name.
        symbol: canonical platform symbol.
        timeframe: canonical timeframe code.
        direction: LONG, SHORT or WAIT.
        confidence: [0, 1]. The strategy's own assessment of how well its
            conditions were met. **Not a probability of profit** - it is not
            calibrated against outcomes, and nothing in the platform treats it
            as if it were.
        timestamp: bar the signal refers to. The signal is generated from that
            bar's *close*, so execution happens at the next bar's open.
        entry_price: reference price (the signal bar's close). The backtester
            does not fill here; it fills at the next bar's open plus costs.
        stop_loss: protective stop. Required for actionable signals - without
            it the risk engine cannot size the position.
        take_profit: optional target.
        reasoning: evidence, drawn from actual computed values.
        metadata: raw indicator values behind the decision, for auditing.
    """

    strategy: str
    symbol: str
    timeframe: str
    direction: SignalDirection
    confidence: float = 0.0
    timestamp: pd.Timestamp | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    reasoning: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"{self.strategy}: confidence must be in [0, 1], got {self.confidence}"
            )

        if not self.direction.is_actionable:
            return

        # --- everything below applies only to actionable signals -------------
        if self.entry_price is None:
            raise ValueError(f"{self.strategy}: an actionable signal requires entry_price")
        if self.entry_price <= 0:
            raise ValueError(
                f"{self.strategy}: entry_price must be positive, got {self.entry_price}"
            )

        if self.stop_loss is None:
            raise ValueError(
                f"{self.strategy}: an actionable signal requires stop_loss. Without it the "
                "risk engine cannot compute a position size, and an unsized position is an "
                "uncapped loss."
            )
        if self.stop_loss <= 0:
            raise ValueError(f"{self.strategy}: stop_loss must be positive, got {self.stop_loss}")

        if self.direction is SignalDirection.LONG:
            if self.stop_loss >= self.entry_price:
                raise ValueError(
                    f"{self.strategy}: LONG stop_loss ({self.stop_loss}) must be below "
                    f"entry_price ({self.entry_price})"
                )
            if self.take_profit is not None and self.take_profit <= self.entry_price:
                raise ValueError(
                    f"{self.strategy}: LONG take_profit ({self.take_profit}) must be above "
                    f"entry_price ({self.entry_price})"
                )
        else:  # SHORT
            if self.stop_loss <= self.entry_price:
                raise ValueError(
                    f"{self.strategy}: SHORT stop_loss ({self.stop_loss}) must be above "
                    f"entry_price ({self.entry_price})"
                )
            if self.take_profit is not None and self.take_profit >= self.entry_price:
                raise ValueError(
                    f"{self.strategy}: SHORT take_profit ({self.take_profit}) must be below "
                    f"entry_price ({self.entry_price})"
                )

    # ------------------------------------------------------------- properties

    @property
    def is_actionable(self) -> bool:
        return self.direction.is_actionable

    @property
    def risk_per_unit(self) -> float | None:
        """Distance from entry to stop, in price units. The risk engine's input."""
        if not self.is_actionable or self.entry_price is None or self.stop_loss is None:
            return None
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_per_unit(self) -> float | None:
        """Distance from entry to target, in price units."""
        if not self.is_actionable or self.entry_price is None or self.take_profit is None:
            return None
        return abs(self.take_profit - self.entry_price)

    @property
    def reward_risk_ratio(self) -> float | None:
        """Target distance divided by stop distance.

        A high ratio is not a good signal on its own: it usually means a distant
        target that is rarely reached. Expectancy needs the hit rate too, which
        only the backtester can supply.
        """
        risk, reward = self.risk_per_unit, self.reward_per_unit
        if risk is None or reward is None or risk == 0:
            return None
        return reward / risk

    # ------------------------------------------------------------ constructors

    @classmethod
    def wait(
        cls,
        strategy: str,
        symbol: str,
        timeframe: str,
        timestamp: pd.Timestamp | None = None,
        reasoning: tuple[str, ...] | list[str] = (),
        **metadata: Any,
    ) -> Signal:
        """A no-trade decision, with the reason recorded.

        Recording *why* nothing happened is as valuable as recording trades: it
        is the only way to distinguish "the strategy saw nothing" from "the
        strategy was broken and produced nothing".
        """
        return cls(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            direction=SignalDirection.WAIT,
            confidence=0.0,
            timestamp=timestamp,
            reasoning=tuple(reasoning),
            metadata=metadata,
        )

    # -------------------------------------------------------------- rendering

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for storage and experiment records."""
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": str(self.direction),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat() if self.timestamp is not None else None,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_per_unit": self.risk_per_unit,
            "reward_risk_ratio": self.reward_risk_ratio,
            "reasoning": list(self.reasoning),
            "metadata": {k: _jsonable(v) for k, v in self.metadata.items()},
        }

    def describe(self) -> str:
        if not self.is_actionable:
            reasons = "; ".join(self.reasoning) or "no reason recorded"
            return f"{self.strategy} {self.symbol} {self.timeframe}: WAIT ({reasons})"

        rr = self.reward_risk_ratio
        parts = [
            f"{self.strategy} {self.symbol} {self.timeframe}: {self.direction}",
            f"confidence {self.confidence:.2f}",
            f"entry {self.entry_price:.5g}",
            f"stop {self.stop_loss:.5g}",
        ]
        if self.take_profit is not None:
            parts.append(f"target {self.take_profit:.5g}")
        if rr is not None:
            parts.append(f"R:R {rr:.2f}")
        return " | ".join(parts)

    def __str__(self) -> str:
        return self.describe()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)
