"""Signal aggregation.

Combines the strategies' individual opinions into one decision. Three modes are
supported, all of which can return WAIT:

- ``majority``   - unweighted vote. Simple, and mostly a baseline to beat.
- ``weighted``   - votes scaled by the selector's regime-aware weights.
- ``unanimous``  - every non-suppressed strategy with an opinion must agree.

**The system must be comfortable doing nothing.** Every path here can conclude
WAIT, and each records why. A combiner that always produces a trade has merely
relabelled noise as consensus.

Conflicting signals are treated as information, not as a problem to be averaged
away: when strategies genuinely disagree, that disagreement is itself a reason
to stand aside, and ``conflict_policy`` decides how strictly.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from app.regimes.models import MarketRegime
from app.signals.models import Signal, SignalDirection
from app.signals.selector import StrategyWeight
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AggregationMethod(StrEnum):
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"


class ConflictPolicy(StrEnum):
    """What to do when strategies point in opposite directions."""

    #: Stand aside entirely. The most conservative reading of a split market.
    ABSTAIN = "abstain"
    #: Let the weighted score decide, provided the margin is decisive.
    NET_SCORE = "net_score"


@dataclass(frozen=True)
class AggregatedDecision:
    """The combined verdict, with the full audit trail behind it."""

    direction: SignalDirection
    confidence: float
    symbol: str
    timeframe: str
    timestamp: pd.Timestamp | None = None

    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None

    contributing: tuple[str, ...] = ()
    opposing: tuple[str, ...] = ()
    method: AggregationMethod = AggregationMethod.WEIGHTED
    scores: dict[str, float] = field(default_factory=dict)
    reasoning: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.direction.is_actionable

    def to_signal(self, strategy_name: str = "ensemble") -> Signal:
        """Render as a standard :class:`Signal` for the risk engine.

        The rest of the pipeline consumes signals, not decisions, so the
        ensemble presents itself in exactly the same shape as any single
        strategy - which is what lets the risk engine stay ignorant of how the
        decision was reached.
        """
        if not self.is_actionable:
            return Signal.wait(
                strategy=strategy_name,
                symbol=self.symbol,
                timeframe=self.timeframe,
                timestamp=self.timestamp,
                reasoning=self.reasoning,
                method=str(self.method),
            )
        return Signal(
            strategy=strategy_name,
            symbol=self.symbol,
            timeframe=self.timeframe,
            direction=self.direction,
            confidence=self.confidence,
            timestamp=self.timestamp,
            entry_price=self.entry_price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            reasoning=self.reasoning,
            metadata={
                "method": str(self.method),
                "contributing": list(self.contributing),
                "opposing": list(self.opposing),
                "scores": dict(self.scores),
                **self.metadata,
            },
        )

    def describe(self) -> str:
        lines = [f"FINAL DECISION: {self.direction}"]
        if self.is_actionable:
            lines.append(f"Confidence: {self.confidence:.2f}")
            lines.append(f"Entry {self.entry_price:.6g}  Stop {self.stop_loss:.6g}")
            if self.take_profit is not None:
                lines.append(f"Target {self.take_profit:.6g}")
        lines.append("Reasons:")
        lines.extend(f"  - {r}" for r in self.reasoning)
        return "\n".join(lines)


class SignalAggregator:
    """Combines strategy signals into one decision."""

    def __init__(
        self,
        method: AggregationMethod | str = AggregationMethod.WEIGHTED,
        min_confidence: float = 0.5,
        min_score_margin: float = 0.15,
        conflict_policy: ConflictPolicy | str = ConflictPolicy.ABSTAIN,
    ) -> None:
        self.method = AggregationMethod(method)
        # Below this, the ensemble stands aside rather than trading a weak edge.
        self.min_confidence = min_confidence
        # How decisively one side must beat the other, as a fraction of total weight.
        self.min_score_margin = min_score_margin
        self.conflict_policy = ConflictPolicy(conflict_policy)

    def aggregate(
        self,
        signals: list[Signal],
        weights: dict[str, StrategyWeight] | None = None,
        regime: MarketRegime | None = None,
        symbol: str = "",
        timeframe: str = "",
        timestamp: pd.Timestamp | None = None,
    ) -> AggregatedDecision:
        """Combine ``signals`` into one decision."""
        symbol = symbol or (signals[0].symbol if signals else "")
        timeframe = timeframe or (signals[0].timeframe if signals else "")
        if timestamp is None and signals:
            timestamp = signals[0].timestamp

        reasoning: list[str] = []
        if regime is not None:
            reasoning.append(
                f"Market regime {regime.regime} at {regime.confidence:.0%} confidence"
            )

        actionable = [s for s in signals if s.is_actionable]
        waiting = [s for s in signals if not s.is_actionable]

        for signal in waiting:
            first = signal.reasoning[0] if signal.reasoning else "no reason recorded"
            reasoning.append(f"{signal.strategy}: WAIT - {first}")

        if not actionable:
            reasoning.append("No strategy proposed a trade")
            return self._wait(symbol, timeframe, timestamp, reasoning)

        scores, contributions = self._score(actionable, weights)
        long_score, short_score = scores[SignalDirection.LONG], scores[SignalDirection.SHORT]
        total = long_score + short_score

        for signal in actionable:
            weight = self._weight_of(signal.strategy, weights)
            reasoning.append(
                f"{signal.strategy}: {signal.direction} at {signal.confidence:.2f} "
                f"confidence, weight {weight:.2f}"
            )

        if total <= 0:
            reasoning.append(
                "Every strategy proposing a trade was suppressed by the regime-aware "
                "selector, so no weighted opinion remains"
            )
            return self._wait(symbol, timeframe, timestamp, reasoning)

        # --- conflict ---------------------------------------------------------
        conflicted = long_score > 0 and short_score > 0
        if conflicted:
            names_long = [s.strategy for s in actionable if s.direction is SignalDirection.LONG]
            names_short = [s.strategy for s in actionable if s.direction is SignalDirection.SHORT]
            reasoning.append(
                f"Strategies disagree: {', '.join(names_long)} long vs "
                f"{', '.join(names_short)} short"
            )
            if self.conflict_policy is ConflictPolicy.ABSTAIN:
                reasoning.append(
                    "Conflict policy is ABSTAIN - a split market is a reason to stand "
                    "aside, not to average two opposing views into a third one"
                )
                return self._wait(symbol, timeframe, timestamp, reasoning, scores=scores)

        direction = (
            SignalDirection.LONG if long_score > short_score else SignalDirection.SHORT
        )
        winning, losing = max(long_score, short_score), min(long_score, short_score)
        margin = (winning - losing) / total

        if margin < self.min_score_margin:
            reasoning.append(
                f"Winning margin {margin:.0%} is below the {self.min_score_margin:.0%} "
                "minimum; the two sides are too evenly matched to act on"
            )
            return self._wait(symbol, timeframe, timestamp, reasoning, scores=scores)

        # --- method-specific gates --------------------------------------------
        supporting = [s for s in actionable if s.direction is direction]
        opposing = [s for s in actionable if s.direction is not direction]

        if self.method is AggregationMethod.UNANIMOUS and opposing:
            reasoning.append(
                f"Method is UNANIMOUS and {len(opposing)} strategy/strategies disagree"
            )
            return self._wait(symbol, timeframe, timestamp, reasoning, scores=scores)

        if self.method is AggregationMethod.MAJORITY and len(supporting) <= len(opposing):
            reasoning.append(
                f"Method is MAJORITY and support ({len(supporting)}) does not exceed "
                f"opposition ({len(opposing)})"
            )
            return self._wait(symbol, timeframe, timestamp, reasoning, scores=scores)

        # --- confidence ---------------------------------------------------------
        confidence = self._confidence(supporting, weights, margin)
        if confidence < self.min_confidence:
            reasoning.append(
                f"Combined confidence {confidence:.2f} is below the "
                f"{self.min_confidence:.2f} minimum"
            )
            return self._wait(symbol, timeframe, timestamp, reasoning, scores=scores)

        entry, stop, target = self._levels(supporting, direction, weights)
        reasoning.append(
            f"{direction} carries {margin:.0%} of the weighted vote across "
            f"{len(supporting)} strategy/strategies"
        )

        return AggregatedDecision(
            direction=direction,
            confidence=confidence,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            contributing=tuple(s.strategy for s in supporting),
            opposing=tuple(s.strategy for s in opposing),
            method=self.method,
            scores={str(k): v for k, v in scores.items()},
            reasoning=tuple(reasoning),
            metadata={"margin": margin, "contributions": contributions},
        )

    # -------------------------------------------------------------- internals

    def _score(
        self, signals: list[Signal], weights: dict[str, StrategyWeight] | None
    ) -> tuple[dict[SignalDirection, float], dict[str, float]]:
        scores: dict[SignalDirection, float] = defaultdict(float)
        contributions: dict[str, float] = {}

        for signal in signals:
            weight = self._weight_of(signal.strategy, weights)
            if weight <= 0:
                contributions[signal.strategy] = 0.0
                continue
            # Confidence scales the vote in weighted mode only; majority and
            # unanimous count strategies, not conviction.
            score = (
                weight * signal.confidence
                if self.method is AggregationMethod.WEIGHTED
                else weight
            )
            scores[signal.direction] += score
            contributions[signal.strategy] = score

        scores.setdefault(SignalDirection.LONG, 0.0)
        scores.setdefault(SignalDirection.SHORT, 0.0)
        return dict(scores), contributions

    @staticmethod
    def _weight_of(strategy: str, weights: dict[str, StrategyWeight] | None) -> float:
        if weights is None:
            return 1.0
        weight = weights.get(strategy)
        return 1.0 if weight is None else weight.weight

    def _confidence(
        self,
        supporting: list[Signal],
        weights: dict[str, StrategyWeight] | None,
        margin: float,
    ) -> float:
        """Weighted mean confidence of the supporting strategies, scaled by margin.

        Deliberately not the maximum: one very confident strategy should not
        speak for the ensemble.
        """
        total_weight = sum(self._weight_of(s.strategy, weights) for s in supporting)
        if total_weight <= 0:
            mean_confidence = sum(s.confidence for s in supporting) / len(supporting)
        else:
            mean_confidence = (
                sum(self._weight_of(s.strategy, weights) * s.confidence for s in supporting)
                / total_weight
            )
        # A narrow win reduces confidence even when the winners were sure.
        return float(min(1.0, mean_confidence * (0.5 + 0.5 * margin)))

    def _levels(
        self,
        supporting: list[Signal],
        direction: SignalDirection,
        weights: dict[str, StrategyWeight] | None,
    ) -> tuple[float, float, float | None]:
        """Combine entry, stop and target across the supporting strategies.

        Entry is the weighted mean (they all reference the same close, so this
        is effectively that close). The **stop is the most conservative** - the
        tightest of the proposals - because adopting the widest stop would
        silently increase the risk the strategies each believed they were
        taking.
        """
        total_weight = sum(self._weight_of(s.strategy, weights) for s in supporting) or 1.0
        entry = sum(
            self._weight_of(s.strategy, weights) * float(s.entry_price) for s in supporting
        ) / total_weight

        stops = [float(s.stop_loss) for s in supporting]
        stop = max(stops) if direction is SignalDirection.LONG else min(stops)

        targets = [float(s.take_profit) for s in supporting if s.take_profit is not None]
        if targets:
            target = min(targets) if direction is SignalDirection.LONG else max(targets)
        else:
            target = None

        # A combined stop must still sit on the correct side of the combined entry.
        if direction is SignalDirection.LONG and stop >= entry:
            stop = min(stops)
        elif direction is SignalDirection.SHORT and stop <= entry:
            stop = max(stops)

        return entry, stop, target

    @staticmethod
    def _wait(
        symbol: str,
        timeframe: str,
        timestamp: pd.Timestamp | None,
        reasoning: list[str],
        scores: dict[SignalDirection, float] | None = None,
    ) -> AggregatedDecision:
        return AggregatedDecision(
            direction=SignalDirection.WAIT,
            confidence=0.0,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            reasoning=tuple(reasoning),
            scores={str(k): v for k, v in (scores or {}).items()},
        )
