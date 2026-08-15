"""Regime types and the regime result object."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd


class RegimeType(StrEnum):
    """Market states the detector can report.

    ``UNCERTAIN`` is a first-class answer, not a failure. Most of the time the
    market is not doing anything cleanly classifiable, and a detector that
    always picks a confident label is lying at least half the time.
    """

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    UNCERTAIN = "UNCERTAIN"


class VolatilityState(StrEnum):
    """Volatility bucket, orthogonal to trend direction."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class MarketRegime:
    """The detector's verdict for one bar.

    Attributes:
        regime: primary classification.
        confidence: [0, 1]. How well the evidence fits the label - *not* a
            probability that the regime will persist, and emphatically not a
            forecast.
        volatility: volatility bucket.
        trend_strength: [0, 1], derived from ADX. Strength only, no direction.
        timestamp: the bar this describes.
        reasoning: human-readable evidence, drawn from actual computed values.
            Never decorative text written after the fact.
        metrics: the raw numbers behind the decision, for auditing.
    """

    regime: RegimeType
    confidence: float
    volatility: VolatilityState
    trend_strength: float
    timestamp: pd.Timestamp | None = None
    reasoning: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not 0.0 <= self.trend_strength <= 1.0:
            raise ValueError(f"trend_strength must be in [0, 1], got {self.trend_strength}")

    @property
    def is_trending(self) -> bool:
        return self.regime in {RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN}

    @property
    def is_uncertain(self) -> bool:
        return self.regime is RegimeType.UNCERTAIN

    @property
    def direction(self) -> int:
        """+1 up, -1 down, 0 for non-directional regimes."""
        if self.regime is RegimeType.TRENDING_UP:
            return 1
        if self.regime is RegimeType.TRENDING_DOWN:
            return -1
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": str(self.regime),
            "confidence": round(self.confidence, 4),
            "volatility": str(self.volatility),
            "trend_strength": round(self.trend_strength, 4),
            "timestamp": self.timestamp.isoformat() if self.timestamp is not None else None,
            "reasoning": list(self.reasoning),
            "metrics": {k: round(float(v), 6) for k, v in self.metrics.items()},
        }

    def describe(self) -> str:
        lines = [
            f"Regime:         {self.regime}",
            f"Confidence:     {self.confidence:.2f}",
            f"Volatility:     {self.volatility}",
            f"Trend strength: {self.trend_strength:.2f}",
        ]
        if self.reasoning:
            lines.append("Reasoning:")
            lines.extend(f"  - {r}" for r in self.reasoning)
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"{self.regime} (confidence {self.confidence:.2f}, vol {self.volatility})"
