"""Regime-aware strategy selection.

Decides which strategies' opinions count at this bar, and how much.

The brief is explicit that naive vote-counting is not enough, and the reason is
worth stating: three strategies agreeing is not evidence of anything if all
three are trend followers looking at the same moving average. Weight has to
come from the market's current state and from how a strategy has *actually*
performed in that state - never from how many modules happen to exist.

**On using historical performance:** the selector accepts a performance table
keyed by (strategy, regime). It must be built from data strictly before the bar
being decided, or the selector becomes the most powerful look-ahead bug in the
platform - weighting strategies by results they had not yet produced. The
:class:`RegimePerformanceTracker` here enforces that by construction: it only
learns from trades that have already closed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from app.regimes.models import MarketRegime
from app.signals.models import Signal, SignalDirection
from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, not runtime behaviour
    # app.strategies.base imports app.signals, which imports this module. A
    # runtime import here therefore fails whenever app.strategies is imported
    # first - `import app.strategies.registry` in a fresh interpreter raised
    # ImportError while the identical code worked if app.signals happened to be
    # imported earlier. The annotation is all this module needs.
    from app.strategies.base import Strategy

logger = get_logger(__name__)


@dataclass(frozen=True)
class StrategyWeight:
    """One strategy's weight at a bar, and why it has it."""

    strategy: str
    weight: float
    base_weight: float
    regime_factor: float
    performance_factor: float
    reasoning: tuple[str, ...] = ()

    @property
    def suppressed(self) -> bool:
        return self.weight <= 0.0


@dataclass
class RegimePerformanceTracker:
    """Realised performance per (strategy, regime), learned only from closed trades.

    This is the evidence the selector is supposed to use instead of assumption.
    It is deliberately incremental: a trade contributes only once it has closed,
    so at any bar the table reflects strictly past information.
    """

    #: (strategy, regime) -> list of R multiples
    records: dict[tuple[str, str], list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    min_samples: int = 10

    def record_trade(self, strategy: str, regime: str, r_multiple: float | None) -> None:
        if r_multiple is None or pd.isna(r_multiple):
            return
        self.records[(strategy, regime)].append(float(r_multiple))

    def expectancy(self, strategy: str, regime: str) -> float | None:
        """Mean R for this pairing, or None when the sample is too small.

        Returning None below ``min_samples`` matters: a strategy that happened
        to win its first two trades in a regime is not evidence, and treating it
        as such is how a selector overfits to noise within a single backtest.
        """
        samples = self.records.get((strategy, regime), [])
        if len(samples) < self.min_samples:
            return None
        return float(sum(samples) / len(samples))

    def sample_count(self, strategy: str, regime: str) -> int:
        return len(self.records.get((strategy, regime), []))

    def to_frame(self) -> pd.DataFrame:
        """Regime-conditioned performance table, the phase-4 research artefact."""
        rows = [
            {
                "strategy": strategy,
                "regime": regime,
                "trades": len(samples),
                "mean_r": sum(samples) / len(samples) if samples else 0.0,
                "total_r": sum(samples),
                "win_rate": sum(1 for r in samples if r > 0) / len(samples) if samples else 0.0,
            }
            for (strategy, regime), samples in sorted(self.records.items())
        ]
        return pd.DataFrame(rows)


class StrategySelector:
    """Assigns weights to strategies based on regime and measured performance."""

    #: Weight multipliers applied from a strategy's own declared regime fit.
    PREFERRED_FACTOR = 1.0
    NEUTRAL_FACTOR = 0.6
    AVOIDED_FACTOR = 0.15

    def __init__(
        self,
        performance: RegimePerformanceTracker | None = None,
        suppress_avoided: bool = True,
        min_weight: float = 0.05,
    ) -> None:
        self.performance = performance or RegimePerformanceTracker()
        # A strategy that declares a regime hostile is silenced there rather
        # than merely down-weighted. Mean reversion in a strong trend is not a
        # weak opinion, it is a wrong one.
        self.suppress_avoided = suppress_avoided
        self.min_weight = min_weight

    def select(
        self,
        strategies: list[Strategy],
        regime: MarketRegime,
    ) -> dict[str, StrategyWeight]:
        """Weight each strategy for the current regime.

        Returns:
            Mapping of strategy name to :class:`StrategyWeight`. Weights are
            normalised across non-suppressed strategies, so they sum to 1.0
            (or to 0.0 when everything is suppressed - a legitimate outcome).
        """
        weights: dict[str, StrategyWeight] = {}

        for strategy in strategies:
            meta = strategy.metadata
            reasoning: list[str] = []

            # A configured standing weight, before any of the selector's own
            # judgement. Normally 1.0; a research variant sets it to test
            # whether tilting away from a redundant strategy helps.
            base = float(getattr(strategy.config, "weight", 1.0))
            if base != 1.0:
                reasoning.append(f"Configured standing weight {base:g}")

            regime_factor = self.NEUTRAL_FACTOR
            if regime.regime in meta.preferred_regimes:
                regime_factor = self.PREFERRED_FACTOR
                reasoning.append(f"{regime.regime} is a preferred regime for this strategy")
            elif regime.regime in meta.avoided_regimes:
                regime_factor = 0.0 if self.suppress_avoided else self.AVOIDED_FACTOR
                reasoning.append(
                    f"{regime.regime} is declared hostile to this strategy; "
                    f"{'suppressed' if self.suppress_avoided else 'heavily down-weighted'}"
                )
            else:
                reasoning.append(f"{regime.regime} is neutral for this strategy")

            # Regime confidence scales the whole judgement: an uncertain regime
            # reading should not drive a confident weighting decision.
            regime_factor *= 0.5 + 0.5 * regime.confidence

            performance_factor, performance_note = self._performance_factor(
                strategy.name, str(regime.regime)
            )
            reasoning.append(performance_note)

            raw = base * regime_factor * performance_factor
            weights[strategy.name] = StrategyWeight(
                strategy=strategy.name,
                weight=raw,
                base_weight=base,
                regime_factor=regime_factor,
                performance_factor=performance_factor,
                reasoning=tuple(reasoning),
            )

        return self._normalise(weights)

    def _performance_factor(self, strategy: str, regime: str) -> tuple[float, str]:
        """Scale weight by measured expectancy in this regime.

        Neutral (1.0) until there is enough evidence to justify anything else.
        """
        expectancy = self.performance.expectancy(strategy, regime)
        samples = self.performance.sample_count(strategy, regime)

        if expectancy is None:
            return 1.0, (
                f"No regime-conditioned performance yet ({samples} closed trades, "
                f"{self.performance.min_samples} needed); weighting on regime fit alone"
            )

        # Map mean R onto a bounded multiplier. A strategy losing money in this
        # regime is damped hard; one making money is boosted modestly, because
        # in-sample expectancy is a weak predictor of out-of-sample expectancy.
        if expectancy <= -0.5:
            factor = 0.2
        elif expectancy < 0:
            factor = 0.5
        elif expectancy < 0.2:
            factor = 1.0
        else:
            factor = min(1.5, 1.0 + expectancy)

        return factor, (
            f"Measured expectancy {expectancy:+.2f}R over {samples} closed trades "
            f"in {regime} -> weight factor {factor:.2f}"
        )

    def _normalise(self, weights: dict[str, StrategyWeight]) -> dict[str, StrategyWeight]:
        total = sum(w.weight for w in weights.values())
        if total <= 0:
            return {
                name: StrategyWeight(
                    strategy=w.strategy, weight=0.0, base_weight=w.base_weight,
                    regime_factor=w.regime_factor,
                    performance_factor=w.performance_factor,
                    reasoning=w.reasoning,
                )
                for name, w in weights.items()
            }

        out: dict[str, StrategyWeight] = {}
        for name, w in weights.items():
            normalised = w.weight / total
            if 0 < normalised < self.min_weight:
                normalised = 0.0
            out[name] = StrategyWeight(
                strategy=w.strategy,
                weight=normalised,
                base_weight=w.base_weight,
                regime_factor=w.regime_factor,
                performance_factor=w.performance_factor,
                reasoning=w.reasoning,
            )

        # Renormalise after dropping sub-threshold weights.
        total = sum(w.weight for w in out.values())
        if total > 0:
            out = {
                name: StrategyWeight(
                    strategy=w.strategy, weight=w.weight / total, base_weight=w.base_weight,
                    regime_factor=w.regime_factor,
                    performance_factor=w.performance_factor, reasoning=w.reasoning,
                )
                for name, w in out.items()
            }
        return out


def signals_conflict(signals: list[Signal]) -> bool:
    """Whether actionable signals disagree on direction."""
    directions = {s.direction for s in signals if s.is_actionable}
    return SignalDirection.LONG in directions and SignalDirection.SHORT in directions
