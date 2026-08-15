"""Support and resistance strategy.

Hypothesis: price reacts at levels where it previously reversed, so a rejection
at confirmed structure precedes a move away from that level.

The honest caveat: support and resistance is the most subjective idea in
technical analysis, and a level drawn after the fact always looks predictive.
This implementation uses only **confirmed swing pivots** from
:mod:`app.features.structure`, which are published ``swing_right`` bars after
they occur - precisely so the strategy cannot trade a level the market had not
yet established.

Entry: price approaches a confirmed level and shows a rejection wick.
"""

from __future__ import annotations

import pandas as pd

from app.regimes.models import RegimeType
from app.signals.models import Signal, SignalDirection
from app.strategies.base import Strategy, StrategyContext, StrategyMetadata


class SupportResistanceStrategy(Strategy):
    """Trades rejections at confirmed swing highs and lows."""

    metadata = StrategyMetadata(
        name="support_resistance",
        description=(
            "Trades rejection candles at confirmed swing structure: long at "
            "support, short at resistance."
        ),
        supported_timeframes=("4H", "1D"),
        min_history_bars=200,
        indicators_used=("swing_points", "atr"),
        preferred_regimes=(RegimeType.RANGING, RegimeType.LOW_VOLATILITY),
        avoided_regimes=(RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN, RegimeType.BREAKOUT),
        assumptions=(
            "Levels are only valid once confirmed by subsequent price action",
            "Rejection wicks indicate absorption at the level",
            "Fails badly when a level breaks during a strong trend",
        ),
    )

    def _generate(self, ctx: StrategyContext) -> Signal:
        row, bar = ctx.row, ctx.bar
        proximity = float(self.param("proximity_atr", 0.5))
        min_age = int(self.param("min_level_age", 5))
        max_age = int(self.param("max_level_age", 120))
        require_rejection = bool(self.param("require_rejection", True))
        wick_ratio_min = float(self.param("rejection_wick_ratio", 0.5))

        atr = row.get("atr")
        if atr is None or pd.isna(atr) or float(atr) <= 0:
            return self._wait(ctx, "ATR unavailable; cannot measure distance to structure")
        atr = float(atr)

        dist_resistance = row.get("dist_to_resistance_atr")
        dist_support = row.get("dist_to_support_atr")
        resistance_age = row.get("last_swing_high_age")
        support_age = row.get("last_swing_low_age")

        near_support = self._near(dist_support, proximity) and self._age_ok(
            support_age, min_age, max_age
        )
        near_resistance = self._near(dist_resistance, proximity) and self._age_ok(
            resistance_age, min_age, max_age
        )

        if not (near_support or near_resistance):
            return self._wait(
                ctx,
                f"Price is not within {proximity:g} ATR of usable structure "
                f"(support {self._fmt(dist_support)} ATR, resistance "
                f"{self._fmt(dist_resistance)} ATR away)",
            )

        # If price sits near both, it is inside a tight range with structure on
        # each side. Neither side is a trade.
        if near_support and near_resistance:
            return self._wait(
                ctx,
                "Price is close to both support and resistance - the range is too "
                "tight for either side to offer an edge",
            )

        direction = SignalDirection.LONG if near_support else SignalDirection.SHORT
        level = float(
            row.get("last_swing_low") if near_support else row.get("last_swing_high")
        )
        age = float(support_age if near_support else resistance_age)

        rejection_ratio = self._rejection_ratio(bar, direction)
        if require_rejection and rejection_ratio < wick_ratio_min:
            return self._wait(
                ctx,
                (
                    f"At {'support' if near_support else 'resistance'} {level:.5g} but the bar "
                    f"shows no rejection (wick {rejection_ratio:.0%} of range, "
                    f"{wick_ratio_min:.0%} required)"
                ),
                level=level,
            )

        reasoning = [
            f"Price {ctx.close:.5g} is at confirmed "
            f"{'support' if near_support else 'resistance'} {level:.5g}",
            f"Level was confirmed {age:.0f} bars ago",
            f"Rejection wick is {rejection_ratio:.0%} of the bar's range",
        ]

        # The stop goes beyond the level, not at the entry: if the level breaks,
        # the premise of the trade is gone.
        stop_multiple = float(self.param("stop_atr_multiple", 1.5))
        stop = (
            level - atr * stop_multiple
            if direction is SignalDirection.LONG
            else level + atr * stop_multiple
        )

        # A stop placed beyond a distant level can end up on the wrong side of
        # the entry; that is a malformed trade, not a tighter one.
        if direction is SignalDirection.LONG and stop >= ctx.close:
            return self._wait(ctx, "Stop beyond support would sit above the entry price")
        if direction is SignalDirection.SHORT and stop <= ctx.close:
            return self._wait(ctx, "Stop beyond resistance would sit below the entry price")

        target_multiple = float(self.param("target_atr_multiple", 3.0))
        target = (
            ctx.close + atr * target_multiple
            if direction is SignalDirection.LONG
            else ctx.close - atr * target_multiple
        )

        confidence = self._score(rejection_ratio, age, min_age, max_age, ctx)

        if ctx.regime.regime in self.metadata.avoided_regimes:
            reasoning.append(
                f"Regime is {ctx.regime.regime}; levels break more often in trends - "
                "confidence reduced"
            )

        return self._build_signal(
            ctx, direction, confidence, stop, target, reasoning,
            level=level, level_age=age, rejection_ratio=round(rejection_ratio, 4), atr=atr,
        )

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _near(distance: object, proximity: float) -> bool:
        """Within ``proximity`` ATR of the level, and not already through it."""
        if distance is None or pd.isna(distance):
            return False
        value = float(distance)
        return 0.0 <= value <= proximity

    @staticmethod
    def _age_ok(age: object, min_age: int, max_age: int) -> bool:
        """A level too fresh has not proven itself; too old and it stops mattering."""
        if age is None or pd.isna(age):
            return False
        return min_age <= float(age) <= max_age

    @staticmethod
    def _rejection_ratio(bar: pd.Series, direction: SignalDirection) -> float:
        """Fraction of the bar's range formed by the rejection wick.

        For a long at support, that is the lower wick: price probed below and
        was pushed back up.
        """
        high, low = float(bar["high"]), float(bar["low"])
        open_, close = float(bar["open"]), float(bar["close"])
        bar_range = high - low
        if bar_range <= 0:
            return 0.0

        if direction is SignalDirection.LONG:
            wick = min(open_, close) - low
        else:
            wick = high - max(open_, close)
        return max(0.0, wick / bar_range)

    @staticmethod
    def _fmt(value: object) -> str:
        return "n/a" if value is None or pd.isna(value) else f"{float(value):.2f}"

    def _score(
        self,
        rejection_ratio: float,
        age: float,
        min_age: int,
        max_age: int,
        ctx: StrategyContext,
    ) -> float:
        rejection_score = min(rejection_ratio / 0.7, 1.0)

        # Levels in the middle of the acceptable age band are the most useful:
        # old enough to be established, recent enough to still matter.
        span = max(max_age - min_age, 1)
        position = (age - min_age) / span
        age_score = 1.0 - abs(position - 0.35) / 0.65
        age_score = max(0.0, min(age_score, 1.0))

        regime = ctx.regime.regime
        if regime in self.metadata.preferred_regimes:
            regime_score = 0.5 + 0.5 * ctx.regime.confidence
        elif regime in self.metadata.avoided_regimes:
            regime_score = 0.15
        else:
            regime_score = 0.5

        return 0.40 * rejection_score + 0.20 * age_score + 0.40 * regime_score
