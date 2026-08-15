"""Breakout strategy.

Hypothesis: when price leaves a consolidation range, the move continues far
enough to pay for the false breaks.

Breakouts are notorious for false signals - most breaks fail. The two filters
here (a prior tightening range, and a buffer the close must clear) exist to
reduce that, and both are parameters to be tested rather than trusted.

The Donchian channel used for detection **excludes the current bar**, otherwise
a bar could never break a maximum that includes its own high. See
:func:`app.features.structure.donchian_channel`.
"""

from __future__ import annotations

import pandas as pd

from app.regimes.models import RegimeType
from app.signals.models import Signal, SignalDirection
from app.strategies.base import Strategy, StrategyContext, StrategyMetadata


class BreakoutStrategy(Strategy):
    """Donchian-channel breakout with consolidation and volume confirmation."""

    metadata = StrategyMetadata(
        name="breakout",
        description=(
            "Trades closes beyond a trailing Donchian channel, preferring breaks "
            "that follow a tightening range."
        ),
        supported_timeframes=("4H", "1D"),
        min_history_bars=200,
        indicators_used=("donchian_channel", "atr", "volume"),
        preferred_regimes=(RegimeType.BREAKOUT, RegimeType.LOW_VOLATILITY),
        avoided_regimes=(RegimeType.HIGH_VOLATILITY,),
        assumptions=(
            "Ranges resolve into directional moves often enough to pay for failures",
            "Most breakouts fail; the edge depends on the size of those that do not",
            "Volume confirmation is only meaningful where volume is real",
        ),
    )

    def _generate(self, ctx: StrategyContext) -> Signal:
        row = ctx.row
        buffer_atr = float(self.param("buffer_atr", 0.25))
        max_consolidation = float(self.param("max_consolidation_ratio", 0.9))
        require_volume = bool(self.param("require_volume_confirmation", False))
        volume_multiple = float(self.param("volume_multiple", 1.5))

        atr = row.get("atr")
        upper, lower = row.get("donchian_upper"), row.get("donchian_lower")

        if any(v is None or pd.isna(v) for v in (atr, upper, lower)):
            return self._wait(ctx, "Donchian channel or ATR unavailable at this bar")

        atr, upper, lower = float(atr), float(upper), float(lower)
        if atr <= 0:
            return self._wait(ctx, "ATR is zero; cannot size a breakout buffer")

        close = ctx.close
        buffer = atr * buffer_atr

        broke_up = close > upper + buffer
        broke_down = close < lower - buffer

        if not (broke_up or broke_down):
            return self._wait(
                ctx,
                (
                    f"Close {close:.5g} is inside the channel "
                    f"[{lower:.5g}, {upper:.5g}] plus a {buffer_atr:g} ATR buffer"
                ),
                donchian_upper=upper,
                donchian_lower=lower,
            )

        # Only a *fresh* break is a breakout. During a sustained trend every bar
        # sets a new channel high; entering on each of them is not a breakout
        # strategy, it is buying every bar of a trend at progressively worse
        # prices with the same stop distance.
        fresh_up = bool(row.get("breakout_up_fresh", broke_up))
        fresh_down = bool(row.get("breakout_down_fresh", broke_down))
        if (broke_up and not fresh_up) or (broke_down and not fresh_down):
            return self._wait(
                ctx,
                (
                    "Channel was already broken on recent bars - this is a trend in "
                    "progress, not a fresh breakout"
                ),
                donchian_upper=upper,
                donchian_lower=lower,
            )

        direction = SignalDirection.LONG if broke_up else SignalDirection.SHORT
        level = upper if broke_up else lower

        # --- consolidation filter -------------------------------------------
        consolidation = row.get("consolidation_ratio")
        consolidated = consolidation is not None and not pd.isna(consolidation)
        if consolidated and float(consolidation) > max_consolidation:
            return self._wait(
                ctx,
                (
                    f"Range was already wide before the break (consolidation ratio "
                    f"{float(consolidation):.2f} > {max_consolidation:g}); breaks from a wide "
                    "range are more often noise than signal"
                ),
                consolidation_ratio=float(consolidation),
            )

        reasoning = [
            f"Close {close:.5g} broke {'above' if broke_up else 'below'} the "
            f"{int(self.param('channel_period', 20))}-bar channel at {level:.5g}",
            f"Cleared the level by more than the {buffer_atr:g} ATR buffer",
        ]
        if consolidated:
            reasoning.append(
                f"Range had tightened beforehand (consolidation ratio {float(consolidation):.2f})"
            )

        # --- volume confirmation, only where volume means something ----------
        volume_confirmed = None
        relative_volume = row.get("relative_volume")
        volume_available = relative_volume is not None and not pd.isna(relative_volume)

        if require_volume:
            if not volume_available:
                # Suppressed volume features are not a reason to block the trade;
                # the instrument simply has no meaningful volume to confirm with.
                reasoning.append(
                    "Volume confirmation requested but unavailable for this instrument; "
                    "proceeding without it"
                )
            else:
                volume_confirmed = float(relative_volume) >= volume_multiple
                if not volume_confirmed:
                    return self._wait(
                        ctx,
                        (
                            f"Breakout lacks volume confirmation (relative volume "
                            f"{float(relative_volume):.2f} < {volume_multiple:g})"
                        ),
                        relative_volume=float(relative_volume),
                    )
                reasoning.append(
                    f"Volume confirms the break (relative volume {float(relative_volume):.2f})"
                )
        elif volume_available:
            volume_confirmed = float(relative_volume) >= volume_multiple

        stop_multiple = float(self.param("stop_atr_multiple", 2.0))
        target_multiple = float(self.param("target_atr_multiple", 4.0))

        # The stop sits back inside the channel: if price returns there, the
        # break has failed by definition.
        stop = (
            min(level - atr * 0.25, close - atr * stop_multiple)
            if direction is SignalDirection.LONG
            else max(level + atr * 0.25, close + atr * stop_multiple)
        )
        target = (
            close + atr * target_multiple
            if direction is SignalDirection.LONG
            else close - atr * target_multiple
        )

        excess = abs(close - level) / atr
        confidence = self._score(excess, consolidation, volume_confirmed, ctx)
        reasoning.append(f"Close is {excess:.2f} ATR beyond the level")

        return self._build_signal(
            ctx, direction, confidence, stop, target, reasoning,
            level=level, excess_atr=round(excess, 4),
            consolidation_ratio=None if not consolidated else float(consolidation),
            volume_confirmed=volume_confirmed, atr=atr,
        )

    def _score(
        self,
        excess: float,
        consolidation: object,
        volume_confirmed: bool | None,
        ctx: StrategyContext,
    ) -> float:
        # A decisive break clears the level by a meaningful margin, but an
        # enormous gap beyond it means chasing an extended move.
        excess_score = min(excess / 1.0, 1.0) if excess <= 2.0 else max(0.3, 1.0 - (excess - 2.0) / 3.0)

        if consolidation is None or pd.isna(consolidation):
            consolidation_score = 0.5
        else:
            # Tighter prior range scores higher.
            consolidation_score = max(0.0, min(1.0, 1.5 - float(consolidation)))

        # None means volume is unavailable for this instrument: neither credit
        # nor penalty, since its absence says nothing about the breakout.
        volume_score = {None: 0.5, True: 1.0, False: 0.3}[volume_confirmed]

        regime = ctx.regime.regime
        if regime in self.metadata.preferred_regimes:
            regime_score = 0.5 + 0.5 * ctx.regime.confidence
        elif regime in self.metadata.avoided_regimes:
            regime_score = 0.25
        else:
            regime_score = 0.5

        return (
            0.30 * excess_score
            + 0.25 * consolidation_score
            + 0.15 * volume_score
            + 0.30 * regime_score
        )
