"""Trend-following strategy.

Hypothesis: when a market is trending, an established directional move is more
likely to continue than to reverse within the holding period.

This is the most widely-documented anomaly in systematic trading and also one
of the most crowded. It is a hypothesis this platform exists to *test*, not a
fact it assumes. Trend following characteristically has a low win rate and
survives on the size of its winners, so judging it by win rate will produce
exactly the wrong conclusion.

Entry: fast EMA above slow EMA, price on the correct side of the long-term
filter, and ADX confirming a trend is actually present.
Exit: ATR-based stop and target (the backtester applies them).
"""

from __future__ import annotations

import pandas as pd

from app.regimes.models import RegimeType
from app.signals.models import Signal, SignalDirection
from app.strategies.base import Strategy, StrategyContext, StrategyMetadata


class TrendFollowingStrategy(Strategy):
    """EMA crossover with a long-term filter and an ADX trend-strength gate."""

    metadata = StrategyMetadata(
        name="trend_following",
        description=(
            "Trades in the direction of an established trend, confirmed by EMA "
            "alignment, a long-term moving-average filter and ADX strength."
        ),
        supported_timeframes=("4H", "1D"),
        min_history_bars=250,
        indicators_used=("ema", "sma", "adx", "atr"),
        preferred_regimes=(RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN, RegimeType.BREAKOUT),
        avoided_regimes=(RegimeType.RANGING, RegimeType.HIGH_VOLATILITY),
        assumptions=(
            "Trends persist longer than the holding period",
            "Low win rate is expected; profitability depends on winner size",
            "Performs poorly in ranging markets, where it is whipsawed",
        ),
    )

    def _validate_params(self) -> None:
        fast = int(self.param("fast_ema", 21))
        slow = int(self.param("slow_ema", 50))
        if fast >= slow:
            raise ValueError(
                f"trend_following: fast_ema ({fast}) must be shorter than slow_ema ({slow})"
            )

    def _generate(self, ctx: StrategyContext) -> Signal:
        row = ctx.row
        fast_period = int(self.param("fast_ema", 21))
        slow_period = int(self.param("slow_ema", 50))
        filter_period = int(self.param("trend_filter_sma", 200))
        adx_minimum = float(self.param("adx_minimum", 22.0))

        fast = row.get(f"ema_{fast_period}")
        slow = row.get(f"ema_{slow_period}")
        long_filter = row.get(f"sma_{filter_period}")
        adx = row.get("adx")

        missing = [
            label
            for label, value in {
                f"ema_{fast_period}": fast,
                f"ema_{slow_period}": slow,
                f"sma_{filter_period}": long_filter,
                "adx": adx,
            }.items()
            if value is None or pd.isna(value)
        ]
        if missing:
            return self._wait(
                ctx,
                f"Required indicators unavailable: {', '.join(missing)}. Check that "
                "configs/features.yaml computes the periods this strategy asks for.",
            )

        fast, slow = float(fast), float(slow)
        long_filter, adx = float(long_filter), float(adx)
        close = ctx.close

        if adx < adx_minimum:
            return self._wait(
                ctx,
                f"ADX {adx:.1f} is below the {adx_minimum:g} minimum - no trend strong "
                "enough to follow",
                adx=adx,
            )

        bullish = fast > slow and close > long_filter
        bearish = fast < slow and close < long_filter

        if not (bullish or bearish):
            return self._wait(
                ctx,
                (
                    f"EMAs and the {filter_period}-period filter disagree "
                    f"(fast {fast:.5g}, slow {slow:.5g}, filter {long_filter:.5g})"
                ),
                adx=adx,
            )

        direction = SignalDirection.LONG if bullish else SignalDirection.SHORT

        reasoning = [
            f"EMA({fast_period}) {fast:.5g} is {'above' if bullish else 'below'} "
            f"EMA({slow_period}) {slow:.5g}",
            f"Price {close:.5g} is {'above' if bullish else 'below'} the "
            f"{filter_period}-period trend filter {long_filter:.5g}",
            f"ADX {adx:.1f} confirms trend strength (minimum {adx_minimum:g})",
        ]

        # Separation between the EMAs, in ATR units, distinguishes a decisive
        # trend from two averages sitting on top of each other.
        stop, atr = self._atr_stop(ctx, direction, float(self.param("stop_atr_multiple", 2.0)))
        separation = abs(fast - slow) / atr if atr > 0 else 0.0

        confidence = self._score(adx, adx_minimum, separation, ctx)
        reasoning.append(f"EMA separation {separation:.2f} ATR")

        if ctx.regime.regime in self.metadata.avoided_regimes:
            reasoning.append(
                f"Regime is {ctx.regime.regime}, which this strategy performs poorly in - "
                "confidence reduced"
            )

        target_multiple = float(self.param("target_atr_multiple", 3.0))
        target = (
            ctx.close + atr * target_multiple
            if direction is SignalDirection.LONG
            else ctx.close - atr * target_multiple
        )

        return self._build_signal(
            ctx, direction, confidence, stop, target, reasoning,
            adx=adx, ema_separation_atr=round(separation, 4), atr=atr,
        )

    def _score(
        self, adx: float, adx_minimum: float, separation: float, ctx: StrategyContext
    ) -> float:
        """Blend trend strength, EMA separation and regime agreement into [0, 1].

        Confidence expresses how well *this strategy's own conditions* are met.
        It is not a probability of profit and is not calibrated against outcomes.
        """
        # ADX from the minimum up to 40 spans the usable range of trend strength.
        adx_score = min((adx - adx_minimum) / (40.0 - adx_minimum), 1.0) if adx > adx_minimum else 0.0
        separation_score = min(separation / 2.0, 1.0)

        regime = ctx.regime.regime
        if regime in self.metadata.preferred_regimes:
            regime_score = 0.5 + 0.5 * ctx.regime.confidence
        elif regime in self.metadata.avoided_regimes:
            regime_score = 0.2
        else:
            regime_score = 0.5

        return 0.40 * adx_score + 0.25 * separation_score + 0.35 * regime_score
