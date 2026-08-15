"""Mean-reversion strategy.

Hypothesis: after an extreme move away from a short-term average, price tends
to return toward it.

The critical risk here is obvious and fatal if ignored: **fading a strong trend
loses money indefinitely.** "Oversold" in a downtrend means the downtrend is
working. This strategy therefore refuses to trade when ADX indicates a trend,
which is the single most important line in the file.

Mean reversion characteristically has a high win rate and small winners, which
is the mirror image of trend following. Judging it by win rate flatters it just
as badly as win rate damns trend following - only expectancy settles it.

Entry: price outside a Bollinger band, RSI at an extreme, ADX low.
"""

from __future__ import annotations

import pandas as pd

from app.regimes.models import RegimeType
from app.signals.models import Signal, SignalDirection
from app.strategies.base import Strategy, StrategyContext, StrategyMetadata


class MeanReversionStrategy(Strategy):
    """Fades stretched moves in non-trending markets."""

    metadata = StrategyMetadata(
        name="mean_reversion",
        description=(
            "Buys oversold and sells overbought extremes in ranging markets, "
            "confirmed by Bollinger Bands and RSI, and blocked when ADX shows a trend."
        ),
        supported_timeframes=("1H", "4H", "1D"),
        min_history_bars=200,
        indicators_used=("bollinger_bands", "rsi", "adx", "atr"),
        preferred_regimes=(RegimeType.RANGING, RegimeType.LOW_VOLATILITY),
        avoided_regimes=(
            RegimeType.TRENDING_UP,
            RegimeType.TRENDING_DOWN,
            RegimeType.BREAKOUT,
            RegimeType.HIGH_VOLATILITY,
        ),
        assumptions=(
            "Price oscillates around a short-term mean when no trend is present",
            "High win rate with small winners; one trend can erase many wins",
            "Must never fade a strong trend - the ADX gate is not optional",
        ),
    )

    def _validate_params(self) -> None:
        oversold = float(self.param("rsi_oversold", 30.0))
        overbought = float(self.param("rsi_overbought", 70.0))
        if oversold >= overbought:
            raise ValueError(
                f"mean_reversion: rsi_oversold ({oversold}) must be below "
                f"rsi_overbought ({overbought})"
            )

    def _generate(self, ctx: StrategyContext) -> Signal:
        row = ctx.row
        oversold = float(self.param("rsi_oversold", 30.0))
        overbought = float(self.param("rsi_overbought", 70.0))
        max_adx = float(self.param("max_adx", 20.0))
        min_distance = float(self.param("min_distance_atr", 1.5))

        rsi = row.get("rsi")
        adx = row.get("adx")
        pct_b = row.get("bb_pct_b")
        middle = row.get("bb_middle")
        atr = row.get("atr")

        missing = [
            label
            for label, value in {
                "rsi": rsi, "adx": adx, "bb_pct_b": pct_b, "bb_middle": middle, "atr": atr
            }.items()
            if value is None or pd.isna(value)
        ]
        if missing:
            return self._wait(ctx, f"Required indicators unavailable: {', '.join(missing)}")

        rsi, adx, pct_b = float(rsi), float(adx), float(pct_b)
        middle, atr = float(middle), float(atr)
        close = ctx.close

        if atr <= 0:
            return self._wait(ctx, "ATR is zero; cannot place a stop")

        # --- the trend gate: the most important check in this strategy -------
        if adx > max_adx:
            return self._wait(
                ctx,
                (
                    f"ADX {adx:.1f} exceeds the {max_adx:g} ceiling - a trend is present and "
                    "fading it is how mean-reversion strategies lose indefinitely"
                ),
                adx=adx,
            )

        if ctx.regime.regime in {RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN}:
            return self._wait(
                ctx,
                f"Regime is {ctx.regime.regime}; this strategy does not fade trends",
                regime=str(ctx.regime.regime),
            )

        # --- stretch from the mean -------------------------------------------
        distance_atr = abs(close - middle) / atr
        if distance_atr < min_distance:
            return self._wait(
                ctx,
                (
                    f"Price is only {distance_atr:.2f} ATR from the mean "
                    f"({min_distance:g} required) - not stretched enough to fade"
                ),
                distance_atr=round(distance_atr, 4),
            )

        # pct_b < 0 means price is below the lower band entirely.
        below_band = pct_b <= 0.0
        above_band = pct_b >= 1.0

        long_setup = below_band and rsi <= oversold
        short_setup = above_band and rsi >= overbought

        if not (long_setup or short_setup):
            return self._wait(
                ctx,
                (
                    f"No extreme: RSI {rsi:.1f} (thresholds {oversold:g}/{overbought:g}), "
                    f"Bollinger %B {pct_b:.2f}"
                ),
                rsi=rsi, bb_pct_b=round(pct_b, 4),
            )

        direction = SignalDirection.LONG if long_setup else SignalDirection.SHORT

        reasoning = [
            f"Price is {'below the lower' if long_setup else 'above the upper'} "
            f"Bollinger band (%B {pct_b:.2f})",
            f"RSI {rsi:.1f} is {'oversold' if long_setup else 'overbought'} "
            f"(threshold {oversold if long_setup else overbought:g})",
            f"ADX {adx:.1f} confirms no trend is present (ceiling {max_adx:g})",
            f"Price is {distance_atr:.2f} ATR from its {int(self.param('bb_period', 20))}-period mean",
        ]

        stop_multiple = float(self.param("stop_atr_multiple", 1.5))
        stop, _ = self._atr_stop(ctx, direction, stop_multiple)

        # The target is the mean itself - that is the entire thesis of the trade,
        # not an arbitrary ATR multiple.
        target = middle
        if direction is SignalDirection.LONG and target <= close:
            return self._wait(ctx, "Mean is not above the entry; no reversion room")
        if direction is SignalDirection.SHORT and target >= close:
            return self._wait(ctx, "Mean is not below the entry; no reversion room")

        confidence = self._score(rsi, oversold, overbought, distance_atr, adx, max_adx, ctx)

        return self._build_signal(
            ctx, direction, confidence, stop, target, reasoning,
            rsi=rsi, adx=adx, bb_pct_b=round(pct_b, 4),
            distance_atr=round(distance_atr, 4), atr=atr,
        )

    def _score(
        self,
        rsi: float,
        oversold: float,
        overbought: float,
        distance_atr: float,
        adx: float,
        max_adx: float,
        ctx: StrategyContext,
    ) -> float:
        # How far past the RSI threshold the reading is.
        if rsi <= oversold:
            rsi_score = min((oversold - rsi) / oversold, 1.0)
        else:
            rsi_score = min((rsi - overbought) / (100.0 - overbought), 1.0)

        distance_score = min(distance_atr / 3.0, 1.0)
        # A lower ADX is better for this strategy.
        adx_score = max(0.0, 1.0 - adx / max_adx)

        regime = ctx.regime.regime
        if regime in self.metadata.preferred_regimes:
            regime_score = 0.5 + 0.5 * ctx.regime.confidence
        elif regime in self.metadata.avoided_regimes:
            regime_score = 0.1
        else:
            regime_score = 0.5

        return 0.30 * rsi_score + 0.25 * distance_score + 0.15 * adx_score + 0.30 * regime_score
