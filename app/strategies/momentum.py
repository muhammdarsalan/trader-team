"""Momentum strategy.

Hypothesis: instruments that have moved strongly over a recent window continue
to move in the same direction over the next one.

Momentum and trend following are close relatives and their signals correlate
heavily. That correlation is itself a research question the platform must
answer in phase 4: five strategies that all lose in the same week are not a
diversified portfolio, they are one strategy wearing five hats.

The distinction here: trend following waits for a *structural* alignment of
moving averages, while momentum reacts to the *rate* of recent change. They can
disagree, most obviously at turning points.

Entry: rate of change beyond a threshold, RSI on the correct side of the
midline, and optional MACD agreement.
"""

from __future__ import annotations

import pandas as pd

from app.regimes.models import RegimeType
from app.signals.models import Signal, SignalDirection
from app.strategies.base import Strategy, StrategyContext, StrategyMetadata


class MomentumStrategy(Strategy):
    """Trades continuation of strong recent rate of change."""

    metadata = StrategyMetadata(
        name="momentum",
        description=(
            "Trades in the direction of strong recent price momentum, confirmed "
            "by RSI position and optionally MACD."
        ),
        supported_timeframes=("4H", "1D"),
        min_history_bars=200,
        indicators_used=("roc", "rsi", "macd", "atr"),
        preferred_regimes=(RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN, RegimeType.BREAKOUT),
        avoided_regimes=(RegimeType.RANGING,),
        assumptions=(
            "Recent relative strength persists over the holding period",
            "Signals correlate strongly with trend following - verify the pair adds "
            "diversification before running both",
            "Vulnerable to sharp reversals at momentum extremes",
        ),
    )

    def _generate(self, ctx: StrategyContext) -> Signal:
        row = ctx.row
        roc_threshold = float(self.param("roc_threshold", 0.02))
        rsi_bull_floor = float(self.param("rsi_bull_floor", 50.0))
        rsi_bear_ceiling = float(self.param("rsi_bear_ceiling", 50.0))
        require_macd = bool(self.param("require_macd_agreement", True))

        roc = row.get("roc")
        rsi = row.get("rsi")
        atr = row.get("atr")
        macd_hist = row.get("macd_hist")

        missing = [
            label
            for label, value in {"roc": roc, "rsi": rsi, "atr": atr}.items()
            if value is None or pd.isna(value)
        ]
        if missing:
            return self._wait(ctx, f"Required indicators unavailable: {', '.join(missing)}")

        roc, rsi, atr = float(roc), float(rsi), float(atr)
        if atr <= 0:
            return self._wait(ctx, "ATR is zero; cannot place a stop")

        bullish = roc >= roc_threshold and rsi >= rsi_bull_floor
        bearish = roc <= -roc_threshold and rsi <= rsi_bear_ceiling

        if not (bullish or bearish):
            return self._wait(
                ctx,
                (
                    f"Momentum is not decisive: rate of change {roc:+.2%} "
                    f"(threshold +/-{roc_threshold:.2%}), RSI {rsi:.1f}"
                ),
                roc=round(roc, 6), rsi=rsi,
            )

        direction = SignalDirection.LONG if bullish else SignalDirection.SHORT

        # --- MACD agreement ---------------------------------------------------
        macd_available = macd_hist is not None and not pd.isna(macd_hist)
        macd_agrees = None
        if macd_available:
            macd_hist = float(macd_hist)
            macd_agrees = (macd_hist > 0) if bullish else (macd_hist < 0)
            if require_macd and not macd_agrees:
                return self._wait(
                    ctx,
                    (
                        f"MACD histogram {macd_hist:+.5g} contradicts "
                        f"{'bullish' if bullish else 'bearish'} momentum"
                    ),
                    macd_hist=macd_hist,
                )
        elif require_macd:
            return self._wait(ctx, "MACD unavailable and agreement is required")

        reasoning = [
            f"Rate of change {roc:+.2%} over "
            f"{int(self.param('roc_period', 10))} bars exceeds the {roc_threshold:.2%} threshold",
            f"RSI {rsi:.1f} is {'above' if bullish else 'below'} the "
            f"{rsi_bull_floor if bullish else rsi_bear_ceiling:g} midline",
        ]
        if macd_agrees is not None:
            reasoning.append(
                f"MACD histogram {'confirms' if macd_agrees else 'does not confirm'} the direction"
            )

        stop_multiple = float(self.param("stop_atr_multiple", 2.0))
        target_multiple = float(self.param("target_atr_multiple", 3.0))
        stop, _ = self._atr_stop(ctx, direction, stop_multiple)
        target = (
            ctx.close + atr * target_multiple
            if direction is SignalDirection.LONG
            else ctx.close - atr * target_multiple
        )

        confidence = self._score(roc, roc_threshold, rsi, macd_agrees, ctx)

        if ctx.regime.regime is RegimeType.RANGING:
            reasoning.append(
                "Regime is RANGING, where momentum signals reverse frequently - "
                "confidence reduced"
            )

        return self._build_signal(
            ctx, direction, confidence, stop, target, reasoning,
            roc=round(roc, 6), rsi=rsi,
            macd_hist=None if not macd_available else float(macd_hist),
            atr=atr,
        )

    def _score(
        self,
        roc: float,
        threshold: float,
        rsi: float,
        macd_agrees: bool | None,
        ctx: StrategyContext,
    ) -> float:
        # Momentum well past the threshold scores higher, but an extreme reading
        # is where reversals happen, so the score is capped rather than unbounded.
        magnitude = abs(roc) / threshold if threshold > 0 else 0.0
        roc_score = min(magnitude / 3.0, 1.0)

        # RSI distance from the midline, capped at the conventional extremes.
        rsi_distance = abs(rsi - 50.0) / 30.0
        rsi_score = min(rsi_distance, 1.0)
        if rsi > 80.0 or rsi < 20.0:
            # Already extended: reversal risk rises.
            rsi_score *= 0.7

        # None means MACD was unavailable: neither confirmation nor conflict.
        macd_score = {None: 0.5, True: 1.0, False: 0.2}[macd_agrees]

        regime = ctx.regime.regime
        if regime in self.metadata.preferred_regimes:
            regime_score = 0.5 + 0.5 * ctx.regime.confidence
        elif regime in self.metadata.avoided_regimes:
            regime_score = 0.2
        else:
            regime_score = 0.5

        return 0.30 * roc_score + 0.20 * rsi_score + 0.15 * macd_score + 0.35 * regime_score
