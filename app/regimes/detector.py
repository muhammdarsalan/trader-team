"""Deterministic market-regime detection.

Rule-based on purpose. Everything the detector concludes can be traced to a
number you can print, which means a wrong classification can be debugged rather
than shrugged at. An ML classifier can be added later behind the same
interface; starting with one would mean never being able to tell a modelling
bug from a market that simply changed.

The detector reads features at a single bar and never looks forward. It is also
free to answer UNCERTAIN, which it does often, because that is usually the
truthful answer.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.config.models import RegimeConfig
from app.features.engine import FeatureSet
from app.regimes.models import MarketRegime, RegimeType, VolatilityState
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _finite(value: object) -> float | None:
    """Coerce to float, returning None for NaN/inf/missing.

    Feature rows are full of legitimate NaN during warm-up. Treating those as
    zero would manufacture confident conclusions out of absent data.
    """
    if value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) or math.isinf(out) else out


class RegimeDetector:
    """Classifies a bar into a market regime from its feature row."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    # ------------------------------------------------------------------- API

    def detect(self, features: FeatureSet, timestamp: pd.Timestamp | None = None) -> MarketRegime:
        """Classify one bar.

        Args:
            features: computed feature set.
            timestamp: bar to classify. Defaults to the most recent bar.

        Returns:
            A :class:`MarketRegime`. Returns UNCERTAIN with zero confidence when
            the required features are unavailable, rather than guessing.
        """
        if features.is_empty:
            return self._unknown("No features available", timestamp)

        if timestamp is None:
            timestamp = features.df.index[-1]
        if timestamp not in features.df.index:
            return self._unknown(f"No feature row at {timestamp}", timestamp)

        if not features.is_warm_at(timestamp):
            return self._unknown(
                f"Bar is inside the {features.warmup_bars}-bar warm-up period; "
                "indicators are not yet fully formed",
                timestamp,
            )

        return self._classify(features.df.loc[timestamp], timestamp)

    def detect_series(self, features: FeatureSet) -> pd.DataFrame:
        """Classify every bar. Used for regime-conditioned performance research.

        Returns:
            DataFrame indexed like the features with columns ``regime``,
            ``confidence``, ``volatility``, ``trend_strength``.
        """
        if features.is_empty:
            return pd.DataFrame(
                columns=["regime", "confidence", "volatility", "trend_strength"]
            )

        records = []
        for timestamp in features.df.index:
            regime = self.detect(features, timestamp)
            records.append(
                {
                    "regime": str(regime.regime),
                    "confidence": regime.confidence,
                    "volatility": str(regime.volatility),
                    "trend_strength": regime.trend_strength,
                }
            )
        return pd.DataFrame(records, index=features.df.index)

    # -------------------------------------------------------------- internals

    def _classify(self, row: pd.Series, timestamp: pd.Timestamp) -> MarketRegime:
        cfg = self.config

        adx = _finite(row.get("adx"))
        di_plus = _finite(row.get("di_plus"))
        di_minus = _finite(row.get("di_minus"))
        ma_slope = _finite(row.get("ma_slope"))
        atr_pct_rank = _finite(row.get("atr_percentile"))
        bb_width = _finite(row.get("bb_width"))
        structure = row.get("structure")
        # Fresh breaks only. In a sustained trend every bar sets a new N-bar
        # high, so the plain breakout flag stays true indefinitely and would
        # permanently mask the trend it is part of.
        breakout_up = bool(row.get("breakout_up_fresh", False))
        breakout_down = bool(row.get("breakout_down_fresh", False))
        consolidation = _finite(row.get("consolidation_ratio"))

        if adx is None:
            return self._unknown("ADX unavailable at this bar", timestamp)

        metrics = {
            k: v
            for k, v in {
                "adx": adx,
                "di_plus": di_plus,
                "di_minus": di_minus,
                "ma_slope": ma_slope,
                "atr_percentile": atr_pct_rank,
                "bb_width": bb_width,
                "consolidation_ratio": consolidation,
            }.items()
            if v is not None
        }

        volatility = self._volatility_state(atr_pct_rank)
        trend_strength = self._trend_strength(adx)
        reasoning: list[str] = []

        # --- directional evidence -------------------------------------------
        # Three independent votes: ADX's own DI spread, the normalised MA slope,
        # and confirmed swing structure. Requiring agreement is what keeps the
        # detector from calling a trend off one noisy indicator.
        votes: list[int] = []

        if di_plus is not None and di_minus is not None:
            if di_plus > di_minus:
                votes.append(1)
            elif di_minus > di_plus:
                votes.append(-1)

        if ma_slope is not None:
            if ma_slope > cfg.slope_threshold:
                votes.append(1)
            elif ma_slope < -cfg.slope_threshold:
                votes.append(-1)
            else:
                votes.append(0)

        if structure == "UPTREND":
            votes.append(1)
        elif structure == "DOWNTREND":
            votes.append(-1)

        direction = 0
        agreement = 0.0
        if votes:
            net = sum(votes)
            direction = int(np.sign(net))
            # Fraction of votes pointing the same way as the net direction.
            agreement = abs(net) / len(votes)

        trending = adx >= cfg.adx_trending
        ranging = adx < cfg.adx_ranging

        # --- breakout takes precedence: it is a transition, not a state ------
        squeeze = consolidation is not None and consolidation < 1.0
        if (breakout_up or breakout_down) and direction != 0:
            breakout_aligned = (breakout_up and direction > 0) or (
                breakout_down and direction < 0
            )
            if breakout_aligned:
                reasoning.append(
                    f"Price broke the {'upper' if breakout_up else 'lower'} channel "
                    f"in the direction indicators already pointed"
                )
                if squeeze:
                    reasoning.append(
                        f"Range had tightened beforehand (consolidation ratio {consolidation:.2f})"
                    )
                confidence = self._blend(
                    [
                        (agreement, 0.4),
                        (min(adx / cfg.adx_trending, 1.5) / 1.5, 0.3),
                        (0.8 if squeeze else 0.4, 0.3),
                    ]
                )
                return self._finalise(
                    RegimeType.BREAKOUT, confidence, volatility, trend_strength,
                    timestamp, reasoning, metrics,
                )

        # --- trending --------------------------------------------------------
        if trending and direction != 0 and agreement >= 0.5:
            regime = RegimeType.TRENDING_UP if direction > 0 else RegimeType.TRENDING_DOWN
            reasoning.append(f"ADX {adx:.1f} is above the trending threshold {cfg.adx_trending:g}")
            if ma_slope is not None:
                reasoning.append(f"Moving-average slope {ma_slope:+.3f} ATR/bar")
            if structure in {"UPTREND", "DOWNTREND"}:
                reasoning.append(f"Confirmed swing structure is {structure}")
            reasoning.append(f"{agreement:.0%} of directional indicators agree")

            confidence = self._blend(
                [
                    (agreement, 0.45),
                    (min((adx - cfg.adx_ranging) / (2 * cfg.adx_trending), 1.0), 0.35),
                    (trend_strength, 0.20),
                ]
            )
            return self._finalise(
                regime, confidence, volatility, trend_strength, timestamp, reasoning, metrics
            )

        # --- ranging ---------------------------------------------------------
        if ranging:
            reasoning.append(f"ADX {adx:.1f} is below the ranging threshold {cfg.adx_ranging:g}")
            if squeeze:
                reasoning.append(f"Range is tightening (consolidation ratio {consolidation:.2f})")
            if direction == 0:
                reasoning.append("No directional agreement among indicators")

            # Volatility extremes are reported in preference to a bare RANGING
            # label, because they change how a strategy should size and stop.
            if volatility is VolatilityState.HIGH:
                reasoning.append(
                    f"Volatility is elevated (ATR at the {atr_pct_rank:.0%} percentile) "
                    "despite the absence of a trend - a choppy, expensive market to trade"
                )
                confidence = self._blend([(1.0 - adx / cfg.adx_ranging, 0.5), (0.8, 0.5)])
                return self._finalise(
                    RegimeType.HIGH_VOLATILITY, confidence, volatility, trend_strength,
                    timestamp, reasoning, metrics,
                )

            if volatility is VolatilityState.LOW:
                reasoning.append(
                    f"Volatility is compressed (ATR at the {atr_pct_rank:.0%} percentile)"
                )
                confidence = self._blend([(1.0 - adx / cfg.adx_ranging, 0.5), (0.8, 0.5)])
                return self._finalise(
                    RegimeType.LOW_VOLATILITY, confidence, volatility, trend_strength,
                    timestamp, reasoning, metrics,
                )

            confidence = self._blend(
                [(1.0 - adx / cfg.adx_ranging, 0.6), (1.0 - agreement, 0.4)]
            )
            return self._finalise(
                RegimeType.RANGING, confidence, volatility, trend_strength,
                timestamp, reasoning, metrics,
            )

        # --- the deliberate no-man's-land ------------------------------------
        reasoning.append(
            f"ADX {adx:.1f} sits between the ranging ({cfg.adx_ranging:g}) and trending "
            f"({cfg.adx_trending:g}) thresholds"
        )
        if direction == 0:
            reasoning.append("Directional indicators disagree")
        elif agreement < 0.5:
            reasoning.append(f"Only {agreement:.0%} of directional indicators agree")

        return self._finalise(
            RegimeType.UNCERTAIN, 0.3, volatility, trend_strength,
            timestamp, reasoning, metrics,
        )

    # ---------------------------------------------------------------- helpers

    def _volatility_state(self, atr_percentile: float | None) -> VolatilityState:
        if atr_percentile is None:
            return VolatilityState.UNKNOWN
        if atr_percentile >= self.config.high_volatility_percentile:
            return VolatilityState.HIGH
        if atr_percentile <= self.config.low_volatility_percentile:
            return VolatilityState.LOW
        return VolatilityState.MEDIUM

    @staticmethod
    def _trend_strength(adx: float) -> float:
        """ADX mapped to [0, 1].

        ADX above 50 is rare and extreme; anchoring the top of the scale there
        keeps the common 15-35 range spread across most of the interval instead
        of bunched at the bottom.
        """
        return float(np.clip(adx / 50.0, 0.0, 1.0))

    @staticmethod
    def _blend(weighted: list[tuple[float, float]]) -> float:
        """Weighted mean of evidence scores, clipped to [0, 1]."""
        total_weight = sum(w for _, w in weighted)
        if total_weight <= 0:
            return 0.0
        score = sum(float(np.clip(v, 0.0, 1.0)) * w for v, w in weighted) / total_weight
        return float(np.clip(score, 0.0, 1.0))

    def _finalise(
        self,
        regime: RegimeType,
        confidence: float,
        volatility: VolatilityState,
        trend_strength: float,
        timestamp: pd.Timestamp,
        reasoning: list[str],
        metrics: dict[str, float],
    ) -> MarketRegime:
        """Downgrade to UNCERTAIN when confidence is too low to act on."""
        if confidence < self.config.min_confidence and regime is not RegimeType.UNCERTAIN:
            reasoning.append(
                f"Confidence {confidence:.2f} is below the {self.config.min_confidence:.2f} "
                f"minimum, so the {regime} reading is not asserted"
            )
            regime = RegimeType.UNCERTAIN

        return MarketRegime(
            regime=regime,
            confidence=round(float(confidence), 4),
            volatility=volatility,
            trend_strength=round(float(trend_strength), 4),
            timestamp=timestamp,
            reasoning=tuple(reasoning),
            metrics=metrics,
        )

    @staticmethod
    def _unknown(reason: str, timestamp: pd.Timestamp | None) -> MarketRegime:
        return MarketRegime(
            regime=RegimeType.UNCERTAIN,
            confidence=0.0,
            volatility=VolatilityState.UNKNOWN,
            trend_strength=0.0,
            timestamp=timestamp,
            reasoning=(reason,),
        )
