"""The feature engine: MarketData in, FeatureSet out.

The engine is deliberately dumb about *meaning*. It computes a standard panel
of features and reports which ones are usable; deciding whether a feature
predicts anything is the research process's job, not the engine's.

Two guarantees hold for everything produced here:

1. **Causality.** A value at bar ``t`` uses only bars ``<= t``.
2. **Honest absence.** A feature that cannot be computed is NaN, never a
   filled-in guess. Warm-up rows stay NaN, and volume features are suppressed
   entirely where volume is meaningless for the instrument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.config.models import AssetConfig, FeatureConfig
from app.data.schema import CLOSE, MarketData, validate_schema
from app.features import indicators as ind
from app.features import structure as struct
from app.features import volume as vol
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureSet:
    """Computed features for one symbol/timeframe, with provenance."""

    symbol: str
    timeframe: str
    df: pd.DataFrame
    warmup_bars: int
    suppressed: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)

    @property
    def is_empty(self) -> bool:
        return self.df.empty

    def warm(self) -> pd.DataFrame:
        """Only the rows where every feature is defined.

        Use this for model training and statistics. Using the full frame means
        silently training on rows where half the features are NaN.
        """
        if self.df.empty:
            return self.df
        return self.df.iloc[self.warmup_bars :]

    def at(self, timestamp: pd.Timestamp) -> pd.Series:
        """Feature row for one bar.

        Raises:
            KeyError: if the timestamp is not in the index. Callers must not
                silently fall back to a neighbouring bar.
        """
        if timestamp not in self.df.index:
            raise KeyError(f"No feature row at {timestamp} for {self.symbol} {self.timeframe}")
        return self.df.loc[timestamp]

    def latest(self) -> pd.Series | None:
        """The most recent feature row, or None if empty."""
        return None if self.df.empty else self.df.iloc[-1]

    def is_warm_at(self, timestamp: pd.Timestamp) -> bool:
        """Whether ``timestamp`` is past the warm-up period."""
        if timestamp not in self.df.index:
            return False
        return self.df.index.get_loc(timestamp) >= self.warmup_bars

    def describe(self) -> str:
        suppressed = f", suppressed: {', '.join(self.suppressed)}" if self.suppressed else ""
        return (
            f"{self.symbol} {self.timeframe}: {len(self.df):,} rows, "
            f"{len(self.df.columns)} features, warmup {self.warmup_bars}{suppressed}"
        )


class FeatureEngine:
    """Computes the standard feature panel."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def compute(self, data: MarketData, asset: AssetConfig | None = None) -> FeatureSet:
        """Compute all features for ``data``.

        Args:
            data: validated market data from the phase-1 pipeline.
            asset: asset config. Used to decide whether volume features are
                meaningful for this instrument.

        Returns:
            A :class:`FeatureSet`. Never raises on short input - an empty or
            very short series simply yields mostly-NaN features and a warm-up
            count the caller can check.
        """
        validate_schema(data.df, strict=False)
        cfg = self.config
        df = data.df

        if df.empty:
            return FeatureSet(
                symbol=data.symbol,
                timeframe=data.timeframe.code,
                df=pd.DataFrame(index=df.index),
                warmup_bars=cfg.warmup_bars(),
                metadata={"empty": True},
            )

        close = df[CLOSE]
        parts: list[pd.DataFrame] = []
        suppressed: list[str] = []

        # --- volatility first: other features normalise against ATR ----------
        atr = ind.atr(df, cfg.atr_period)
        volatility = pd.DataFrame(
            {
                "atr": atr,
                "atr_pct": atr / close.replace(0.0, pd.NA).astype("float64"),
                "atr_percentile": ind.rolling_percentile(atr, cfg.volatility_lookback),
                "realised_vol": ind.rolling_std(close.pct_change(), cfg.bb_period),
            }
        )
        parts.append(volatility)
        parts.append(ind.bollinger_bands(close, cfg.bb_period, cfg.bb_std))

        # --- trend -----------------------------------------------------------
        trend = pd.DataFrame(index=df.index)
        for period in cfg.sma_periods:
            trend[f"sma_{period}"] = ind.sma(close, period)
            trend[f"close_vs_sma_{period}_atr"] = ind.distance_from(
                close, trend[f"sma_{period}"], atr
            )
        for period in cfg.ema_periods:
            trend[f"ema_{period}"] = ind.ema(close, period)

        # Slope is normalised by ATR so one threshold works across instruments.
        primary_ma = f"sma_{cfg.sma_periods[0]}" if cfg.sma_periods else None
        if primary_ma:
            trend["ma_slope"] = ind.slope(trend[primary_ma], cfg.slope_period, normalize_by=atr)
        parts.append(trend)
        parts.append(ind.adx(df, cfg.adx_period))

        # --- momentum --------------------------------------------------------
        momentum = pd.DataFrame(
            {
                "rsi": ind.rsi(close, cfg.rsi_period),
                "roc": ind.rate_of_change(close, cfg.roc_period),
                "momentum": ind.momentum(close, cfg.momentum_period),
            }
        )
        parts.append(momentum)
        parts.append(ind.macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal))

        # --- structure -------------------------------------------------------
        swings = struct.swing_points(
            df, struct.SwingConfig(left=cfg.swing_left, right=cfg.swing_right)
        )
        parts.append(swings)
        parts.append(struct.trend_structure(swings))
        parts.append(struct.support_resistance_distance(df, swings, atr))
        parts.append(
            struct.breakout_levels(
                df, period=cfg.donchian_period, atr_series=atr, buffer_atr=0.0
            )
        )

        # --- volume ----------------------------------------------------------
        volume_reliable = asset is None or asset.has_reliable_volume
        if volume_reliable:
            volume_frame = vol.volume_features(
                df, cfg.volume_period, cfg.volume_spike_threshold
            )
            if volume_frame["relative_volume"].isna().all():
                # Configured as reliable, but the series carries nothing usable.
                suppressed.append("volume (no data in series)")
            parts.append(volume_frame)
        else:
            suppressed.append("volume (instrument has no meaningful volume)")
            logger.debug(
                "Volume features suppressed",
                extra={"symbol": data.symbol, "reason": "has_reliable_volume=false"},
            )

        features = pd.concat(parts, axis=1)

        duplicated = features.columns[features.columns.duplicated()]
        if len(duplicated):
            raise ValueError(f"Feature name collision: {sorted(set(duplicated))}")

        features.index.name = df.index.name

        result = FeatureSet(
            symbol=data.symbol,
            timeframe=data.timeframe.code,
            df=features,
            warmup_bars=min(cfg.warmup_bars(), len(features)),
            suppressed=tuple(suppressed),
            metadata={
                "provider": data.provider,
                "volume_reliable": volume_reliable,
                "feature_count": len(features.columns),
                "config": cfg.model_dump(),
            },
        )
        logger.info("Computed features", extra={"summary": result.describe()})
        return result
