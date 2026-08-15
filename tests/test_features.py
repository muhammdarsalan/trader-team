"""The feature engine: composition, warm-up, volume suppression, causality."""

from __future__ import annotations

import pandas as pd
import pytest

from app.config.models import AssetConfig, FeatureConfig
from app.data.schema import MarketData, empty_frame
from app.features.engine import FeatureEngine, FeatureSet
from tests.conftest import make_ohlcv
from tests.helpers import assert_causal, assert_no_nan_after_warmup


def wrap(df: pd.DataFrame, symbol: str = "XAUUSD", timeframe: str = "1D") -> MarketData:
    return MarketData(symbol=symbol, timeframe=timeframe, df=df, provider="synthetic")


def _asset(**overrides) -> AssetConfig:
    base = {
        "symbol": "XAUUSD", "name": "Gold", "asset_class": "METAL",
        "quote_currency": "USD", "tick_size": 0.01, "typical_spread": 0.3,
    }
    return AssetConfig(**{**base, **overrides})


@pytest.fixture
def engine() -> FeatureEngine:
    return FeatureEngine(FeatureConfig())


@pytest.fixture
def features(engine) -> FeatureSet:
    return engine.compute(wrap(make_ohlcv(periods=600, seed=31)))


# ------------------------------------------------------------------ composition

def test_computes_the_expected_feature_families(features):
    expected = [
        "atr", "atr_percentile", "bb_upper", "bb_pct_b", "bb_width",
        "sma_20", "sma_50", "sma_200", "ema_9", "ema_21", "ma_slope",
        "adx", "di_plus", "di_minus",
        "rsi", "roc", "momentum", "macd", "macd_signal", "macd_hist",
        "last_swing_high", "last_swing_low", "structure",
        "donchian_upper", "donchian_lower", "breakout_up",
        "dist_to_resistance_atr", "dist_to_support_atr",
    ]
    missing = [c for c in expected if c not in features.columns]
    assert not missing, f"missing features: {missing}"


def test_index_matches_the_input(features):
    df = make_ohlcv(periods=600, seed=31)
    assert features.df.index.equals(df.index)


def test_no_duplicate_feature_names(features):
    assert len(features.columns) == len(set(features.columns))


def test_close_vs_sma_is_expressed_in_atr_units(features):
    assert "close_vs_sma_20_atr" in features.columns


# ---------------------------------------------------------------------- warm-up

def test_warmup_reflects_the_longest_indicator():
    config = FeatureConfig(sma_periods=[200], volatility_lookback=100)
    assert config.warmup_bars() >= 200


def test_features_are_defined_after_warmup(features):
    assert_no_nan_after_warmup(
        features.df.select_dtypes(include="number"),
        features.warmup_bars,
        label="feature engine",
    )


def test_warm_excludes_the_warmup_rows(features):
    assert len(features.warm()) == len(features.df) - features.warmup_bars


def test_is_warm_at_is_false_inside_warmup(features):
    early = features.df.index[5]
    late = features.df.index[-1]
    assert features.is_warm_at(early) is False
    assert features.is_warm_at(late) is True


def test_warmup_is_capped_at_series_length(engine):
    """A short series must not report a warm-up longer than it is."""
    result = engine.compute(wrap(make_ohlcv(periods=50)))
    assert result.warmup_bars <= len(result)


# ------------------------------------------------------------- volume handling

def test_volume_features_present_when_volume_is_reliable(engine):
    result = engine.compute(wrap(make_ohlcv(periods=400)), _asset(has_reliable_volume=True))
    assert "relative_volume" in result.columns
    assert not result.suppressed


def test_volume_features_suppressed_when_volume_is_meaningless(engine):
    """Spot FX has no real volume. Computing it anyway would poison any model."""
    result = engine.compute(wrap(make_ohlcv(periods=400)), _asset(has_reliable_volume=False))
    assert "relative_volume" not in result.columns
    assert any("volume" in s for s in result.suppressed)


def test_suppression_is_reported_not_silent(engine):
    result = engine.compute(wrap(make_ohlcv(periods=400)), _asset(has_reliable_volume=False))
    assert result.suppressed
    assert "volume" in result.describe()


def test_volume_features_are_nan_when_series_has_no_volume(engine):
    df = make_ohlcv(periods=400, with_volume=False)
    result = engine.compute(wrap(df), _asset(has_reliable_volume=True))
    assert result.df["relative_volume"].isna().all()
    assert any("no data" in s for s in result.suppressed)


# ------------------------------------------------------------------ edge cases

def test_empty_input_yields_empty_features(engine):
    result = engine.compute(wrap(empty_frame()))
    assert result.is_empty
    assert result.metadata["empty"] is True


def test_short_series_does_not_raise(engine):
    """Short input yields mostly-NaN features, not an exception."""
    result = engine.compute(wrap(make_ohlcv(periods=10)))
    assert len(result) == 10


def test_at_raises_for_an_unknown_timestamp(features):
    with pytest.raises(KeyError, match="No feature row"):
        features.at(pd.Timestamp("1999-01-01", tz="UTC"))


def test_latest_returns_the_final_row(features):
    latest = features.latest()
    assert latest is not None
    assert latest.name == features.df.index[-1]


def test_latest_of_empty_is_none(engine):
    assert engine.compute(wrap(empty_frame())).latest() is None


# ---------------------------------------------------------------- provenance

def test_metadata_records_the_config(features):
    assert features.metadata["feature_count"] == len(features.columns)
    assert "config" in features.metadata
    assert features.metadata["config"]["rsi_period"] == 14


# ------------------------------------------------------------------ causality

def test_whole_feature_engine_is_causal():
    """The end-to-end guarantee: no feature uses information from the future."""
    engine = FeatureEngine(FeatureConfig())
    df = make_ohlcv(periods=500, seed=41)

    def compute(frame: pd.DataFrame) -> pd.DataFrame:
        return engine.compute(wrap(frame)).df

    assert_causal(compute, df, label="FeatureEngine")


def test_feature_engine_is_causal_without_volume():
    engine = FeatureEngine(FeatureConfig())
    df = make_ohlcv(periods=500, seed=43, with_volume=False)

    def compute(frame: pd.DataFrame) -> pd.DataFrame:
        return engine.compute(wrap(frame), _asset(has_reliable_volume=False)).df

    assert_causal(compute, df, label="FeatureEngine (no volume)")


# ------------------------------------------------------------- config validation

def test_macd_periods_must_be_ordered():
    with pytest.raises(ValueError, match="must be shorter"):
        FeatureConfig(macd_fast=26, macd_slow=12)
