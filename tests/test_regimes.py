"""Market-regime detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.models import FeatureConfig, RegimeConfig
from app.data.schema import MarketData, coerce_schema, empty_frame
from app.features.engine import FeatureEngine
from app.regimes.detector import RegimeDetector
from app.regimes.models import MarketRegime, RegimeType, VolatilityState
from tests.conftest import make_ohlcv


def wrap(df: pd.DataFrame) -> MarketData:
    return MarketData(symbol="XAUUSD", timeframe="1D", df=df, provider="synthetic")


def features_for(df: pd.DataFrame):
    return FeatureEngine(FeatureConfig()).compute(wrap(df))


def synthetic_trend(n: int = 400, slope: float = 1.0, noise: float = 0.2) -> pd.DataFrame:
    """A clean directional market: unambiguous by construction."""
    rng = np.random.default_rng(7)
    index = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    # Base high enough that a sustained downtrend never reaches zero; negative
    # prices would be rejected by the schema and are not a market anyway.
    close = 500.0 + slope * np.arange(n) + rng.normal(0, noise, n)
    return coerce_schema(
        pd.DataFrame(
            {
                "open": close - slope * 0.5,
                "high": close + abs(slope) * 0.6 + 0.1,
                "low": close - abs(slope) * 0.6 - 0.1,
                "close": close,
                "volume": np.full(n, 1000.0),
            },
            index=index,
        )
    )


def synthetic_range(n: int = 400, amplitude: float = 3.0) -> pd.DataFrame:
    """A clean oscillating market with no net direction."""
    index = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    close = 100.0 + amplitude * np.sin(np.arange(n) * 0.6)
    return coerce_schema(
        pd.DataFrame(
            {
                "open": close, "high": close + 0.6, "low": close - 0.6,
                "close": close, "volume": np.full(n, 1000.0),
            },
            index=index,
        )
    )


@pytest.fixture
def detector() -> RegimeDetector:
    return RegimeDetector(RegimeConfig())


# ------------------------------------------------------------ clear-cut markets

def test_strong_uptrend_is_detected(detector):
    regime = detector.detect(features_for(synthetic_trend(slope=1.0)))
    assert regime.regime is RegimeType.TRENDING_UP
    assert regime.direction == 1
    assert regime.is_trending


def test_strong_downtrend_is_detected(detector):
    regime = detector.detect(features_for(synthetic_trend(slope=-1.0)))
    assert regime.regime is RegimeType.TRENDING_DOWN
    assert regime.direction == -1


def test_oscillating_market_is_not_called_a_trend(detector):
    regime = detector.detect(features_for(synthetic_range()))
    assert not regime.is_trending


def test_sustained_trend_is_not_reported_as_a_permanent_breakout(detector):
    """In a steady trend every bar sets a new channel high.

    Without freshness filtering the detector would report BREAKOUT forever and
    never once identify the trend it is part of.
    """
    features = features_for(synthetic_trend(slope=1.0))
    labels = detector.detect_series(features)["regime"].iloc[features.warmup_bars :]

    assert (labels == "TRENDING_UP").mean() > 0.5
    assert (labels == "BREAKOUT").mean() < 0.2


def test_trend_strength_is_higher_in_a_trend_than_a_range(detector):
    trending = detector.detect(features_for(synthetic_trend(slope=1.0)))
    ranging = detector.detect(features_for(synthetic_range()))
    assert trending.trend_strength > ranging.trend_strength


# ---------------------------------------------------------------- uncertainty

def test_empty_features_yield_uncertain(detector):
    regime = detector.detect(features_for(empty_frame().assign()) if False else
                             FeatureEngine().compute(wrap(empty_frame())))
    assert regime.regime is RegimeType.UNCERTAIN
    assert regime.confidence == 0.0


def test_warmup_bars_yield_uncertain(detector):
    """Indicators are not fully formed; claiming a regime would be invention."""
    features = features_for(make_ohlcv(periods=400))
    early = features.df.index[5]
    regime = detector.detect(features, early)
    assert regime.regime is RegimeType.UNCERTAIN
    assert "warm-up" in " ".join(regime.reasoning)


def test_unknown_timestamp_yields_uncertain(detector):
    features = features_for(make_ohlcv(periods=400))
    regime = detector.detect(features, pd.Timestamp("1999-01-01", tz="UTC"))
    assert regime.regime is RegimeType.UNCERTAIN


def test_no_regime_is_asserted_below_the_confidence_floor():
    """The invariant: a label the evidence does not support is never asserted."""
    strict = RegimeDetector(RegimeConfig(min_confidence=0.99))
    features = features_for(make_ohlcv(periods=600, seed=61))
    result = strict.detect_series(features).iloc[features.warmup_bars :]

    asserted = result[result["regime"] != "UNCERTAIN"]
    assert (asserted["confidence"] >= 0.99).all()


def test_downgrade_records_why_the_label_was_withheld():
    strict = RegimeDetector(RegimeConfig(min_confidence=0.99))
    features = features_for(make_ohlcv(periods=600, seed=61))

    # Find a bar the permissive detector labels but the strict one does not.
    permissive = RegimeDetector(RegimeConfig(min_confidence=0.1))
    for timestamp in features.df.index[features.warmup_bars :]:
        if permissive.detect(features, timestamp).regime is not RegimeType.UNCERTAIN:
            strict_regime = strict.detect(features, timestamp)
            if strict_regime.regime is RegimeType.UNCERTAIN:
                assert "below" in " ".join(strict_regime.reasoning)
                return
    pytest.fail("No bar was labelled by the permissive detector; fixture is unsuitable")


# ------------------------------------------------------------------ volatility

def test_volatility_state_is_reported(detector):
    regime = detector.detect(features_for(make_ohlcv(periods=500)))
    assert regime.volatility in set(VolatilityState)


def test_volatility_thresholds_must_be_ordered():
    with pytest.raises(ValueError, match="must be below"):
        RegimeConfig(low_volatility_percentile=0.9, high_volatility_percentile=0.2)


def test_adx_thresholds_must_be_ordered():
    with pytest.raises(ValueError, match="must not exceed"):
        RegimeConfig(adx_trending=15.0, adx_ranging=30.0)


# ------------------------------------------------------------------- reasoning

def test_reasoning_comes_from_computed_values(detector):
    regime = detector.detect(features_for(synthetic_trend(slope=1.0)))
    assert regime.reasoning
    joined = " ".join(regime.reasoning)
    assert "ADX" in joined
    # The stated ADX must match the metric actually recorded.
    assert f"{regime.metrics['adx']:.1f}" in joined


def test_metrics_record_the_evidence(detector):
    regime = detector.detect(features_for(synthetic_trend(slope=1.0)))
    assert "adx" in regime.metrics
    assert "ma_slope" in regime.metrics


def test_describe_is_human_readable(detector):
    text = detector.detect(features_for(synthetic_trend(slope=1.0))).describe()
    assert "Regime:" in text and "Confidence:" in text


def test_to_dict_is_serialisable(detector):
    payload = detector.detect(features_for(synthetic_trend(slope=1.0))).to_dict()
    assert isinstance(payload["regime"], str)
    assert isinstance(payload["reasoning"], list)


# ---------------------------------------------------------------- series mode

def test_detect_series_classifies_every_bar(detector):
    features = features_for(make_ohlcv(periods=400))
    result = detector.detect_series(features)
    assert len(result) == len(features)
    assert set(result.columns) == {"regime", "confidence", "volatility", "trend_strength"}


def test_detect_series_marks_warmup_as_uncertain(detector):
    features = features_for(make_ohlcv(periods=400))
    result = detector.detect_series(features)
    warmup_labels = result["regime"].iloc[: features.warmup_bars].unique()
    assert set(warmup_labels) == {"UNCERTAIN"}


def test_detect_series_on_empty_features(detector):
    assert detector.detect_series(FeatureEngine().compute(wrap(empty_frame()))).empty


# ----------------------------------------------------------- model validation

def test_confidence_must_be_bounded():
    with pytest.raises(ValueError, match=r"confidence must be in \[0, 1\]"):
        MarketRegime(
            regime=RegimeType.RANGING, confidence=1.5,
            volatility=VolatilityState.LOW, trend_strength=0.5,
        )


def test_trend_strength_must_be_bounded():
    with pytest.raises(ValueError, match="trend_strength"):
        MarketRegime(
            regime=RegimeType.RANGING, confidence=0.5,
            volatility=VolatilityState.LOW, trend_strength=2.0,
        )


def test_non_directional_regimes_report_zero_direction():
    regime = MarketRegime(
        regime=RegimeType.RANGING, confidence=0.6,
        volatility=VolatilityState.MEDIUM, trend_strength=0.2,
    )
    assert regime.direction == 0


# ------------------------------------------------------------------- causality

def test_regime_at_a_bar_does_not_change_when_later_bars_arrive(detector):
    """The regime at bar t must be identical whether or not t+1..n exist yet."""
    df = make_ohlcv(periods=500, seed=53)
    full_features = features_for(df)

    cutoff = 400
    truncated_features = features_for(df.iloc[:cutoff])
    timestamp = df.index[cutoff - 1]

    from_full = detector.detect(full_features, timestamp)
    from_truncated = detector.detect(truncated_features, timestamp)

    assert from_full.regime is from_truncated.regime
    assert from_full.confidence == pytest.approx(from_truncated.confidence)
    assert from_full.volatility is from_truncated.volatility
