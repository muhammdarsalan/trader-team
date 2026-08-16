"""Degraded data must not reach a position size by a side entrance.

The quality engine refuses to *hand over* FAIL-grade data on the normal path,
but that path is not the only way into the backtester: a caller can disable
validation, build a ``MarketData`` straight from a frame, or simply forget to
pass the grade along. These tests pin the property that matters - a series that
was never verified, or that was verified and failed, cannot produce a trade
without someone having explicitly said so in configuration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import Backtester
from app.config.loader import get_config, override_config
from app.data.schema import MarketData, coerce_schema
from app.data.validators.quality import QualityStatus
from app.risk.engine import normalize_quality


@pytest.fixture
def config(config_dir):
    return get_config(config_dir)


def trending(n: int = 700, slope: float = 1.2, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-01", periods=n, freq="1D", tz="UTC")
    close = 500.0 + slope * np.arange(n) + rng.normal(0, 2.0, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return coerce_schema(
        pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) + 2.0,
                "low": np.minimum(open_, close) - 2.0,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=index,
        )
    )


def market(df: pd.DataFrame) -> MarketData:
    return MarketData(symbol="XAUUSD", timeframe="1D", df=df, provider="synthetic")


# ------------------------------------------------------------- normalisation

@pytest.mark.parametrize("value", [None, "UNKNOWN", "unknown", "nonsense", ""])
def test_non_grades_normalise_to_unverified(value):
    assert normalize_quality(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("PASS", QualityStatus.PASS),
        ("warning", QualityStatus.WARNING),
        ("FAIL", QualityStatus.FAIL),
        (QualityStatus.FAIL, QualityStatus.FAIL),
    ],
)
def test_real_grades_normalise(value, expected):
    assert normalize_quality(value) is expected


# ------------------------------------------------------------- the backtester

def test_backtester_grades_data_it_was_not_told_about(config):
    """Reaching the backtester directly must not skip the quality gate."""
    result = Backtester(config, config.assets.get("XAUUSD")).run(market(trending()))

    assert result.provenance.data_quality_source == "self-graded"
    assert result.provenance.data_quality in {"PASS", "WARNING", "FAIL"}


def test_a_supplied_grade_is_recorded_as_supplied(config):
    result = Backtester(config, config.assets.get("XAUUSD")).run(
        market(trending()), quality_status="PASS"
    )

    assert result.provenance.data_quality == "PASS"
    assert result.provenance.data_quality_source == "caller"


def test_failed_data_produces_no_trades(config):
    """A FAIL grade is not a footnote on the report; it stops the trading."""
    result = Backtester(config, config.assets.get("XAUUSD")).run(
        market(trending()), quality_status="FAIL"
    )

    assert result.metrics.total_trades == 0
    assert result.blocked_counts.get("BLOCKED_DATA_QUALITY", 0) > 0


def test_unverified_data_produces_no_trades(config):
    """The placeholder that used to be the default is now a refusal."""
    result = Backtester(config, config.assets.get("XAUUSD")).run(
        market(trending()), quality_status="UNKNOWN"
    )

    assert result.metrics.total_trades == 0
    assert result.blocked_counts.get("BLOCKED_DATA_QUALITY_UNKNOWN", 0) > 0


def test_blocking_on_warnings_is_configurable_and_effective(config):
    """Paper trading wants a stricter setting than historical research does."""
    strict = override_config(
        config, risk=config.risk.model_copy(update={"block_on_data_quality_warning": True})
    )
    result = Backtester(strict, strict.assets.get("XAUUSD")).run(
        market(trending()), quality_status="WARNING"
    )

    assert result.metrics.total_trades == 0
    assert result.blocked_counts.get("BLOCKED_DATA_QUALITY", 0) > 0


def test_clean_data_still_trades(config):
    """The gate must not be so eager that it blocks everything."""
    result = Backtester(config, config.assets.get("XAUUSD")).run(
        market(trending()), quality_status="PASS"
    )

    assert result.metrics.total_trades > 0


def test_the_refusal_is_recorded_with_a_reason(config):
    result = Backtester(config, config.assets.get("XAUUSD")).run(
        market(trending()), quality_status="FAIL"
    )

    reasons = set(result.decisions["block_reason"].dropna())
    assert "DATA_QUALITY" in reasons
    assert "quality" in result.render().lower()
