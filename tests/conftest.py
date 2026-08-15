"""Shared pytest fixtures.

Every fixture here is offline and deterministic. Tests that need the network
are marked ``@pytest.mark.network`` and are excluded from the default run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.schema import coerce_schema


@pytest.fixture(autouse=True)
def _isolate_filesystem(tmp_path, monkeypatch):
    """Point every path helper at a per-test temporary directory.

    Without this a test run would read, and worse write, the real dataset.
    """
    monkeypatch.setenv("GTP_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("GTP_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("GTP_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("GTP_LOGS_DIR", str(tmp_path / "logs"))

    from app.config.loader import reset_config_cache

    reset_config_cache()
    yield
    reset_config_cache()


def make_ohlcv(
    periods: int = 200,
    freq: str = "1D",
    start: str = "2020-01-01",
    start_price: float = 100.0,
    seed: int = 7,
    with_volume: bool = True,
    weekdays_only: bool = True,
) -> pd.DataFrame:
    """A clean, well-formed canonical OHLCV frame."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=periods * 2, freq=freq, tz="UTC")
    if weekdays_only:
        index = index[index.dayofweek < 5]
    index = index[:periods]

    n = len(index)
    steps = rng.normal(0, 0.01, n)
    close = start_price * np.exp(np.cumsum(steps))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0, 0.004, n)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = rng.integers(1_000, 10_000, n).astype(float) if with_volume else np.full(n, np.nan)

    return coerce_schema(
        pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
    )


@pytest.fixture
def clean_daily() -> pd.DataFrame:
    """200 clean daily bars on weekdays."""
    return make_ohlcv(periods=200, freq="1D")


@pytest.fixture
def clean_hourly() -> pd.DataFrame:
    """480 clean hourly bars on a continuous grid."""
    return make_ohlcv(periods=480, freq="1h", weekdays_only=False)


@pytest.fixture
def config_dir(tmp_path) -> pathlib.Path:  # noqa: F821
    """A minimal but valid config directory."""
    import pathlib
    import shutil

    from app.utils.paths import CONFIG_DIR

    target = tmp_path / "configs"
    target.mkdir()
    for path in CONFIG_DIR.glob("*.yaml"):
        shutil.copy(path, target / path.name)
    return pathlib.Path(target)
