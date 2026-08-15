"""Typed configuration layer.

Nothing in the platform hard-codes symbols, thresholds or paths: it all comes
from ``configs/*.yaml``, validated into pydantic models at load time so that a
typo fails loudly at startup instead of silently mid-backtest.
"""

from app.config.loader import (
    ConfigError,
    get_config,
    load_assets,
    load_data_config,
    load_platform_config,
    reset_config_cache,
)
from app.config.models import (
    AssetConfig,
    AssetUniverse,
    DataConfig,
    PlatformConfig,
    QualityThresholds,
    TradingMode,
)

__all__ = [
    "AssetConfig",
    "AssetUniverse",
    "ConfigError",
    "DataConfig",
    "PlatformConfig",
    "QualityThresholds",
    "TradingMode",
    "get_config",
    "load_assets",
    "load_data_config",
    "load_platform_config",
    "reset_config_cache",
]
