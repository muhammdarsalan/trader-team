"""Typed configuration layer.

Nothing in the platform hard-codes symbols, thresholds or paths: it all comes
from ``configs/*.yaml``, validated into pydantic models at load time so that a
typo fails loudly at startup instead of silently mid-backtest.
"""

from app.config.loader import (
    AppConfig,
    ConfigError,
    get_config,
    load_assets,
    load_data_config,
    load_feature_config,
    load_platform_config,
    load_regime_config,
    load_strategies_config,
    reset_config_cache,
)
from app.config.models import (
    AssetConfig,
    AssetUniverse,
    DataConfig,
    FeatureConfig,
    PlatformConfig,
    QualityThresholds,
    RegimeConfig,
    StrategiesConfig,
    StrategyConfig,
    TradingMode,
)

__all__ = [
    "AppConfig",
    "AssetConfig",
    "AssetUniverse",
    "ConfigError",
    "DataConfig",
    "FeatureConfig",
    "PlatformConfig",
    "QualityThresholds",
    "RegimeConfig",
    "StrategiesConfig",
    "StrategyConfig",
    "TradingMode",
    "get_config",
    "load_assets",
    "load_data_config",
    "load_feature_config",
    "load_platform_config",
    "load_regime_config",
    "load_strategies_config",
    "reset_config_cache",
]
