"""YAML configuration loading with environment-variable overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import ValidationError

from app.config.models import (
    AssetUniverse,
    BacktestConfig,
    DataConfig,
    ExecutionConfig,
    FeatureConfig,
    PlatformConfig,
    RegimeConfig,
    RiskConfig,
    StrategiesConfig,
)
from app.utils.paths import CONFIG_DIR

T = TypeVar("T")


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing, malformed or invalid."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(
            f"Configuration file not found: {path}\n"
            f"Expected it under {CONFIG_DIR}. Copy the shipped defaults if you removed it."
        )
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level, got {type(data)}")
    return data


def _build(model: type[T], data: dict[str, Any], path: Path) -> T:
    try:
        return model(**data)  # type: ignore[call-arg]
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {path}:\n{exc}") from exc


def _config_path(filename: str, config_dir: Path | None = None) -> Path:
    base = config_dir or Path(os.environ.get("GTP_CONFIG_DIR", CONFIG_DIR))
    return base / filename


def load_assets(config_dir: Path | None = None) -> AssetUniverse:
    """Load ``configs/assets.yaml``."""
    path = _config_path("assets.yaml", config_dir)
    return _build(AssetUniverse, _read_yaml(path), path)


def load_data_config(config_dir: Path | None = None) -> DataConfig:
    """Load ``configs/data.yaml``."""
    path = _config_path("data.yaml", config_dir)
    return _build(DataConfig, _read_yaml(path), path)


def load_feature_config(config_dir: Path | None = None) -> FeatureConfig:
    """Load ``configs/features.yaml``."""
    path = _config_path("features.yaml", config_dir)
    return _build(FeatureConfig, _read_yaml(path), path)


def load_regime_config(config_dir: Path | None = None) -> RegimeConfig:
    """Load ``configs/regimes.yaml``."""
    path = _config_path("regimes.yaml", config_dir)
    return _build(RegimeConfig, _read_yaml(path), path)


def load_strategies_config(config_dir: Path | None = None) -> StrategiesConfig:
    """Load ``configs/strategies.yaml``."""
    path = _config_path("strategies.yaml", config_dir)
    return _build(StrategiesConfig, _read_yaml(path), path)


def load_risk_config(config_dir: Path | None = None) -> RiskConfig:
    """Load ``configs/risk.yaml``."""
    path = _config_path("risk.yaml", config_dir)
    return _build(RiskConfig, _read_yaml(path), path)


def load_execution_config(config_dir: Path | None = None) -> ExecutionConfig:
    """Load ``configs/execution.yaml``."""
    path = _config_path("execution.yaml", config_dir)
    return _build(ExecutionConfig, _read_yaml(path), path)


def load_backtest_config(config_dir: Path | None = None) -> BacktestConfig:
    """Load ``configs/backtest.yaml``."""
    path = _config_path("backtest.yaml", config_dir)
    return _build(BacktestConfig, _read_yaml(path), path)


# Environment variables that override platform.yaml, so that a run can be
# reconfigured without editing tracked files.
_ENV_OVERRIDES: dict[str, str] = {
    "TRADING_MODE": "mode",
    "TRADING_ENABLED": "trading_enabled",
    "GTP_LOG_LEVEL": "log_level",
    "GTP_RANDOM_SEED": "random_seed",
    "GTP_STARTING_BALANCE": "starting_balance",
    "GTP_BASE_CURRENCY": "base_currency",
}

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def _coerce_env(field: str, raw: str) -> Any:
    if field == "trading_enabled":
        low = raw.strip().lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ConfigError(f"TRADING_ENABLED must be a boolean-like value, got {raw!r}")
    if field == "random_seed":
        return int(raw)
    if field == "starting_balance":
        return float(raw)
    return raw


def load_platform_config(config_dir: Path | None = None) -> PlatformConfig:
    """Load ``configs/platform.yaml``, then apply environment overrides."""
    path = _config_path("platform.yaml", config_dir)
    data = _read_yaml(path)
    for env_key, field in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_key)
        if raw is not None and raw != "":
            data[field] = _coerce_env(field, raw)
    return _build(PlatformConfig, data, path)


@dataclass(frozen=True)
class AppConfig:
    """Every configuration section, loaded together."""

    platform: PlatformConfig
    data: DataConfig
    assets: AssetUniverse
    features: FeatureConfig
    regimes: RegimeConfig
    strategies: StrategiesConfig
    risk: RiskConfig
    execution: ExecutionConfig
    backtest: BacktestConfig


@lru_cache(maxsize=8)
def _cached_config(config_dir: str | None) -> AppConfig:
    directory = Path(config_dir) if config_dir else None
    return AppConfig(
        platform=load_platform_config(directory),
        data=load_data_config(directory),
        assets=load_assets(directory),
        features=load_feature_config(directory),
        regimes=load_regime_config(directory),
        strategies=load_strategies_config(directory),
        risk=load_risk_config(directory),
        execution=load_execution_config(directory),
        backtest=load_backtest_config(directory),
    )


def get_config(config_dir: Path | None = None) -> AppConfig:
    """Load (and memoise) the full configuration.

    Call :func:`reset_config_cache` after mutating config files or environment
    variables within a single process, which tests routinely do.
    """
    return _cached_config(str(config_dir) if config_dir else None)


def reset_config_cache() -> None:
    """Drop the memoised configuration."""
    _cached_config.cache_clear()
