"""Configuration loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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
    PlatformConfig,
    QualityThresholds,
    TradingMode,
)

# ----------------------------------------------------------------- shipped configs

def test_shipped_configs_load(config_dir):
    assets = load_assets(config_dir)
    data = load_data_config(config_dir)
    platform = load_platform_config(config_dir)

    assert "XAUUSD" in assets.assets
    assert data.default_provider == "yahoo"
    assert platform.base_currency == "USD"


def test_shipped_config_ships_with_trading_disabled(config_dir):
    """The kill switch must default to off. A repo that ships armed is a hazard."""
    assert load_platform_config(config_dir).trading_enabled is False


def test_get_config_bundles_all_sections(config_dir):
    cfg = get_config(config_dir)
    assert cfg.assets.assets and cfg.data.default_provider and cfg.platform.mode


def test_missing_config_file_raises_helpfully(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_assets(tmp_path)


def test_malformed_yaml_raises(tmp_path):
    (tmp_path / "assets.yaml").write_text("assets: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="Malformed YAML"):
        load_assets(tmp_path)


def test_invalid_values_raise_with_file_context(tmp_path):
    (tmp_path / "platform.yaml").write_text("starting_balance: -5", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid configuration"):
        load_platform_config(tmp_path)


# ------------------------------------------------------------------ asset universe

def _asset(**overrides):
    base = {
        "symbol": "TESTUSD", "name": "Test", "asset_class": "FX",
        "quote_currency": "USD", "tick_size": 0.0001, "typical_spread": 0.0002,
    }
    return AssetConfig(**{**base, **overrides})


def test_asset_symbol_is_uppercased():
    assert _asset(symbol=" eurusd ").symbol == "EURUSD"


def test_asset_provider_symbol_falls_back_to_canonical():
    asset = _asset(provider_symbols={"yahoo": "GC=F"})
    assert asset.provider_symbol("yahoo") == "GC=F"
    assert asset.provider_symbol("csv") == "TESTUSD"


def test_asset_rejects_unknown_timeframe():
    with pytest.raises(ValidationError):
        _asset(default_timeframe="3H")


def test_asset_rejects_unknown_field():
    """A typo'd key must fail loudly, not be silently ignored."""
    with pytest.raises(ValidationError):
        _asset(tik_size=0.01)


def test_universe_accepts_mapping_form():
    uni = AssetUniverse(
        assets={"EURUSD": {"name": "Euro", "asset_class": "FX", "quote_currency": "USD",
                           "tick_size": 0.00001, "typical_spread": 0.0001}}
    )
    assert uni.get("eurusd").name == "Euro"


def test_universe_unknown_symbol_lists_alternatives():
    uni = AssetUniverse(assets={"EURUSD": _asset(symbol="EURUSD").model_dump()})
    with pytest.raises(KeyError, match="EURUSD"):
        uni.get("NOPE")


def test_universe_enabled_symbols_excludes_disabled():
    uni = AssetUniverse(
        assets={
            "A": _asset(symbol="A").model_dump(),
            "B": {**_asset(symbol="B").model_dump(), "enabled": False},
        }
    )
    assert uni.enabled_symbols() == ["A"]


# --------------------------------------------------------------- platform config

def test_live_mode_is_rejected():
    """Real-money execution is out of scope and must be impossible to select."""
    with pytest.raises(ValidationError, match="LIVE mode is not implemented"):
        PlatformConfig(mode=TradingMode.LIVE)


@pytest.mark.parametrize("mode", ["BACKTEST", "PAPER", "RESEARCH", "ANALYSIS"])
def test_supported_modes_are_accepted(mode):
    assert PlatformConfig(mode=mode).mode.value == mode


def test_starting_balance_must_be_positive():
    with pytest.raises(ValidationError):
        PlatformConfig(starting_balance=0)


# ------------------------------------------------------------- env var overrides

def test_env_overrides_platform_config(config_dir, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "BACKTEST")
    monkeypatch.setenv("TRADING_ENABLED", "true")
    monkeypatch.setenv("GTP_RANDOM_SEED", "1234")
    reset_config_cache()

    cfg = load_platform_config(config_dir)
    assert cfg.mode is TradingMode.BACKTEST
    assert cfg.trading_enabled is True
    assert cfg.random_seed == 1234


def test_env_override_rejects_non_boolean(config_dir, monkeypatch):
    monkeypatch.setenv("TRADING_ENABLED", "maybe")
    with pytest.raises(ConfigError, match="boolean-like"):
        load_platform_config(config_dir)


def test_env_cannot_enable_live_mode(config_dir, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    with pytest.raises(ConfigError, match="LIVE mode is not implemented"):
        load_platform_config(config_dir)


# ------------------------------------------------------------ quality thresholds

def test_fail_threshold_must_exceed_warn():
    with pytest.raises(ValidationError, match="must be >="):
        QualityThresholds(max_missing_ratio_warn=0.5, max_missing_ratio_fail=0.1)


def test_quality_ratios_are_bounded():
    with pytest.raises(ValidationError):
        QualityThresholds(max_missing_ratio_warn=1.5)
