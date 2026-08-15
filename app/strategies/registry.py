"""Strategy registry.

Adding a strategy is: write the class, call :func:`register_strategy`, add a
block to ``configs/strategies.yaml``. The graph, backtester, risk engine and
dashboard all discover it from here without modification.
"""

from __future__ import annotations

from app.config.models import StrategiesConfig, StrategyConfig
from app.strategies.base import Strategy
from app.utils.logging import get_logger

logger = get_logger(__name__)

_REGISTRY: dict[str, type[Strategy]] = {}


class StrategyNotRegisteredError(KeyError):
    """Raised when a configured strategy name has no registered implementation."""


def register_strategy(
    name: str, strategy_class: type[Strategy], *, overwrite: bool = False
) -> None:
    """Register a strategy class under ``name``.

    Args:
        name: key used in ``configs/strategies.yaml``.
        strategy_class: a :class:`Strategy` subclass.
        overwrite: allow replacing an existing registration.

    Raises:
        TypeError: if the class is not a Strategy subclass.
        ValueError: on duplicate registration without ``overwrite``.
    """
    if not (isinstance(strategy_class, type) and issubclass(strategy_class, Strategy)):
        raise TypeError(
            f"{strategy_class!r} must be a subclass of Strategy to be registered as {name!r}"
        )

    key = name.strip().lower()
    if key in _REGISTRY and not overwrite:
        raise ValueError(
            f"Strategy {key!r} is already registered to {_REGISTRY[key].__name__}. "
            "Pass overwrite=True to replace it."
        )

    _REGISTRY[key] = strategy_class
    logger.debug("Registered strategy", extra={"strategy": key})


def get_strategy_class(name: str) -> type[Strategy]:
    """Look up a registered strategy class."""
    key = name.strip().lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise StrategyNotRegisteredError(
            f"No strategy registered as {name!r}. Registered: {known}"
        )
    return _REGISTRY[key]


def create_strategy(name: str, config: StrategyConfig | None = None) -> Strategy:
    """Instantiate one registered strategy."""
    return get_strategy_class(name)(config=config)


def available_strategies() -> list[str]:
    """Names of all registered strategies."""
    return sorted(_REGISTRY)


def build_enabled_strategies(config: StrategiesConfig) -> list[Strategy]:
    """Instantiate every enabled, registered strategy from configuration.

    A configured-but-unregistered strategy is logged and skipped rather than
    raising: one bad config entry should not prevent the other four strategies
    from running. A registered-but-unconfigured strategy is simply not built -
    configuration is what activates a strategy, not the mere fact of its
    existence in the codebase.
    """
    strategies: list[Strategy] = []
    for name in config.enabled_names():
        try:
            strategies.append(create_strategy(name, config.get(name)))
        except StrategyNotRegisteredError:
            logger.error(
                "Configured strategy is not registered and will be skipped",
                extra={"strategy": name, "registered": available_strategies()},
            )
    return strategies
