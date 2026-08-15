"""Provider registry.

Adding a data source means writing one class and registering it. No other
module changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.data.interfaces import MarketDataProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)

ProviderFactory = Callable[..., MarketDataProvider]

_REGISTRY: dict[str, ProviderFactory] = {}


class ProviderNotRegisteredError(KeyError):
    """Raised when a configured provider name has no registered implementation."""


def register_provider(name: str, factory: ProviderFactory, *, overwrite: bool = False) -> None:
    """Register a provider factory under ``name``.

    Args:
        name: key used in ``configs/data.yaml``.
        factory: callable returning a :class:`MarketDataProvider`.
        overwrite: allow replacing an existing registration.

    Raises:
        ValueError: on duplicate registration without ``overwrite``.
    """
    key = name.strip().lower()
    if key in _REGISTRY and not overwrite:
        raise ValueError(
            f"Provider {key!r} is already registered. Pass overwrite=True to replace it."
        )
    _REGISTRY[key] = factory
    logger.debug("Registered market-data provider", extra={"provider": key})


def get_provider(name: str, **kwargs: Any) -> MarketDataProvider:
    """Instantiate the provider registered under ``name``."""
    key = name.strip().lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ProviderNotRegisteredError(
            f"No market-data provider registered as {name!r}. Registered: {known}"
        )
    return _REGISTRY[key](**kwargs)


def available_providers() -> list[str]:
    """Names of all registered providers."""
    return sorted(_REGISTRY)
