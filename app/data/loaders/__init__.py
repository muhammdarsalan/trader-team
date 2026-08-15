"""Concrete market-data providers and the provider registry."""

from app.data.loaders.csv_loader import CsvProvider
from app.data.loaders.registry import (
    ProviderNotRegisteredError,
    available_providers,
    get_provider,
    register_provider,
)
from app.data.loaders.synthetic import SyntheticProvider
from app.data.loaders.yahoo import YahooProvider


def _register_builtin_providers() -> None:
    """Register the providers shipped with the platform.

    Third-party providers register themselves the same way - one call, no
    changes anywhere else in the codebase.
    """
    builtins: list[tuple[str, type]] = [
        ("yahoo", YahooProvider),
        ("csv", CsvProvider),
        ("synthetic", SyntheticProvider),
    ]
    for name, cls in builtins:
        register_provider(name, _adapt(cls), overwrite=True)


def _adapt(cls: type):
    """Drop keyword arguments a provider does not accept.

    The service passes `assets` and `config` uniformly; simpler providers
    (the synthetic one) take neither.
    """
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}

    def factory(**kwargs):
        return cls(**{k: v for k, v in kwargs.items() if k in accepted})

    return factory


_register_builtin_providers()

__all__ = [
    "CsvProvider",
    "ProviderNotRegisteredError",
    "SyntheticProvider",
    "YahooProvider",
    "available_providers",
    "get_provider",
    "register_provider",
]
