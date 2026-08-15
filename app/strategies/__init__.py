"""Independent strategy modules.

Each strategy is self-contained: it receives data, features and the regime, and
returns a standardized :class:`~app.signals.models.Signal`. None of them knows
about the graph, the risk engine, the backtester or each other.

Adding a strategy:

1. Subclass :class:`~app.strategies.base.Strategy` in a new module.
2. Declare its :class:`~app.strategies.base.StrategyMetadata`.
3. Register it below.
4. Add a block to ``configs/strategies.yaml``.

Nothing else in the codebase changes.
"""

from app.strategies.base import (
    Strategy,
    StrategyContext,
    StrategyError,
    StrategyMetadata,
)
from app.strategies.breakout import BreakoutStrategy
from app.strategies.mean_reversion import MeanReversionStrategy
from app.strategies.momentum import MomentumStrategy
from app.strategies.registry import (
    StrategyNotRegisteredError,
    available_strategies,
    build_enabled_strategies,
    create_strategy,
    get_strategy_class,
    register_strategy,
)
from app.strategies.support_resistance import SupportResistanceStrategy
from app.strategies.trend import TrendFollowingStrategy


def _register_builtin_strategies() -> None:
    """Register the strategies shipped with the platform."""
    builtins: list[type[Strategy]] = [
        TrendFollowingStrategy,
        SupportResistanceStrategy,
        BreakoutStrategy,
        MeanReversionStrategy,
        MomentumStrategy,
    ]
    for cls in builtins:
        register_strategy(cls.metadata.name, cls, overwrite=True)


_register_builtin_strategies()

__all__ = [
    "BreakoutStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "Strategy",
    "StrategyContext",
    "StrategyError",
    "StrategyMetadata",
    "StrategyNotRegisteredError",
    "SupportResistanceStrategy",
    "TrendFollowingStrategy",
    "available_strategies",
    "build_enabled_strategies",
    "create_strategy",
    "get_strategy_class",
    "register_strategy",
]
