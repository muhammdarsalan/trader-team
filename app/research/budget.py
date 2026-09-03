"""Research search-budget controls.

A validation process must know how much optimisation/search it is allowed to
perform before it starts. Otherwise the search can expand until something looks
good and only afterwards ask whether too many attempts were made.
"""

from __future__ import annotations

from dataclasses import dataclass


class SearchBudgetExceededError(RuntimeError):
    """Raised when a study attempts to exceed its declared search budget."""


@dataclass(frozen=True)
class SearchBudget:
    """Fixed limit on distinct configurations evaluated during a study."""

    maximum_trials: int
    used_trials: int = 0

    def __post_init__(self) -> None:
        if self.maximum_trials < 1:
            raise ValueError("maximum_trials must be at least 1")
        if self.used_trials < 0:
            raise ValueError("used_trials cannot be negative")
        if self.used_trials > self.maximum_trials:
            raise ValueError("used_trials cannot exceed maximum_trials")

    @property
    def remaining_trials(self) -> int:
        return self.maximum_trials - self.used_trials

    @property
    def exhausted(self) -> bool:
        return self.remaining_trials == 0

    def consume(self, count: int = 1) -> SearchBudget:
        if count < 1:
            raise ValueError("count must be at least 1")

        if self.used_trials + count > self.maximum_trials:
            raise SearchBudgetExceededError(
                f"Search budget exceeded: attempted to use "
                f"{self.used_trials + count} trials with a maximum of "
                f"{self.maximum_trials}"
            )

        return SearchBudget(
            maximum_trials=self.maximum_trials,
            used_trials=self.used_trials + count,
        )


class SearchBudgetTracker:
    """Mutable tracker for one validation study's declared search budget."""

    def __init__(self, maximum_trials: int) -> None:
        self._budget = SearchBudget(maximum_trials=maximum_trials)

    @property
    def maximum_trials(self) -> int:
        return self._budget.maximum_trials

    @property
    def used_trials(self) -> int:
        return self._budget.used_trials

    @property
    def remaining_trials(self) -> int:
        return self._budget.remaining_trials

    @property
    def exhausted(self) -> bool:
        return self._budget.exhausted

    def consume(self, count: int = 1) -> None:
        self._budget = self._budget.consume(count)

    def summary(self) -> dict[str, int | bool]:
        return {
            "maximum_trials": self.maximum_trials,
            "used_trials": self.used_trials,
            "remaining_trials": self.remaining_trials,
            "exhausted": self.exhausted,
        }
