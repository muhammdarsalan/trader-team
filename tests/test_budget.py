import pytest

from app.research.budget import (
    SearchBudget,
    SearchBudgetExceededError,
    SearchBudgetTracker,
)


def test_budget_starts_with_all_trials_remaining():
    budget = SearchBudget(maximum_trials=5)

    assert budget.used_trials == 0
    assert budget.remaining_trials == 5
    assert budget.exhausted is False


def test_budget_consumes_trials():
    budget = SearchBudget(maximum_trials=5)

    updated = budget.consume(2)

    assert updated.used_trials == 2
    assert updated.remaining_trials == 3


def test_budget_becomes_exhausted():
    budget = SearchBudget(maximum_trials=2).consume(2)

    assert budget.exhausted is True
    assert budget.remaining_trials == 0


def test_budget_refuses_to_exceed_limit():
    budget = SearchBudget(maximum_trials=2).consume(2)

    with pytest.raises(SearchBudgetExceededError):
        budget.consume()


def test_budget_rejects_invalid_maximum():
    with pytest.raises(ValueError):
        SearchBudget(maximum_trials=0)


def test_tracker_consumes_trials():
    tracker = SearchBudgetTracker(maximum_trials=3)

    tracker.consume()
    tracker.consume()

    assert tracker.used_trials == 2
    assert tracker.remaining_trials == 1


def test_tracker_summary():
    tracker = SearchBudgetTracker(maximum_trials=2)
    tracker.consume()

    assert tracker.summary() == {
        "maximum_trials": 2,
        "used_trials": 1,
        "remaining_trials": 1,
        "exhausted": False,
    }


def test_tracker_refuses_to_exceed_limit():
    tracker = SearchBudgetTracker(maximum_trials=1)
    tracker.consume()

    with pytest.raises(SearchBudgetExceededError):
        tracker.consume()
