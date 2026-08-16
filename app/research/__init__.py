"""Research and validation.

Phase 3 answered "what would this configuration have done on this history?".
That is the weakest question in the project. This package asks the harder ones:

- Does it still do it on data it was never developed on?
- Does it keep doing it as the window rolls forward through time?
- Does the result survive the trades arriving in a different order?
- Does it survive small changes to parameters nobody has any reason to believe
  are exactly right?
- How much of what we are looking at is explained by how many things we tried?

Nothing here optimises toward a return, and nothing here concludes that a
configuration is profitable. The output is evidence and its limits, including
the frequent and entirely legitimate conclusion that a configuration has not
been shown to work.
"""

from app.research.experiments import ExperimentStore, reproducible_experiment_id
from app.research.harness import ResearchHarness, SegmentRun
from app.research.splits import (
    Segment,
    SegmentRole,
    WalkForwardFold,
    assert_temporal_separation,
    chronological_split,
    walk_forward_folds,
)

__all__ = [
    "ExperimentStore",
    "ResearchHarness",
    "Segment",
    "SegmentRole",
    "SegmentRun",
    "WalkForwardFold",
    "assert_temporal_separation",
    "chronological_split",
    "reproducible_experiment_id",
    "walk_forward_folds",
]
