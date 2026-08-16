"""Temporal splits.

Every validation claim in this package rests on one structural property: the
bars a result is measured on must not have participated in producing the thing
being measured. Randomly shuffling rows into train and test sets - the default
in most of machine learning - destroys that property completely on a time
series, because a randomly chosen "test" bar sits between two "training" bars
whose prices it all but determines.

So splits here are strictly chronological, and three details make them honest:

**Warm-up prefixes are not evaluation.** A feature needs history before it is
defined. A segment therefore carries a read-only prefix of bars preceding it,
used to warm indicators up and never to decide or to score. Reading price
history that precedes a decision is not look-ahead; that is simply what a
trader has. Reading anything after it is, and no prefix here ever extends
forward.

**Segments are separated by an embargo.** A trade opened near the end of one
window can still be open at the start of the next. Without a gap, the same
market episode is scored twice and the second scoring is contaminated by a
position the first one opened. The embargo drops bars between windows so each
segment resolves its own trades.

**Nothing is scored twice.** :func:`assert_temporal_separation` checks the
evaluation windows are ordered and disjoint, and it is called by the split
constructors themselves rather than left to a test to remember.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class SegmentRole(StrEnum):
    """What a segment is for. The role decides how its numbers may be read."""

    #: Where the configuration was developed. Its results are description, not
    #: evidence: the configuration exists because of these bars.
    IN_SAMPLE = "IN_SAMPLE"
    #: Used to choose between candidates. Once it has been looked at more than
    #: a couple of times it is in-sample too, and the report says so.
    VALIDATION = "VALIDATION"
    #: Touched once, at the end. The only segment whose result carries weight,
    #: and only for as long as it stays untouched.
    OUT_OF_SAMPLE = "OUT_OF_SAMPLE"
    #: A walk-forward fold's training window.
    TRAIN = "TRAIN"
    #: A walk-forward fold's testing window.
    TEST = "TEST"


@dataclass(frozen=True)
class Segment:
    """A contiguous, chronological slice of a series.

    Two ranges matter and they are not the same range:

    ``[data_start, end)`` is the data the run is *given* - evaluation window
    plus the warm-up prefix in front of it.

    ``[start, end)`` is the window the run is *scored on*. Only these bars may
    produce a decision, and only their outcomes reach the metrics.
    """

    name: str
    role: SegmentRole
    #: First bar of the evaluation window, as a position in the parent index.
    start: int
    #: One past the last bar of the evaluation window.
    end: int
    #: First bar of the data slice, at or before ``start``.
    data_start: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp

    def __post_init__(self) -> None:
        if self.data_start > self.start:
            raise ValueError(
                f"Segment {self.name!r} has a warm-up prefix starting at {self.data_start}, "
                f"after its evaluation start {self.start}. A prefix may only precede."
            )
        if self.end <= self.start:
            raise ValueError(
                f"Segment {self.name!r} has an empty evaluation window "
                f"[{self.start}, {self.end})"
            )

    @property
    def bars(self) -> int:
        """Bars scored."""
        return self.end - self.start

    @property
    def warmup_bars(self) -> int:
        """Bars supplied purely to warm features up."""
        return self.start - self.data_start

    @property
    def total_bars(self) -> int:
        return self.end - self.data_start

    def slice_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """The data this segment is run on, warm-up prefix included."""
        return df.iloc[self.data_start : self.end]

    def describe(self) -> str:
        return (
            f"{self.name} [{self.role}] {self.start_time:%Y-%m-%d} -> "
            f"{self.end_time:%Y-%m-%d}  {self.bars:,} bars scored "
            f"(+{self.warmup_bars:,} warm-up)"
        )


@dataclass(frozen=True)
class WalkForwardFold:
    """One train/test pair in a rolling evaluation."""

    index: int
    train: Segment
    test: Segment

    def describe(self) -> str:
        return (
            f"fold {self.index:02d}: train {self.train.start_time:%Y-%m-%d}"
            f"->{self.train.end_time:%Y-%m-%d} ({self.train.bars:,}), "
            f"test {self.test.start_time:%Y-%m-%d}->{self.test.end_time:%Y-%m-%d} "
            f"({self.test.bars:,})"
        )


class SplitError(ValueError):
    """Raised when the requested split cannot be made from the data supplied."""


# --------------------------------------------------------------------- checks


def assert_temporal_separation(segments: list[Segment]) -> None:
    """Assert evaluation windows are ordered in time and never overlap.

    This is the property the whole package depends on, so it is asserted
    mechanically rather than argued for in a comment. Warm-up prefixes are
    deliberately exempt: they are allowed to reach back into an earlier
    segment, because past prices are what any trader has at that moment. What
    is forbidden is any bar being *scored* twice, or a later segment being
    scored before an earlier one.

    Raises:
        SplitError: on any overlap or out-of-order window.
    """
    ordered = sorted(segments, key=lambda s: s.start)
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if later.start < earlier.end:
            raise SplitError(
                f"Segments {earlier.name!r} and {later.name!r} both score bars "
                f"[{later.start}, {earlier.end}). Overlapping evaluation windows mean "
                "the same market episode is counted twice, and the second count is "
                "contaminated by the first."
            )
        if later.data_start > later.end:
            raise SplitError(f"Segment {later.name!r} has an inverted data range")


def resolve_embargo(embargo_bars: int) -> int:
    """Embargo actually used, never negative."""
    return max(0, int(embargo_bars))


# --------------------------------------------------------------------- splits


def chronological_split(
    index: pd.DatetimeIndex,
    *,
    fractions: tuple[float, float, float] = (0.5, 0.2, 0.3),
    warmup_bars: int = 0,
    embargo_bars: int = 0,
    min_segment_bars: int = 0,
) -> list[Segment]:
    """Split a series into in-sample, validation and out-of-sample windows.

    The out-of-sample window is the *last* one, not a random third. Research
    proceeds forward through time, so the untouched data has to be the data
    that had not happened yet when the configuration was designed.

    Args:
        index: the parent series index.
        fractions: (in-sample, validation, out-of-sample) proportions. Need not
            sum to exactly 1.0; they are normalised.
        warmup_bars: bars of read-only history each window is given in front of
            itself, so that features are defined at its first scored bar.
        embargo_bars: bars dropped between windows so a trade opened in one
            cannot still be open in the next.
        min_segment_bars: refuse to produce a window shorter than this. Zero
            permits any non-empty window, which is right for a caller that wants
            the arithmetic; a study should pass something related to its
            warm-up, since a window shorter than the indicators' own memory
            cannot produce an independent result.

    Raises:
        SplitError: when the series is too short to give every window a
            non-empty evaluation range, or one shorter than ``min_segment_bars``.
    """
    n = len(index)
    if n == 0:
        raise SplitError("Cannot split an empty series")
    if any(f < 0 for f in fractions) or sum(fractions) <= 0:
        raise SplitError(f"Split fractions must be non-negative and sum above zero: {fractions}")

    warmup = max(0, int(warmup_bars))
    embargo = resolve_embargo(embargo_bars)

    # The first window must fit its own warm-up, and each later window loses
    # the embargo. Everything left over is what there is to divide.
    overhead = warmup + 2 * embargo
    tradable = n - overhead
    if tradable < 3:
        raise SplitError(
            f"A {n:,}-bar series cannot be split three ways: {warmup:,} bars of warm-up "
            f"and {embargo:,}-bar embargoes leave {tradable:,} bars to divide. Use a "
            "longer history, a shorter warm-up, or a smaller embargo."
        )

    total = sum(fractions)
    weights = [f / total for f in fractions]
    sizes = [int(tradable * w) for w in weights]
    sizes[-1] = tradable - sizes[0] - sizes[1]   # the remainder lands out-of-sample

    roles = (SegmentRole.IN_SAMPLE, SegmentRole.VALIDATION, SegmentRole.OUT_OF_SAMPLE)
    names = ("in_sample", "validation", "out_of_sample")

    segments: list[Segment] = []
    cursor = warmup
    for name, role, size in zip(names, roles, sizes, strict=True):
        if size <= 0:
            raise SplitError(
                f"Split fractions {fractions} give the {name} window {size} bars on a "
                f"{n:,}-bar series. Every window must be scored on at least one bar."
            )
        if size < min_segment_bars:
            raise SplitError(
                f"The {name} window would be {size:,} bars, below the {min_segment_bars:,} "
                f"minimum. A {n:,}-bar series with a {warmup:,}-bar warm-up does not hold "
                "three windows long enough to be read separately."
            )
        start, end = cursor, cursor + size
        segments.append(
            Segment(
                name=name,
                role=role,
                start=start,
                end=end,
                data_start=max(0, start - warmup),
                start_time=index[start],
                end_time=index[end - 1],
            )
        )
        cursor = end + embargo

    assert_temporal_separation(segments)
    return segments


def walk_forward_folds(
    index: pd.DatetimeIndex,
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    warmup_bars: int = 0,
    embargo_bars: int = 0,
    anchored: bool = False,
    max_folds: int | None = None,
) -> list[WalkForwardFold]:
    """Roll a train/test pair forward through the series.

    This is the closest thing available to asking "would this have worked, run
    the way it would actually have been run?" - decide using only what had
    happened, trade the period that followed, then move on. A single
    in-sample/out-of-sample split answers that question once; walk-forward
    answers it repeatedly, which is what makes a run of good folds harder to
    get by luck than one good split.

    Args:
        train_bars: bars in each training window.
        test_bars: bars in each testing window.
        step_bars: how far the window advances per fold. Defaults to
            ``test_bars``, which makes the test windows tile the series without
            overlapping - the only setting under which the concatenated test
            results form one continuous out-of-sample record.
        warmup_bars: read-only history in front of each window.
        embargo_bars: bars dropped between train and test, and between folds.
        anchored: keep the training window's start fixed so it grows, instead of
            rolling a fixed-length window. Anchored uses all history available at
            each point; rolling adapts faster to a changed market. Neither is
            correct in general.
        max_folds: stop after this many folds.

    Raises:
        SplitError: when not even one fold fits.
    """
    n = len(index)
    train_bars = int(train_bars)
    test_bars = int(test_bars)
    step = int(step_bars) if step_bars else test_bars
    warmup = max(0, int(warmup_bars))
    embargo = resolve_embargo(embargo_bars)

    if train_bars <= 0 or test_bars <= 0:
        raise SplitError(f"train_bars and test_bars must be positive, got {train_bars}/{test_bars}")
    if step <= 0:
        raise SplitError(f"step_bars must be positive, got {step}")

    needed = warmup + train_bars + embargo + test_bars
    if n < needed:
        raise SplitError(
            f"A {n:,}-bar series is too short for even one walk-forward fold: "
            f"{warmup:,} warm-up + {train_bars:,} train + {embargo:,} embargo + "
            f"{test_bars:,} test = {needed:,} bars required."
        )

    folds: list[WalkForwardFold] = []
    anchor = warmup
    train_start = warmup

    while True:
        train_end = train_start + train_bars
        test_start = train_end + embargo
        test_end = test_start + test_bars
        if test_end > n:
            break

        effective_train_start = anchor if anchored else train_start

        train = Segment(
            name=f"fold_{len(folds):02d}_train",
            role=SegmentRole.TRAIN,
            start=effective_train_start,
            end=train_end,
            data_start=max(0, effective_train_start - warmup),
            start_time=index[effective_train_start],
            end_time=index[train_end - 1],
        )
        test = Segment(
            name=f"fold_{len(folds):02d}_test",
            role=SegmentRole.TEST,
            start=test_start,
            end=test_end,
            data_start=max(0, test_start - warmup),
            start_time=index[test_start],
            end_time=index[test_end - 1],
        )
        # Within a fold, the test window must begin strictly after the training
        # window ends. This is the one guarantee the whole analysis rests on.
        if test.start < train.end:
            raise SplitError(
                f"Fold {len(folds)} would test on bars it trained on "
                f"({test.start} < {train.end})"
            )

        folds.append(WalkForwardFold(index=len(folds), train=train, test=test))
        if max_folds is not None and len(folds) >= max_folds:
            break
        train_start += step

    if not folds:
        raise SplitError("No walk-forward fold fitted inside the series")

    assert_temporal_separation([fold.test for fold in folds])
    return folds


def suggest_walk_forward_geometry(
    n_bars: int, warmup_bars: int, folds: int = 5, test_fraction: float = 0.25
) -> tuple[int, int]:
    """Train and test sizes that fit ``folds`` non-overlapping folds into a series.

    A convenience so that a caller with 3,000 bars does not have to solve for
    the window sizes by hand, and so the CLI's defaults adapt to the history
    actually available rather than assuming a decade of it.
    """
    folds = max(1, int(folds))
    usable = max(0, n_bars - warmup_bars)
    if usable < folds * 4:
        raise SplitError(
            f"{n_bars:,} bars with a {warmup_bars:,}-bar warm-up leaves {usable:,} usable, "
            f"too few for {folds} folds."
        )
    test_bars = max(1, int(usable * test_fraction / folds))
    train_bars = max(1, usable - test_bars * folds)
    return train_bars, test_bars
