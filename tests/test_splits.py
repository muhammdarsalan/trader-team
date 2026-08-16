"""Temporal splits: separation, ordering and embargoes.

These are the most important tests in the phase. Every out-of-sample claim the
platform makes rests on the property that the bars a result is measured on did
not participate in producing it, and that property is structural - if the
splits are wrong, nothing downstream can detect it, because a contaminated
result looks exactly like a good one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.research.splits import (
    Segment,
    SegmentRole,
    SplitError,
    assert_temporal_separation,
    chronological_split,
    suggest_walk_forward_geometry,
    walk_forward_folds,
)


@pytest.fixture
def index() -> pd.DatetimeIndex:
    return pd.date_range("2015-01-01", periods=2_000, freq="1D", tz="UTC")


# ------------------------------------------------------- chronological splits

def test_split_produces_three_windows_in_order(index):
    segments = chronological_split(index, warmup_bars=200, embargo_bars=5)

    assert [s.name for s in segments] == ["in_sample", "validation", "out_of_sample"]
    assert [s.role for s in segments] == [
        SegmentRole.IN_SAMPLE, SegmentRole.VALIDATION, SegmentRole.OUT_OF_SAMPLE
    ]
    for earlier, later in zip(segments, segments[1:], strict=False):
        assert later.start_time > earlier.end_time


def test_out_of_sample_is_the_last_window_not_a_random_one(index):
    """Research runs forward. The untouched data must be the most recent data."""
    segments = chronological_split(index, warmup_bars=200)
    out_of_sample = segments[-1]

    assert out_of_sample.role is SegmentRole.OUT_OF_SAMPLE
    assert out_of_sample.end_time == index[-1] or out_of_sample.end >= len(index) - 3


def test_evaluation_windows_never_overlap(index):
    segments = chronological_split(index, warmup_bars=200, embargo_bars=10)
    scored = set()
    for segment in segments:
        window = set(range(segment.start, segment.end))
        assert not (window & scored), f"{segment.name} scores bars another window scored"
        scored |= window


def test_embargo_leaves_a_gap_between_windows(index):
    embargo = 25
    segments = chronological_split(index, warmup_bars=200, embargo_bars=embargo)

    for earlier, later in zip(segments, segments[1:], strict=False):
        assert later.start - earlier.end == embargo


def test_warmup_prefix_precedes_the_window_it_warms(index):
    """Reading history before a decision is what a trader has. After it is not."""
    segments = chronological_split(index, warmup_bars=200, embargo_bars=5)

    for segment in segments:
        assert segment.data_start <= segment.start
        assert segment.warmup_bars >= 0
        # The prefix must never reach past the window's own start.
        assert segment.data_start + segment.warmup_bars == segment.start


def test_first_window_still_gets_its_warmup(index):
    segments = chronological_split(index, warmup_bars=200)
    assert segments[0].data_start == 0
    assert segments[0].warmup_bars == 200


def test_fractions_change_the_proportions(index):
    heavy_oos = chronological_split(index, fractions=(0.3, 0.2, 0.5), warmup_bars=100)
    assert heavy_oos[2].bars > heavy_oos[0].bars


def test_a_series_too_short_to_split_raises_clearly(index):
    with pytest.raises(SplitError, match="cannot be split three ways"):
        chronological_split(index[:100], warmup_bars=200)


def test_windows_below_the_minimum_are_refused(index):
    """Three windows of eight bars are arithmetic, not a study."""
    with pytest.raises(SplitError, match="below the"):
        chronological_split(index[:450], warmup_bars=200, min_segment_bars=100)


def test_the_minimum_defaults_to_no_constraint(index):
    segments = chronological_split(index[:450], warmup_bars=200)
    assert len(segments) == 3


def test_zero_fractions_are_rejected(index):
    with pytest.raises(SplitError):
        chronological_split(index, fractions=(1.0, 0.0, 0.0), warmup_bars=100)


# ------------------------------------------------------------- walk-forward

def test_walk_forward_test_windows_follow_their_training_windows(index):
    folds = walk_forward_folds(
        index, train_bars=400, test_bars=100, warmup_bars=200, embargo_bars=5
    )

    assert folds
    for fold in folds:
        assert fold.test.start >= fold.train.end
        assert fold.test.start_time > fold.train.end_time


def test_walk_forward_test_windows_do_not_overlap_each_other(index):
    folds = walk_forward_folds(index, train_bars=400, test_bars=100, warmup_bars=200)
    scored = set()
    for fold in folds:
        window = set(range(fold.test.start, fold.test.end))
        assert not (window & scored)
        scored |= window


def test_folds_advance_through_time(index):
    folds = walk_forward_folds(index, train_bars=400, test_bars=100, warmup_bars=200)
    starts = [fold.test.start_time for fold in folds]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_rolling_windows_stay_the_same_length(index):
    folds = walk_forward_folds(index, train_bars=400, test_bars=100, warmup_bars=200)
    assert {fold.train.bars for fold in folds} == {400}


def test_anchored_windows_grow(index):
    folds = walk_forward_folds(
        index, train_bars=400, test_bars=100, warmup_bars=200, anchored=True
    )
    sizes = [fold.train.bars for fold in folds]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]
    assert len({fold.train.start for fold in folds}) == 1


def test_embargo_separates_train_from_test(index):
    folds = walk_forward_folds(
        index, train_bars=400, test_bars=100, warmup_bars=200, embargo_bars=15
    )
    for fold in folds:
        assert fold.test.start - fold.train.end == 15


def test_max_folds_is_respected(index):
    folds = walk_forward_folds(
        index, train_bars=300, test_bars=100, warmup_bars=100, max_folds=3
    )
    assert len(folds) == 3


def test_a_series_too_short_for_one_fold_raises(index):
    with pytest.raises(SplitError, match="too short for even one"):
        walk_forward_folds(index[:300], train_bars=400, test_bars=100, warmup_bars=200)


def test_suggested_geometry_fits_the_requested_folds(index):
    train, test = suggest_walk_forward_geometry(len(index), warmup_bars=200, folds=5)
    folds = walk_forward_folds(
        index, train_bars=train, test_bars=test, warmup_bars=200, max_folds=5
    )
    assert len(folds) == 5


# ---------------------------------------------------------------- the checker

def segment(name: str, start: int, end: int) -> Segment:
    index = pd.date_range("2020-01-01", periods=1_000, freq="1D", tz="UTC")
    return Segment(
        name=name, role=SegmentRole.TEST, start=start, end=end, data_start=max(0, start - 10),
        start_time=index[start], end_time=index[end - 1],
    )


def test_the_separation_checker_catches_an_overlap():
    with pytest.raises(SplitError, match="counted twice"):
        assert_temporal_separation([segment("a", 100, 200), segment("b", 150, 250)])


def test_the_separation_checker_accepts_adjacent_windows():
    assert_temporal_separation([segment("a", 100, 200), segment("b", 200, 300)])


def test_a_segment_cannot_warm_up_on_its_own_future():
    index = pd.date_range("2020-01-01", periods=500, freq="1D", tz="UTC")
    with pytest.raises(ValueError, match="may only precede"):
        Segment(
            name="bad", role=SegmentRole.TEST, start=100, end=200, data_start=150,
            start_time=index[100], end_time=index[199],
        )


def test_an_empty_evaluation_window_is_rejected():
    index = pd.date_range("2020-01-01", periods=500, freq="1D", tz="UTC")
    with pytest.raises(ValueError, match="empty evaluation window"):
        Segment(
            name="bad", role=SegmentRole.TEST, start=100, end=100, data_start=90,
            start_time=index[100], end_time=index[100],
        )
