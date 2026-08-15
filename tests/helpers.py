"""Test helpers, chiefly the look-ahead detector.

The most valuable test in a trading platform is not "does this indicator equal
the number I computed by hand" - it is "could this value have been known at the
time it claims to describe".

:func:`assert_causal` answers that mechanically. Compute a feature on the first
``n`` bars, compute it again on the full series, and compare the overlap. If
they differ, the full-series computation used information from bars after ``n``,
which is look-ahead bias. No amount of reading the code proves this as reliably
as running it.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

Computation = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]


def assert_causal(
    compute: Computation,
    df: pd.DataFrame,
    *,
    prefix_lengths: tuple[int, ...] | None = None,
    tolerance: float = 1e-9,
    label: str = "feature",
) -> None:
    """Assert a computation uses no information from future bars.

    Args:
        compute: takes an OHLCV frame, returns a Series or DataFrame indexed
            like it.
        df: the full test frame.
        prefix_lengths: bar counts to truncate at. Defaults to a spread across
            the series.
        tolerance: absolute tolerance for float comparison.
        label: name used in failure messages.

    Raises:
        AssertionError: if any value computed on a prefix differs from the same
            value computed on the full series.
    """
    n = len(df)
    if prefix_lengths is None:
        prefix_lengths = tuple(
            length for length in (int(n * 0.5), int(n * 0.7), int(n * 0.9), n - 1) if length > 20
        )

    full = compute(df)
    full_frame = full.to_frame() if isinstance(full, pd.Series) else full

    for length in prefix_lengths:
        partial = compute(df.iloc[:length])
        partial_frame = partial.to_frame() if isinstance(partial, pd.Series) else partial

        for column in partial_frame.columns:
            if column not in full_frame.columns:
                raise AssertionError(f"{label}: column {column!r} missing from full computation")

            expected = full_frame[column].iloc[:length]
            actual = partial_frame[column]

            is_numeric = pd.api.types.is_numeric_dtype(expected) and not (
                pd.api.types.is_bool_dtype(expected) or pd.api.types.is_bool_dtype(actual)
            )

            if is_numeric:
                both_nan = expected.isna() & actual.isna()
                close_enough = (expected - actual).abs() <= tolerance
                mismatches = ~both_nan & ~close_enough.fillna(False)
                # A NaN on exactly one side is also a mismatch.
                mismatches |= expected.isna() != actual.isna()
            else:
                # Booleans, strings and object columns: exact comparison.
                # Subtracting booleans is a TypeError, and str comparison
                # handles the categorical structure labels correctly.
                mismatches = expected.astype(str) != actual.astype(str)

            if mismatches.any():
                first = mismatches[mismatches].index[0]
                raise AssertionError(
                    f"{label}: column {column!r} is NOT causal.\n"
                    f"  Truncating to {length} bars changed the value at {first}.\n"
                    f"  full-series value:  {full_frame.loc[first, column]!r}\n"
                    f"  prefix-only value:  {partial_frame.loc[first, column]!r}\n"
                    f"  This means the full computation used data from bars after {first} - "
                    f"information that would not exist in live trading."
                )


def assert_no_nan_after_warmup(
    result: pd.Series | pd.DataFrame,
    warmup: int,
    *,
    label: str = "feature",
    allow_columns: tuple[str, ...] = (),
) -> None:
    """Assert a computation is fully defined once past its warm-up period.

    A feature that stays NaN forever is a silent failure: strategies simply
    never fire and nothing reports why.
    """
    frame = result.to_frame() if isinstance(result, pd.Series) else result
    tail = frame.iloc[warmup:]

    for column in frame.columns:
        if column in allow_columns:
            continue
        if tail[column].isna().all():
            raise AssertionError(
                f"{label}: column {column!r} is entirely NaN after the {warmup}-bar warm-up. "
                "A permanently-undefined feature means strategies silently never fire."
            )
