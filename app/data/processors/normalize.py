"""OHLCV cleaning.

Cleaning is deliberately conservative and always *reports* what it did. Silent
repair is how corrupt data reaches a strategy wearing a disguise: the numbers
look plausible, the backtest looks fine, and nobody knows the series was
patched. Every action here is counted and surfaced in the returned report.

The one thing this module will not do is invent prices. Bars that cannot be
repaired are dropped, never interpolated, because an interpolated bar is a
price at which nobody could have traded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.data.schema import CLOSE, HIGH, LOW, OPEN, REQUIRED_COLUMNS, VOLUME, validate_schema


@dataclass
class CleaningReport:
    """What cleaning changed, in counts."""

    rows_in: int = 0
    rows_out: int = 0
    duplicates_dropped: int = 0
    reordered: bool = False
    nan_rows_dropped: int = 0
    nonpositive_rows_dropped: int = 0
    ohlc_bounds_repaired: int = 0
    negative_volume_nulled: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out

    @property
    def changed(self) -> bool:
        return bool(
            self.rows_dropped
            or self.reordered
            or self.ohlc_bounds_repaired
            or self.negative_volume_nulled
        )

    def summary(self) -> str:
        if not self.changed:
            return f"No cleaning required ({self.rows_in:,} rows)"
        parts = [f"{self.rows_in:,} -> {self.rows_out:,} rows"]
        if self.duplicates_dropped:
            parts.append(f"{self.duplicates_dropped} duplicate(s) dropped")
        if self.reordered:
            parts.append("reordered chronologically")
        if self.nan_rows_dropped:
            parts.append(f"{self.nan_rows_dropped} incomplete row(s) dropped")
        if self.nonpositive_rows_dropped:
            parts.append(f"{self.nonpositive_rows_dropped} non-positive price row(s) dropped")
        if self.ohlc_bounds_repaired:
            parts.append(f"{self.ohlc_bounds_repaired} high/low bound(s) repaired")
        if self.negative_volume_nulled:
            parts.append(f"{self.negative_volume_nulled} negative volume(s) nulled")
        return "; ".join(parts)


def clean_ohlcv(
    df: pd.DataFrame,
    *,
    duplicate_policy: str = "last",
    repair_ohlc_bounds: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a canonical-schema frame.

    Steps, in order:

    1. Sort chronologically.
    2. Drop duplicate timestamps (keeping ``duplicate_policy``).
    3. Drop rows with any missing OHLC value.
    4. Drop rows with a zero or negative price.
    5. Repair high/low bounds, where high < max(open, close) or
       low > min(open, close). This is a stamping error in the extremes, not a
       fabricated price: the true extreme is at least the open/close, so
       widening the wick to contain them is the minimal correction.
    6. Null out negative volume (impossible; recorded as unknown, not zero).

    Args:
        df: canonical-schema frame.
        duplicate_policy: ``"last"``, ``"first"``, or ``"drop"`` (remove all
            copies of a duplicated timestamp, for when neither can be trusted).
        repair_ohlc_bounds: repair rather than drop bad high/low bounds.

    Returns:
        The cleaned frame and a :class:`CleaningReport`.
    """
    validate_schema(df, strict=False)
    report = CleaningReport(rows_in=len(df))

    if df.empty:
        report.rows_out = 0
        return df.copy(), report

    out = df

    # --- 1. chronological order --------------------------------------------
    if not out.index.is_monotonic_increasing:
        out = out.sort_index(kind="stable")
        report.reordered = True

    # --- 2. duplicate timestamps -------------------------------------------
    if out.index.has_duplicates:
        before = len(out)
        if duplicate_policy == "drop":
            out = out[~out.index.duplicated(keep=False)]
        elif duplicate_policy in {"first", "last"}:
            out = out[~out.index.duplicated(keep=duplicate_policy)]
        else:
            raise ValueError(
                f"duplicate_policy must be 'first', 'last' or 'drop', got {duplicate_policy!r}"
            )
        report.duplicates_dropped = before - len(out)

    # --- 3. incomplete rows -------------------------------------------------
    complete = out[list(REQUIRED_COLUMNS)].notna().all(axis=1)
    if not complete.all():
        report.nan_rows_dropped = int((~complete).sum())
        out = out[complete]

    # --- 4. impossible prices -----------------------------------------------
    positive = (out[list(REQUIRED_COLUMNS)] > 0).all(axis=1)
    if not positive.all():
        report.nonpositive_rows_dropped = int((~positive).sum())
        out = out[positive]

    if out.empty:
        report.rows_out = 0
        report.notes.append("All rows removed during cleaning")
        return out.copy(), report

    # --- 5a. unrepairable bars ----------------------------------------------
    # high < low is corrupt beyond rescue: which of the two is wrong cannot be
    # known. This must be handled BEFORE bound repair, because clamping both to
    # the candle body would "fix" it into a zero-range bar that never traded.
    inverted = out[HIGH] < out[LOW]
    if inverted.any():
        count = int(inverted.sum())
        out = out[~inverted]
        report.notes.append(f"{count} row(s) with high < low dropped (unrepairable)")

    # --- 5b. high/low bounds -------------------------------------------------
    # A high below the open/close is a stamping error in the extreme, not a
    # fabricated price: the true extreme is at least the open/close, so widening
    # the wick to contain the body is the minimal honest correction.
    body_high = out[[OPEN, CLOSE]].max(axis=1)
    body_low = out[[OPEN, CLOSE]].min(axis=1)
    bad_high = out[HIGH] < body_high
    bad_low = out[LOW] > body_low
    broken = bad_high | bad_low

    if broken.any():
        count = int(broken.sum())
        if repair_ohlc_bounds:
            out = out.copy()
            out.loc[bad_high, HIGH] = body_high[bad_high]
            out.loc[bad_low, LOW] = body_low[bad_low]
            report.ohlc_bounds_repaired = count
        else:
            out = out[~broken]
            report.notes.append(f"{count} row(s) with invalid OHLC bounds dropped")

    # --- 6. volume -----------------------------------------------------------
    if VOLUME in out.columns:
        negative = out[VOLUME] < 0
        if negative.any():
            out = out.copy()
            out.loc[negative, VOLUME] = np.nan
            report.negative_volume_nulled = int(negative.sum())

    report.rows_out = len(out)
    return out.copy(), report
