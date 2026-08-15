"""Timeframe resampling.

Aggregating 1H bars into 4H is legitimate. Splitting 4H bars into 1H is not:
the intrabar path is unknowable, and any reconstruction is fabrication that
will flatter a backtest. This module refuses to upsample.

Bars are labelled by their open, so resampling uses left-closed, left-labelled
windows. Getting this wrong is a classic look-ahead bug: a right-labelled 4H
bar stamped 16:00 contains data from 12:00-16:00, and a strategy reading it at
"16:00" would be trading on information from a window that has only just ended
- or worse, a right-*closed* window would include the 16:00 print itself.
"""

from __future__ import annotations

import pandas as pd

from app.data.schema import CLOSE, HIGH, LOW, OPEN, VOLUME, validate_schema
from app.utils.timeutils import Timeframe, is_upsampling, normalize_timeframe

#: How each column collapses across a resampling window.
AGGREGATION: dict[str, str] = {
    OPEN: "first",
    HIGH: "max",
    LOW: "min",
    CLOSE: "last",
    VOLUME: "sum",
}


class ResampleError(ValueError):
    """Raised for an impossible or unsafe resampling request."""


def resample_ohlcv(
    df: pd.DataFrame,
    source_timeframe: str | Timeframe,
    target_timeframe: str | Timeframe,
    *,
    drop_empty: bool = True,
    origin: str = "start_day",
) -> pd.DataFrame:
    """Aggregate bars from ``source_timeframe`` to ``target_timeframe``.

    Args:
        df: canonical-schema frame at the source timeframe.
        source_timeframe: the frame's current timeframe.
        target_timeframe: desired timeframe; must be coarser or equal.
        drop_empty: remove windows containing no bars (weekends, holidays).
            Keep them only if you specifically want a gap-free calendar grid.
        origin: pandas resample origin. ``"start_day"`` anchors 4H windows to
            00:00/04:00/08:00 UTC, which keeps bar boundaries stable regardless
            of where the dataset happens to begin.

    Returns:
        A canonical-schema frame at the target timeframe.

    Raises:
        ResampleError: when the request would upsample.
    """
    validate_schema(df, strict=False)
    source = normalize_timeframe(source_timeframe)
    target = normalize_timeframe(target_timeframe)

    if is_upsampling(source, target):
        raise ResampleError(
            f"Refusing to upsample {source.code} -> {target.code}. The intrabar price path "
            "cannot be recovered, and reconstructing it would fabricate prices that never "
            "traded. Fetch finer-grained data from the provider instead."
        )

    if target.code == source.code:
        return df.copy()

    if df.empty:
        return df.copy()

    volume_all_nan = df[VOLUME].isna().all()

    resampled = df.resample(
        target.pandas_freq,
        label="left",
        closed="left",
        origin=origin,
    ).agg(AGGREGATION)

    if drop_empty:
        # A window with no source bars yields NaN open/high/low/close. Those are
        # non-trading periods, not missing data.
        resampled = resampled.dropna(subset=[OPEN, HIGH, LOW, CLOSE], how="all")

    # `sum` turns an all-NaN volume window into 0.0, which would misrepresent
    # "no volume data" as "zero volume traded".
    if volume_all_nan:
        resampled[VOLUME] = float("nan")
    else:
        empty_windows = df[VOLUME].resample(
            target.pandas_freq, label="left", closed="left", origin=origin
        ).count()
        empty_windows = empty_windows.reindex(resampled.index, fill_value=0)
        resampled.loc[empty_windows == 0, VOLUME] = float("nan")

    resampled = resampled.astype("float64")
    resampled.index.name = df.index.name
    validate_schema(resampled, strict=False)
    return resampled


def best_source_timeframe(
    target: str | Timeframe,
    available: list[str] | tuple[str, ...],
) -> str | None:
    """Pick the coarsest available timeframe that can still produce ``target``.

    Coarsest-that-works minimises download size and resampling cost: building
    4H bars from 1H beats building them from 1M.

    Returns:
        The chosen timeframe code, or None when nothing available is fine enough.
    """
    tgt = normalize_timeframe(target)
    usable = [normalize_timeframe(tf) for tf in available]
    usable = [tf for tf in usable if tf.minutes <= tgt.minutes]
    if not usable:
        return None
    # Prefer an exact match; otherwise the coarsest that divides evenly, and
    # failing that simply the coarsest available.
    exact = [tf for tf in usable if tf.code == tgt.code]
    if exact:
        return exact[0].code
    divisible = [tf for tf in usable if tgt.minutes % tf.minutes == 0]
    pool = divisible or usable
    return max(pool, key=lambda tf: tf.minutes).code
