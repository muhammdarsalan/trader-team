"""The canonical OHLCV contract.

Every provider, cache read and resampling operation produces a DataFrame in
exactly this shape. Downstream code (features, regimes, strategies, backtester)
is entitled to assume it without re-checking:

    index    : tz-aware UTC DatetimeIndex, strictly increasing, unique,
               named "timestamp", labelled by bar OPEN time
    columns  : open, high, low, close, volume (float64)
    ordering : chronological
    contents : no NaN in OHLC; volume may be NaN where the venue has none

Bars are labelled by their open. A 4H bar stamped 12:00 covers [12:00, 16:00),
so at timestamp 12:00 the close is *not yet known*. Every consumer that acts on
a bar must act on the following bar's open, which is what the backtester does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.utils.timeutils import Timeframe, normalize_timeframe, to_utc_index

INDEX_NAME = "timestamp"

OPEN, HIGH, LOW, CLOSE, VOLUME = "open", "high", "low", "close", "volume"

#: Full canonical column set, in canonical order.
OHLCV_COLUMNS: tuple[str, ...] = (OPEN, HIGH, LOW, CLOSE, VOLUME)

#: Columns that must be present and finite.
REQUIRED_COLUMNS: tuple[str, ...] = (OPEN, HIGH, LOW, CLOSE)

#: Common vendor spellings mapped to canonical names.
COLUMN_ALIASES: dict[str, str] = {
    "o": OPEN, "op": OPEN, "openprice": OPEN, "adj open": OPEN,
    "h": HIGH, "hi": HIGH, "highprice": HIGH,
    "l": LOW, "lo": LOW, "lowprice": LOW,
    "c": CLOSE, "cl": CLOSE, "closeprice": CLOSE, "adj close": CLOSE,
    "adjclose": CLOSE, "adj_close": CLOSE, "price": CLOSE, "last": CLOSE,
    "v": VOLUME, "vol": VOLUME, "tickvol": VOLUME, "tick_volume": VOLUME,
    "volumefrom": VOLUME, "real_volume": VOLUME,
}

#: Common vendor spellings for the timestamp column.
TIMESTAMP_ALIASES: tuple[str, ...] = (
    "timestamp", "time", "date", "datetime", "date_time", "dt",
    "gmt time", "local time", "index", "unnamed: 0",
)


class SchemaError(ValueError):
    """Raised when a frame does not satisfy the canonical OHLCV contract."""


def empty_frame() -> pd.DataFrame:
    """A correctly-typed, correctly-indexed frame with zero rows.

    Returned instead of ``None`` when a query legitimately yields no bars, so
    callers never branch on ``None``.
    """
    index = pd.DatetimeIndex([], tz="UTC", name=INDEX_NAME)
    return pd.DataFrame(
        {col: pd.Series(dtype="float64") for col in OHLCV_COLUMNS},
        index=index,
    )


def coerce_schema(
    df: pd.DataFrame,
    *,
    assume_tz: str = "UTC",
    timestamp_column: str | None = None,
) -> pd.DataFrame:
    """Best-effort conversion of a vendor frame into the canonical shape.

    Handles column aliasing, timestamp promotion, timezone localisation and
    dtype coercion. It does *not* judge data quality - that is the quality
    engine's job. It only guarantees the structural contract.

    Args:
        df: raw vendor frame.
        assume_tz: timezone to attach to naive timestamps.
        timestamp_column: explicit timestamp column; auto-detected when None.

    Raises:
        SchemaError: when a required column cannot be found.
    """
    if df is None:
        raise SchemaError("Cannot coerce None into an OHLCV frame")

    out = df.copy()

    # Flatten a MultiIndex header (yfinance-style (field, ticker) columns).
    # Which level holds the field name varies by version, so prefer whichever
    # part is recognisable as an OHLCV field rather than assuming a position.
    if isinstance(out.columns, pd.MultiIndex):
        known = set(OHLCV_COLUMNS) | set(COLUMN_ALIASES) | set(TIMESTAMP_ALIASES)
        flattened = []
        for tup in out.columns:
            parts = [str(p).strip() for p in tup if str(p).strip()]
            match = next((p for p in parts if p.lower() in known), None)
            flattened.append(match if match is not None else (parts[0] if parts else ""))
        out.columns = flattened

    out.columns = [str(c).strip().lower() for c in out.columns]

    # --- timestamp -> index -------------------------------------------------
    ts_col = timestamp_column.lower() if timestamp_column else None
    if ts_col is None and not isinstance(out.index, pd.DatetimeIndex):
        ts_col = next((c for c in TIMESTAMP_ALIASES if c in out.columns), None)

    if ts_col is not None:
        if ts_col not in out.columns:
            raise SchemaError(f"Timestamp column {ts_col!r} not found. Columns: {list(out.columns)}")
        out = out.set_index(ts_col)
    elif not isinstance(out.index, pd.DatetimeIndex):
        raise SchemaError(
            "No timestamp column found and the index is not datetime-like. "
            f"Columns: {list(df.columns)}. Expected one of {TIMESTAMP_ALIASES}."
        )

    out.index = to_utc_index(out.index, assume_tz=assume_tz)
    out.index.name = INDEX_NAME

    # --- columns ------------------------------------------------------------
    renames = {c: COLUMN_ALIASES[c] for c in out.columns if c in COLUMN_ALIASES}
    out = out.rename(columns=renames)

    # Deduplicate columns that collided during aliasing (e.g. both "close" and
    # "adj close" mapping to close): keep the first, which is the direct match.
    out = out.loc[:, ~out.columns.duplicated(keep="first")]

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise SchemaError(
            f"Missing required column(s) {missing}. Available: {sorted(out.columns)}. "
            "Add an entry to COLUMN_ALIASES if the vendor uses a different spelling."
        )

    if VOLUME not in out.columns:
        out[VOLUME] = np.nan

    out = out[list(OHLCV_COLUMNS)]
    for col in OHLCV_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    # Timestamps that failed to parse became NaT and cannot be ordered.
    out = out[out.index.notna()]
    out = out.sort_index(kind="stable")

    return out


def validate_schema(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Assert the canonical contract holds.

    Args:
        df: frame to check.
        strict: also require a unique, strictly-increasing index and no NaN in
            OHLC. Set False to check structure only (used mid-pipeline, before
            cleaning has run).

    Returns:
        The same frame, unchanged.

    Raises:
        SchemaError: on any contract violation.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaError(f"Expected a DataFrame, got {type(df).__name__}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise SchemaError(f"Index must be a DatetimeIndex, got {type(df.index).__name__}")

    if df.index.tz is None:
        raise SchemaError("Index must be timezone-aware UTC; got a naive DatetimeIndex")

    if str(df.index.tz) not in {"UTC", "utc"}:
        raise SchemaError(f"Index timezone must be UTC, got {df.index.tz}")

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing column(s): {missing}")

    for col in OHLCV_COLUMNS:
        if not pd.api.types.is_float_dtype(df[col]):
            raise SchemaError(f"Column {col!r} must be float64, got {df[col].dtype}")

    if not strict or df.empty:
        return df

    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()][:5].tolist()
        raise SchemaError(f"Index contains duplicate timestamps, e.g. {dupes}")

    if not df.index.is_monotonic_increasing:
        raise SchemaError("Index must be strictly increasing (chronological order)")

    nan_counts = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS}
    bad = {c: n for c, n in nan_counts.items() if n}
    if bad:
        raise SchemaError(f"NaN values present in required OHLC columns: {bad}")

    return df


@dataclass(frozen=True)
class MarketData:
    """An OHLCV series together with everything needed to reproduce it.

    Carrying provenance alongside the numbers is what makes an experiment
    record meaningful: "which exact data produced this result?" must be
    answerable from the object itself.
    """

    symbol: str
    timeframe: Timeframe
    df: pd.DataFrame
    provider: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "timeframe", normalize_timeframe(self.timeframe))
        validate_schema(self.df, strict=False)

    # --- convenience accessors ---------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    @property
    def is_empty(self) -> bool:
        return self.df.empty

    @property
    def start(self) -> pd.Timestamp | None:
        return None if self.df.empty else self.df.index[0]

    @property
    def end(self) -> pd.Timestamp | None:
        return None if self.df.empty else self.df.index[-1]

    @property
    def has_volume(self) -> bool:
        """True when the volume column carries usable information."""
        if self.df.empty or VOLUME not in self.df:
            return False
        vol = self.df[VOLUME]
        return bool(vol.notna().any() and (vol.fillna(0) > 0).any())

    def slice(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> MarketData:
        """Return a new ``MarketData`` restricted to ``[start, end]`` inclusive."""
        df = self.df
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return self.replace(df=df)

    def replace(self, **changes: Any) -> MarketData:
        """Copy with fields replaced."""
        return MarketData(
            symbol=changes.get("symbol", self.symbol),
            timeframe=changes.get("timeframe", self.timeframe),
            df=changes.get("df", self.df),
            provider=changes.get("provider", self.provider),
            metadata=changes.get("metadata", dict(self.metadata)),
        )

    def describe(self) -> str:
        """One-line human summary, used in logs and reports."""
        if self.df.empty:
            return f"{self.symbol} {self.timeframe} [empty] via {self.provider}"
        return (
            f"{self.symbol} {self.timeframe} {len(self.df):,} bars "
            f"{self.start:%Y-%m-%d} -> {self.end:%Y-%m-%d} via {self.provider}"
        )
