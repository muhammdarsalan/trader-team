"""Timezone and timeframe handling.

Two rules hold everywhere in the platform:

1. Every timestamp that crosses a module boundary is timezone-aware UTC.
2. A bar is labelled by its OPEN time. A 4H bar stamped 12:00 covers
   [12:00, 16:00). This matters enormously for look-ahead bias: at the moment
   a bar is stamped, its close is not yet known.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

__all__ = [
    "TIMEFRAMES",
    "UTC",
    "Timeframe",
    "UnknownTimeframeError",
    "format_utc",
    "is_upsampling",
    "normalize_timeframe",
    "timeframe_minutes",
    "to_utc",
    "to_utc_index",
    "utcnow",
]


@dataclass(frozen=True)
class Timeframe:
    """A supported bar interval."""

    code: str  # canonical code used across the platform, e.g. "4H"
    pandas_freq: str  # pandas offset alias for resampling
    minutes: int  # nominal duration in minutes

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.code


# Canonical timeframe table. Strategies declare which of these they support.
TIMEFRAMES: dict[str, Timeframe] = {
    "1M": Timeframe("1M", "1min", 1),
    "5M": Timeframe("5M", "5min", 5),
    "15M": Timeframe("15M", "15min", 15),
    "30M": Timeframe("30M", "30min", 30),
    "1H": Timeframe("1H", "1h", 60),
    "4H": Timeframe("4H", "4h", 240),
    "1D": Timeframe("1D", "1D", 1440),
}

# Accepted spellings -> canonical code. Keeps config files forgiving.
_TIMEFRAME_ALIASES: dict[str, str] = {
    "1MIN": "1M", "M1": "1M", "1MINUTE": "1M",
    "5MIN": "5M", "M5": "5M",
    "15MIN": "15M", "M15": "15M",
    "30MIN": "30M", "M30": "30M",
    "1HOUR": "1H", "H1": "1H", "60M": "1H", "60MIN": "1H",
    "4HOUR": "4H", "H4": "4H", "240M": "4H",
    "D1": "1D", "DAY": "1D", "DAILY": "1D", "1DAY": "1D",
}


class UnknownTimeframeError(ValueError):
    """Raised when a timeframe string cannot be mapped to a canonical code."""


def normalize_timeframe(value: str | Timeframe) -> Timeframe:
    """Map any accepted spelling of a timeframe to its canonical ``Timeframe``.

    Raises:
        UnknownTimeframeError: if the value is not a supported timeframe.
    """
    if isinstance(value, Timeframe):
        return value
    key = str(value).strip().upper()
    key = _TIMEFRAME_ALIASES.get(key, key)
    if key not in TIMEFRAMES:
        supported = ", ".join(TIMEFRAMES)
        raise UnknownTimeframeError(f"Unsupported timeframe {value!r}. Supported: {supported}")
    return TIMEFRAMES[key]


def timeframe_minutes(value: str | Timeframe) -> int:
    return normalize_timeframe(value).minutes


def is_upsampling(source: str | Timeframe, target: str | Timeframe) -> bool:
    """True if going ``source`` -> ``target`` would invent bars we do not have.

    Resampling from 1H to 4H is aggregation and is fine. Going 4H to 1H is
    fabrication and the platform refuses to do it.
    """
    return timeframe_minutes(target) < timeframe_minutes(source)


def to_utc(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    """Coerce a single timestamp to timezone-aware UTC.

    Naive input is *assumed* to already be UTC rather than local time. Silently
    applying the machine's local timezone is a classic source of off-by-hours
    bugs that only appear when someone else runs the backtest.
    """
    ts = pd.Timestamp(value)
    return ts.tz_localize(UTC) if ts.tz is None else ts.tz_convert(UTC)


def to_utc_index(index: pd.Index, assume_tz: str = "UTC") -> pd.DatetimeIndex:
    """Coerce a pandas index to a timezone-aware UTC ``DatetimeIndex``.

    Args:
        index: index to convert.
        assume_tz: timezone to attach when the index is naive. Defaults to UTC.
            Pass the venue timezone (e.g. ``"America/New_York"``) when the
            source is known to publish local timestamps.
    """
    # errors="coerce": unparseable values become NaT rather than aborting the
    # whole load. The caller drops them and the quality engine reports the loss,
    # which beats one bad row making a decade of data unusable.
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=False, errors="coerce"))
    if idx.tz is None:
        idx = idx.tz_localize(assume_tz, ambiguous="NaT", nonexistent="NaT")
    return idx.tz_convert(UTC)


def utcnow() -> pd.Timestamp:
    """Current time as a timezone-aware UTC ``Timestamp``."""
    return pd.Timestamp.now(tz=UTC)


def format_utc(ts: pd.Timestamp | datetime) -> str:
    """Human-readable UTC rendering used in reports and logs."""
    return to_utc(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
