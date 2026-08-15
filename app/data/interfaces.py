"""The market-data provider abstraction.

The platform depends on this interface, never on a concrete vendor. Swapping
Yahoo for a CSV drop, or for a future broker feed, must not require touching a
single line of strategy, risk or backtesting code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from app.data.schema import MarketData
from app.utils.timeutils import Timeframe


class DataProviderError(RuntimeError):
    """Base class for provider failures."""


class DataUnavailableError(DataProviderError):
    """The provider cannot serve this symbol/timeframe/range at all.

    Distinct from a transient failure: retrying will not help. Raised for
    unknown tickers and for ranges outside the vendor's retention window.
    """


class DataFetchError(DataProviderError):
    """A transient failure (network, rate limit, malformed response)."""


class MarketDataProvider(ABC):
    """Source of OHLCV bars.

    Implementations must return frames satisfying the canonical schema in
    :mod:`app.data.schema`, with a tz-aware UTC index. Returning an empty
    ``MarketData`` for a legitimately empty range is correct; raising
    :class:`DataUnavailableError` for an unsupported request is correct;
    returning silently-truncated data is not.
    """

    #: Short stable identifier, used as the key in configs and cache paths.
    name: str = "abstract"

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        start: datetime | pd.Timestamp | str | None = None,
        end: datetime | pd.Timestamp | str | None = None,
    ) -> MarketData:
        """Fetch bars for ``symbol`` in ``[start, end]``.

        Args:
            symbol: canonical platform symbol (e.g. ``"XAUUSD"``), not a vendor
                ticker. The provider performs its own mapping.
            timeframe: any accepted timeframe spelling.
            start: inclusive lower bound; provider's earliest available if None.
            end: inclusive upper bound; latest available if None.

        Raises:
            DataUnavailableError: symbol or range not servable.
            DataFetchError: transient failure after retries.
        """

    @abstractmethod
    def get_latest_data(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        bars: int = 500,
    ) -> MarketData:
        """Fetch the most recent ``bars`` bars.

        Used by paper trading. The final bar may still be forming; the
        ``metadata["last_bar_complete"]`` flag says whether it is closed, and
        consumers must not treat a forming bar as final.
        """

    def supports(self, symbol: str, timeframe: str | Timeframe) -> bool:  # noqa: ARG002
        """Whether this provider can serve the pair. Optimistic by default."""
        return True

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} name={self.name!r}>"
