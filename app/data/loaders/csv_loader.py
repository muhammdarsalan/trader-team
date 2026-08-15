"""CSV provider.

The platform must never be hostage to one vendor. Any OHLCV file you can
export - MetaTrader, Dukascopy, HistData, a Kaggle dataset, a broker report -
can be dropped into ``data/raw/`` and used with no code changes.

Expected filename: ``<SYMBOL>_<TIMEFRAME>.csv``, e.g. ``XAUUSD_1D.csv``.
Header naming is forgiving (see ``COLUMN_ALIASES`` in :mod:`app.data.schema`);
separator, decimal mark and timestamp format are auto-detected.
"""

from __future__ import annotations

import csv as csv_module
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.config.models import AssetUniverse, ProviderConfig
from app.data.interfaces import DataFetchError, DataUnavailableError, MarketDataProvider
from app.data.schema import MarketData, coerce_schema
from app.utils.logging import get_logger
from app.utils.paths import raw_dir
from app.utils.timeutils import Timeframe, normalize_timeframe, to_utc, utcnow

logger = get_logger(__name__)


class CsvProvider(MarketDataProvider):
    """Bars read from local CSV files."""

    name = "csv"

    def __init__(
        self,
        assets: AssetUniverse | None = None,
        config: ProviderConfig | None = None,
        directory: Path | str | None = None,
    ) -> None:
        self.assets = assets
        self.config = config or ProviderConfig()
        if directory is not None:
            base = Path(directory)
        else:
            sub = str(self.config.options.get("directory", "."))
            base = raw_dir() / sub
        self.directory = base.resolve()

    # ------------------------------------------------------------------ API

    def supports(self, symbol: str, timeframe: str | Timeframe) -> bool:
        try:
            return self._locate(symbol, normalize_timeframe(timeframe)) is not None
        except ValueError:
            return False

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        start: datetime | pd.Timestamp | str | None = None,
        end: datetime | pd.Timestamp | str | None = None,
    ) -> MarketData:
        tf = normalize_timeframe(timeframe)
        path = self._locate(symbol, tf)
        if path is None:
            raise DataUnavailableError(
                f"No CSV found for {symbol} {tf.code}. Expected a file named "
                f"'{symbol.upper()}_{tf.code}.csv' in {self.directory}."
            )

        df = self._read(path, symbol)

        if start is not None:
            df = df[df.index >= to_utc(start)]
        if end is not None:
            df = df[df.index <= to_utc(end)]

        stat = path.stat()
        meta: dict[str, Any] = {
            "provider": self.name,
            "source_file": str(path),
            "source_bytes": stat.st_size,
            "source_mtime": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
            "timeframe": tf.code,
            "last_bar_complete": True,  # a static file only holds closed bars
            "retrieved_at": utcnow().isoformat(),
        }
        data = MarketData(symbol=symbol, timeframe=tf, df=df, provider=self.name, metadata=meta)
        logger.info("Loaded CSV market data", extra={"summary": data.describe(), "path": str(path)})
        return data

    def get_latest_data(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        bars: int = 500,
    ) -> MarketData:
        data = self.get_historical_data(symbol, timeframe)
        if len(data) > bars:
            data = data.replace(df=data.df.iloc[-bars:])
        return data

    # -------------------------------------------------------------- internals

    def _locate(self, symbol: str, tf: Timeframe) -> Path | None:
        """Find the file for a symbol/timeframe, tolerating naming variations."""
        if not self.directory.exists():
            return None
        sym = symbol.strip().upper()
        candidates = [
            f"{sym}_{tf.code}.csv",
            f"{sym}_{tf.code.lower()}.csv",
            f"{sym}-{tf.code}.csv",
            f"{sym}{tf.code}.csv",
        ]
        for name in candidates:
            path = self.directory / name
            if path.exists():
                return path
        # Case-insensitive sweep as a last resort.
        wanted = {c.lower() for c in candidates}
        for path in self.directory.glob("*.csv"):
            if path.name.lower() in wanted:
                return path
        return None

    def _read(self, path: Path, symbol: str) -> pd.DataFrame:
        """Parse a CSV into the canonical schema, sniffing dialect and decimals."""
        try:
            sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
        except OSError as exc:
            raise DataFetchError(f"Cannot read {path}: {exc}") from exc

        if not sample.strip():
            raise DataFetchError(f"{path} is empty")

        try:
            dialect = csv_module.Sniffer().sniff(sample, delimiters=",;\t|")
            sep = dialect.delimiter
        except csv_module.Error:
            sep = ","

        # European exports use ';' as separator and ',' as decimal mark.
        decimal = "," if sep == ";" and sample.count(",") > sample.count(".") else "."

        try:
            raw = pd.read_csv(
                path,
                sep=sep,
                decimal=decimal,
                encoding="utf-8-sig",
                skipinitialspace=True,
                # Exact float64 reconstruction; the default parser is off by
                # ~1e-14, which is enough to make a re-import of your own
                # exported data fail a checksum comparison.
                float_precision="round_trip",
            )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise DataFetchError(f"Failed to parse {path}: {exc}") from exc

        if raw.empty:
            raise DataFetchError(f"{path} contains a header but no rows")

        assume_tz = self._assume_tz(symbol)
        df = coerce_schema(raw, assume_tz=assume_tz)

        if df.empty:
            raise DataFetchError(
                f"{path} produced no usable rows after parsing - check the timestamp column format"
            )
        return df

    def _assume_tz(self, symbol: str) -> str:  # noqa: ARG002
        """Timezone to attach to naive CSV timestamps.

        Always UTC unless the operator explicitly says otherwise via
        ``providers.csv.options.assume_timezone``. Guessing the venue timezone
        here would silently shift every bar in a UTC-stamped export, and a
        wrong-by-hours dataset is far worse than one that is obviously wrong.
        """
        return str(self.config.options.get("assume_timezone", "UTC"))
