"""Local market-data cache with provenance manifests.

Two jobs:

1. Stop re-downloading the same decade of bars on every run.
2. Record *exactly* what was downloaded, when, from where, and a checksum of
   the contents - so an experiment can later state which dataset produced it.

Cached payloads are gitignored (they are large and regenerable); manifests are
small JSON files that can be committed if you want dataset versions pinned in
history.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.data.schema import MarketData, coerce_schema, validate_schema
from app.utils.logging import get_logger
from app.utils.paths import cache_dir, ensure_dir
from app.utils.timeutils import Timeframe, normalize_timeframe, utcnow

logger = get_logger(__name__)

MANIFEST_VERSION = 1


@dataclass
class CacheManifest:
    """Provenance record for one cached series."""

    symbol: str
    timeframe: str
    provider: str
    rows: int
    start: str | None
    end: str | None
    checksum: str
    written_at: str
    payload_file: str
    payload_format: str
    source_metadata: dict[str, Any] = field(default_factory=dict)
    manifest_version: int = MANIFEST_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheManifest:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def written_at_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.written_at)

    def age_hours(self) -> float:
        return (utcnow() - self.written_at_ts).total_seconds() / 3600


def frame_checksum(df: pd.DataFrame) -> str:
    """Content hash of a frame: same bars in, same hash out.

    Uses the index and OHLCV values only, so incidental differences (column
    order, dtype backend, metadata) do not change the identity of the data.
    """
    hasher = hashlib.sha256()
    hasher.update(df.index.asi8.tobytes())
    for col in sorted(df.columns):
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
        hasher.update(col.encode())
        hasher.update(values.tobytes())
    return hasher.hexdigest()[:32]


class MarketDataCache:
    """Read/write canonical OHLCV frames on local disk."""

    def __init__(
        self,
        root: Path | None = None,
        fmt: str = "parquet",
    ) -> None:
        if fmt not in {"parquet", "csv"}:
            raise ValueError(f"Unsupported cache format {fmt!r}; use 'parquet' or 'csv'")
        self.root = Path(root) if root is not None else cache_dir()
        self.fmt = fmt

    # ------------------------------------------------------------------ paths

    def _dir(self, provider: str) -> Path:
        return ensure_dir(self.root / provider.lower())

    def _stem(self, symbol: str, tf: Timeframe) -> str:
        return f"{symbol.strip().upper()}_{tf.code}"

    def payload_path(self, symbol: str, timeframe: str | Timeframe, provider: str) -> Path:
        tf = normalize_timeframe(timeframe)
        return self._dir(provider) / f"{self._stem(symbol, tf)}.{self.fmt}"

    def manifest_path(self, symbol: str, timeframe: str | Timeframe, provider: str) -> Path:
        tf = normalize_timeframe(timeframe)
        return self._dir(provider) / f"{self._stem(symbol, tf)}.manifest.json"

    # ------------------------------------------------------------------- read

    def read_manifest(
        self, symbol: str, timeframe: str | Timeframe, provider: str
    ) -> CacheManifest | None:
        path = self.manifest_path(symbol, timeframe, provider)
        if not path.exists():
            return None
        try:
            return CacheManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Unreadable cache manifest", extra={"path": str(path), "error": str(exc)})
            return None

    def load(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        provider: str,
        *,
        max_age_hours: float | None = None,
        verify_checksum: bool = True,
    ) -> MarketData | None:
        """Return the cached series, or None on miss/expiry/corruption."""
        tf = normalize_timeframe(timeframe)
        manifest = self.read_manifest(symbol, tf, provider)
        payload = self.payload_path(symbol, tf, provider)

        if manifest is None or not payload.exists():
            return None

        if max_age_hours is not None and manifest.age_hours() > max_age_hours:
            logger.debug(
                "Cache expired",
                extra={"symbol": symbol, "timeframe": tf.code, "age_h": manifest.age_hours()},
            )
            return None

        try:
            df = self._read_payload(payload)
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable cache payload", extra={"path": str(payload), "error": str(exc)})
            return None

        if verify_checksum:
            actual = frame_checksum(df)
            if actual != manifest.checksum:
                logger.warning(
                    "Cache checksum mismatch - treating as a miss",
                    extra={"symbol": symbol, "expected": manifest.checksum, "actual": actual},
                )
                return None

        return MarketData(
            symbol=symbol,
            timeframe=tf,
            df=df,
            provider=provider,
            metadata={**manifest.source_metadata, "from_cache": True,
                      "cache_written_at": manifest.written_at, "checksum": manifest.checksum},
        )

    def _read_payload(self, path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            # pandas' default CSV float parser is fast but inexact (~1e-14).
            # "round_trip" reproduces the written float64 bit for bit, which is
            # what makes the checksum a real integrity check rather than noise.
            df = pd.read_csv(path, float_precision="round_trip")
        df = coerce_schema(df)
        validate_schema(df, strict=False)
        return df

    # ------------------------------------------------------------------ write

    def save(self, data: MarketData) -> CacheManifest:
        """Persist a series plus its manifest."""
        validate_schema(data.df, strict=False)
        tf = data.timeframe
        payload = self.payload_path(data.symbol, tf, data.provider)

        self._write_payload(data.df, payload)

        manifest = CacheManifest(
            symbol=data.symbol,
            timeframe=tf.code,
            provider=data.provider,
            rows=len(data.df),
            start=data.start.isoformat() if data.start is not None else None,
            end=data.end.isoformat() if data.end is not None else None,
            checksum=frame_checksum(data.df),
            written_at=utcnow().isoformat(),
            payload_file=payload.name,
            payload_format=self.fmt,
            source_metadata=dict(data.metadata),
        )
        self.manifest_path(data.symbol, tf, data.provider).write_text(
            manifest.to_json(), encoding="utf-8"
        )
        logger.info(
            "Cached market data",
            extra={"symbol": data.symbol, "timeframe": tf.code, "rows": len(data.df),
                   "path": str(payload)},
        )
        return manifest

    def _write_payload(self, df: pd.DataFrame, path: Path) -> None:
        ensure_dir(path.parent)
        # Write to a temporary file and swap, so an interrupted run cannot leave
        # a half-written payload that a later run would silently trust.
        tmp = path.with_suffix(path.suffix + ".tmp")
        if self.fmt == "parquet":
            try:
                df.to_parquet(tmp, index=True)
            except (ImportError, ValueError) as exc:
                raise RuntimeError(
                    f"Parquet write failed ({exc}). Install pyarrow, or set "
                    "cache_format: csv in configs/data.yaml."
                ) from exc
        else:
            # %.17g is the shortest format that round-trips a float64 exactly.
            # pandas' default repr loses ~1e-14, which is harmless for analysis
            # but breaks the checksum that proves the cache was not tampered
            # with - and a cache whose integrity check cries wolf gets ignored.
            df.to_csv(tmp, index=True, float_format="%.17g")
        tmp.replace(path)

    # ------------------------------------------------------------- incremental

    def merge(self, cached: MarketData, fresh: MarketData) -> MarketData:
        """Combine cached and freshly-fetched bars.

        Fresh bars win on overlap: a vendor revising a recent bar (common near
        the end of a session) should not be overridden by a stale cached copy.
        """
        if cached.symbol != fresh.symbol or cached.timeframe.code != fresh.timeframe.code:
            raise ValueError(
                f"Cannot merge {cached.symbol}/{cached.timeframe} with "
                f"{fresh.symbol}/{fresh.timeframe}"
            )
        if cached.df.empty:
            return fresh
        if fresh.df.empty:
            return cached

        combined = pd.concat([cached.df, fresh.df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        validate_schema(combined, strict=False)

        return fresh.replace(
            df=combined,
            metadata={**cached.metadata, **fresh.metadata, "merged_from_cache": True},
        )

    # ----------------------------------------------------------------- admin

    def clear(self, symbol: str | None = None, provider: str | None = None) -> int:
        """Delete cached entries. Returns the number of files removed."""
        if not self.root.exists():
            return 0
        pattern = f"{symbol.strip().upper()}_*" if symbol else "*"
        base = self._dir(provider) if provider else self.root
        removed = 0
        for path in base.rglob(pattern):
            if path.is_file() and path.suffix in {".parquet", ".csv", ".json"}:
                path.unlink()
                removed += 1
        return removed

    def entries(self) -> list[CacheManifest]:
        """Every manifest currently on disk."""
        if not self.root.exists():
            return []
        out = []
        for path in sorted(self.root.rglob("*.manifest.json")):
            try:
                out.append(CacheManifest.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        return out
