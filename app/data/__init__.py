"""Market-data subsystem.

Layout:
    schema.py      canonical OHLCV contract every downstream module relies on
    interfaces.py  provider abstraction (vendors are swappable)
    loaders/       concrete providers: yahoo, csv, synthetic
    validators/    data-quality engine
    processors/    normalisation and timeframe resampling
    cache.py       local parquet/csv cache with provenance manifests
    service.py     the facade the rest of the platform actually calls
"""

from app.data.schema import (
    OHLCV_COLUMNS,
    REQUIRED_COLUMNS,
    MarketData,
    SchemaError,
    empty_frame,
    validate_schema,
)

__all__ = [
    "OHLCV_COLUMNS",
    "REQUIRED_COLUMNS",
    "MarketData",
    "SchemaError",
    "empty_frame",
    "validate_schema",
]
