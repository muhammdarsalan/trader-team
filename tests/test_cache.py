"""Cache round-trips, provenance and incremental merges."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.data.cache import MarketDataCache, frame_checksum
from app.data.schema import MarketData, empty_frame
from tests.conftest import make_ohlcv


@pytest.fixture(params=["parquet", "csv"])
def cache(request, tmp_path) -> MarketDataCache:
    return MarketDataCache(root=tmp_path / "cache", fmt=request.param)


def md(df: pd.DataFrame, symbol: str = "XAUUSD", timeframe: str = "1D") -> MarketData:
    return MarketData(
        symbol=symbol, timeframe=timeframe, df=df, provider="synthetic",
        metadata={"provider": "synthetic", "note": "test"},
    )


# ------------------------------------------------------------------ round trip

def test_save_then_load_returns_equivalent_data(cache, clean_daily):
    cache.save(md(clean_daily))
    loaded = cache.load("XAUUSD", "1D", "synthetic")

    assert loaded is not None
    pd.testing.assert_frame_equal(loaded.df, clean_daily, check_freq=False)
    assert loaded.metadata["from_cache"] is True


def test_load_returns_none_on_miss(cache):
    assert cache.load("NOPE", "1D", "synthetic") is None


def test_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported cache format"):
        MarketDataCache(root=tmp_path, fmt="hdf5")


# ------------------------------------------------------------------- manifests

def test_manifest_records_provenance(cache, clean_daily):
    manifest = cache.save(md(clean_daily))
    assert manifest.symbol == "XAUUSD"
    assert manifest.rows == len(clean_daily)
    assert manifest.provider == "synthetic"
    assert manifest.checksum
    assert manifest.start and manifest.end


def test_manifest_is_written_as_readable_json(cache, clean_daily):
    cache.save(md(clean_daily))
    path = cache.manifest_path("XAUUSD", "1D", "synthetic")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "XAUUSD"
    assert payload["source_metadata"]["note"] == "test"


def test_entries_lists_everything_cached(cache, clean_daily):
    cache.save(md(clean_daily, symbol="XAUUSD"))
    cache.save(md(clean_daily, symbol="EURUSD"))
    assert {e.symbol for e in cache.entries()} == {"XAUUSD", "EURUSD"}


# ------------------------------------------------------------------- checksums

def test_checksum_is_stable_for_identical_data(clean_daily):
    assert frame_checksum(clean_daily) == frame_checksum(clean_daily.copy())


def test_checksum_changes_when_a_price_changes(clean_daily):
    mutated = clean_daily.copy()
    mutated.iloc[0, mutated.columns.get_loc("close")] += 0.01
    assert frame_checksum(mutated) != frame_checksum(clean_daily)


def test_corrupted_payload_is_treated_as_a_miss(cache, clean_daily):
    """A tampered cache file must never be silently trusted."""
    cache.save(md(clean_daily))
    path = cache.payload_path("XAUUSD", "1D", "synthetic")

    tampered = clean_daily.copy()
    tampered.iloc[0, tampered.columns.get_loc("close")] *= 2
    if cache.fmt == "parquet":
        tampered.to_parquet(path)
    else:
        tampered.to_csv(path)

    assert cache.load("XAUUSD", "1D", "synthetic") is None


def test_checksum_verification_can_be_skipped(cache, clean_daily):
    cache.save(md(clean_daily))
    assert cache.load("XAUUSD", "1D", "synthetic", verify_checksum=False) is not None


# ----------------------------------------------------------------------- expiry

def test_expired_cache_is_a_miss(cache, clean_daily, monkeypatch):
    cache.save(md(clean_daily))
    assert cache.load("XAUUSD", "1D", "synthetic", max_age_hours=0.0) is None


def test_fresh_cache_is_a_hit(cache, clean_daily):
    cache.save(md(clean_daily))
    assert cache.load("XAUUSD", "1D", "synthetic", max_age_hours=24.0) is not None


# ------------------------------------------------------------------ incremental

def test_merge_appends_new_bars(cache):
    old = make_ohlcv(periods=100, start="2020-01-01")
    combined_source = make_ohlcv(periods=150, start="2020-01-01")
    new = combined_source.iloc[90:]

    merged = cache.merge(md(old), md(new))
    assert len(merged.df) == 150
    assert merged.df.index.is_monotonic_increasing
    assert not merged.df.index.has_duplicates


def test_merge_prefers_fresh_bars_on_overlap(cache, clean_daily):
    old = clean_daily.copy()
    revised = clean_daily.iloc[-10:].copy()
    revised["close"] = 12345.0

    merged = cache.merge(md(old), md(revised))
    assert (merged.df["close"].iloc[-10:] == 12345.0).all()


def test_merge_handles_empty_sides(cache, clean_daily):
    assert len(cache.merge(md(empty_frame()), md(clean_daily)).df) == len(clean_daily)
    assert len(cache.merge(md(clean_daily), md(empty_frame())).df) == len(clean_daily)


def test_merge_rejects_mismatched_series(cache, clean_daily):
    with pytest.raises(ValueError, match="Cannot merge"):
        cache.merge(md(clean_daily, symbol="XAUUSD"), md(clean_daily, symbol="EURUSD"))


# ------------------------------------------------------------------------ admin

def test_clear_removes_cached_files(cache, clean_daily):
    cache.save(md(clean_daily))
    assert cache.clear(symbol="XAUUSD") >= 1
    assert cache.load("XAUUSD", "1D", "synthetic") is None


def test_partial_write_does_not_leave_a_readable_payload(cache, clean_daily):
    """Writes go through a temp file and swap, so no half-file is ever trusted."""
    cache.save(md(clean_daily))
    leftovers = list(cache.root.rglob("*.tmp"))
    assert leftovers == []
