"""Download and cache historical market data.

Examples:
    python scripts/ingest_data.py --symbol XAUUSD --timeframe 1D
    python scripts/ingest_data.py --all --timeframe 1D --start 2010-01-01
    python scripts/ingest_data.py --symbol XAUUSD --timeframe 4H --provider yahoo
    python scripts/ingest_data.py --symbol XAUUSD --timeframe 1D --provider csv

Data is written to the local cache with a manifest recording source, range,
row count and a content checksum, so any later experiment can state exactly
which dataset produced it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.loader import get_config  # noqa: E402
from app.data.interfaces import DataProviderError  # noqa: E402
from app.data.loaders.registry import available_providers, get_provider  # noqa: E402
from app.data.service import MarketDataService  # noqa: E402
from app.data.validators.quality import DataQualityError, QualityStatus  # noqa: E402
from app.utils.logging import get_logger, setup_logging  # noqa: E402
from app.utils.paths import ensure_dir, reports_dir  # noqa: E402

logger = get_logger("ingest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and cache historical OHLCV data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--symbol", "-s", help="Canonical symbol, e.g. XAUUSD")
    target.add_argument("--all", action="store_true", help="Every enabled asset in assets.yaml")

    parser.add_argument("--timeframe", "-t", default="1D", help="Timeframe (default: 1D)")
    parser.add_argument("--start", help="Start date, YYYY-MM-DD (default: earliest available)")
    parser.add_argument("--end", help="End date, YYYY-MM-DD (default: latest available)")
    parser.add_argument(
        "--provider", "-p", help=f"Data provider. Available: {', '.join(available_providers())}"
    )
    parser.add_argument("--refresh", action="store_true", help="Bypass the cache and re-download")
    parser.add_argument(
        "--save-report", action="store_true", help="Write the quality report to reports/"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def ingest_one(service: MarketDataService, symbol: str, args: argparse.Namespace) -> int:
    """Fetch one symbol. Returns a process-style status: 0 ok, 1 failed, 2 warned."""
    print(f"\n{'=' * 64}\n{symbol} {args.timeframe}\n{'=' * 64}")

    try:
        result = service.get_historical_data(
            symbol, args.timeframe, start=args.start, end=args.end, refresh=args.refresh
        )
    except DataQualityError as exc:
        print(exc.report.render())
        print(f"\nFAILED: {symbol} did not pass validation and was not cached.")
        return 1
    except DataProviderError as exc:
        print(f"FAILED: {exc}")
        return 1

    print(result.quality.render())

    if result.cleaning.changed:
        print(f"\nCleaning: {result.cleaning.summary()}")
    if result.resampled_from:
        print(f"Resampled from {result.resampled_from} (provider has no native {args.timeframe})")

    if args.save_report:
        path = ensure_dir(reports_dir() / "data_quality") / f"{symbol}_{args.timeframe}.txt"
        path.write_text(result.quality.render(), encoding="utf-8")
        print(f"Report written to {path}")

    return 2 if result.quality.status is QualityStatus.WARNING else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level)

    config = get_config()
    provider_name = args.provider or config.data.default_provider

    try:
        provider = get_provider(
            provider_name, assets=config.assets, config=config.data.provider(provider_name)
        )
    except KeyError as exc:
        print(f"ERROR: {exc}")
        return 1

    service = MarketDataService(provider=provider)

    symbols = config.assets.enabled_symbols() if args.all else [args.symbol]
    statuses = {sym: ingest_one(service, sym, args) for sym in symbols}

    failed = [s for s, code in statuses.items() if code == 1]
    warned = [s for s, code in statuses.items() if code == 2]

    print(f"\n{'=' * 64}\nSUMMARY\n{'=' * 64}")
    print(f"Requested: {len(symbols)}   OK: {len(symbols) - len(failed)}   Failed: {len(failed)}")
    if warned:
        print(f"With warnings: {', '.join(warned)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
