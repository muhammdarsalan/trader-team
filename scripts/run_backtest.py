"""Run a backtest and write a reproducible experiment record.

Examples:
    python scripts/run_backtest.py --symbol XAUUSD --timeframe 1D --start 2012-01-01
    python scripts/run_backtest.py --symbol XAUUSD --timeframe 1D --save
    python scripts/run_backtest.py --all --timeframe 1D --start 2015-01-01

Results are descriptive of one historical sample. They are not a forecast, and
a backtest is the weakest form of evidence that a strategy works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backtest.engine import Backtester  # noqa: E402
from app.config.loader import get_config, override_config  # noqa: E402
from app.data.service import MarketDataService  # noqa: E402
from app.data.validators.quality import DataQualityError  # noqa: E402
from app.utils.logging import setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a backtest over historical data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--symbol", "-s", help="Canonical symbol, e.g. XAUUSD")
    target.add_argument("--all", action="store_true", help="Every enabled asset")

    parser.add_argument("--timeframe", "-t", default="1D")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--save", action="store_true", help="Write the result to experiments/")
    parser.add_argument(
        "--max-bars", type=int, default=None, help="Smoke-test only; never a research result"
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser


def run_one(symbol: str, args: argparse.Namespace) -> int:
    config = get_config()
    service = MarketDataService()

    print(f"\n{'=' * 60}\n{symbol} {args.timeframe}\n{'=' * 60}")

    try:
        result_data = service.get_historical_data(
            symbol, args.timeframe, start=args.start, end=args.end
        )
    except DataQualityError as exc:
        print(f"Data rejected by the quality gate:\n{exc}")
        return 1

    if args.max_bars:
        config = override_config(
            config, backtest=config.backtest.model_copy(update={"max_bars": args.max_bars})
        )

    backtester = Backtester(
        config=config,
        asset=config.assets.assets.get(symbol.upper()),
        experiment_id=args.experiment_id,
    )
    result = backtester.run(result_data.data, quality_status=str(result_data.quality.status))

    print(result.render())

    if args.save:
        path = result.save()
        print(f"\nSaved to {path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level)

    config = get_config()
    symbols = config.assets.enabled_symbols() if args.all else [args.symbol]

    failures = sum(run_one(symbol, args) for symbol in symbols)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
