"""Run the analysis stack on real market data and explain the result.

Loads validated data, computes features, detects the market regime and runs
every enabled strategy, then prints what each one decided and why.

Examples:
    python scripts/analyze_market.py --symbol XAUUSD --timeframe 1D
    python scripts/analyze_market.py --symbol XAUUSD --timeframe 1D --history 20
    python scripts/analyze_market.py --all --timeframe 1D

This is analysis only. It places no orders, simulates no fills and computes no
performance figures - the backtester arrives in phase 3. Signals shown here are
what the strategies *would* propose, before any risk check.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.loader import get_config  # noqa: E402
from app.data.service import MarketDataService  # noqa: E402
from app.data.validators.quality import DataQualityError  # noqa: E402
from app.features.engine import FeatureEngine  # noqa: E402
from app.regimes.detector import RegimeDetector  # noqa: E402
from app.strategies import build_enabled_strategies  # noqa: E402
from app.utils.logging import setup_logging  # noqa: E402

RULE = "=" * 72
THIN = "-" * 72


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse a market: features, regime and strategy signals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--symbol", "-s", help="Canonical symbol, e.g. XAUUSD")
    target.add_argument("--all", action="store_true", help="Every enabled asset")

    parser.add_argument("--timeframe", "-t", default="1D")
    parser.add_argument("--start", default="2012-01-01", help="History start (default: 2012-01-01)")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--history", type=int, default=0,
        help="Also summarise signals over the last N bars, not just the latest",
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser


def analyse(symbol: str, args: argparse.Namespace) -> int:
    config = get_config()
    service = MarketDataService()

    print(f"\n{RULE}\n{symbol} {args.timeframe}\n{RULE}")

    try:
        result = service.get_historical_data(
            symbol, args.timeframe, start=args.start, end=args.end
        )
    except DataQualityError as exc:
        print(f"Data rejected by the quality gate:\n{exc}")
        return 1

    data = result.data
    asset = config.assets.assets.get(symbol.upper())

    print(f"Data:      {data.describe()}")
    print(f"Quality:   {result.quality.status}")

    features = FeatureEngine(config.features).compute(data, asset)
    print(f"Features:  {features.describe()}")
    if features.suppressed:
        print(f"           suppressed: {', '.join(features.suppressed)}")

    if len(features.df) <= features.warmup_bars:
        print("\nNot enough history past the warm-up period to analyse.")
        return 1

    detector = RegimeDetector(config.regimes)
    strategies = build_enabled_strategies(config.strategies)

    timestamp = features.df.index[-1]
    regime = detector.detect(features, timestamp)

    print(f"\n{THIN}\nMARKET REGIME at {timestamp:%Y-%m-%d}\n{THIN}")
    print(regime.describe())

    print(f"\n{THIN}\nSTRATEGY SIGNALS\n{THIN}")
    directions = Counter()
    for strategy in strategies:
        signal = strategy.generate_signal(data, features, regime, timestamp)
        directions[str(signal.direction)] += 1

        print(f"\n{strategy.name}")
        print(f"  Decision:   {signal.direction}", end="")
        if signal.is_actionable:
            print(f"  (confidence {signal.confidence:.2f})")
            print(f"  Entry:      {signal.entry_price:.5g}")
            print(f"  Stop:       {signal.stop_loss:.5g}")
            if signal.take_profit is not None:
                print(f"  Target:     {signal.take_profit:.5g}")
            if signal.reward_risk_ratio is not None:
                print(f"  R:R:        {signal.reward_risk_ratio:.2f}")
        else:
            suppressed = signal.metadata.get("suppressed_direction")
            print(f"  (would have been {suppressed}, held back)" if suppressed else "")
        for reason in signal.reasoning:
            print(f"    - {reason}")

    print(f"\n{THIN}")
    summary = ", ".join(f"{count} {name}" for name, count in sorted(directions.items()))
    print(f"Tally: {summary}")
    print(
        "No aggregation, risk check or position sizing has been applied - "
        "those arrive in phase 3."
    )

    if args.history:
        _signal_history(data, features, detector, strategies, args.history)

    return 0


def _signal_history(data, features, detector, strategies, bars: int) -> None:
    """How often each strategy proposed a trade over the last N bars."""
    usable = features.df.index[features.warmup_bars :]
    window = usable[-bars:] if len(usable) > bars else usable

    print(f"\n{THIN}\nSIGNAL HISTORY over the last {len(window)} bars\n{THIN}")
    print(f"{'strategy':<22}{'LONG':>6}{'SHORT':>7}{'WAIT':>7}")

    for strategy in strategies:
        counts = Counter()
        for timestamp in window:
            regime = detector.detect(features, timestamp)
            signal = strategy.generate_signal(data, features, regime, timestamp)
            counts[str(signal.direction)] += 1
        print(
            f"{strategy.name:<22}{counts['LONG']:>6}{counts['SHORT']:>7}{counts['WAIT']:>7}"
        )

    regimes = Counter(str(detector.detect(features, ts).regime) for ts in window)
    print("\nRegimes over the same window:")
    for name, count in regimes.most_common():
        print(f"  {name:<18}{count:>5}  ({count / len(window):.0%})")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level)

    config = get_config()
    symbols = config.assets.enabled_symbols() if args.all else [args.symbol]

    failures = sum(analyse(symbol, args) for symbol in symbols)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
