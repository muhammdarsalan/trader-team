"""Run a paper-trading session.

Two sources of bars, one code path:

    # deterministic replay over cached history (no network needed once cached)
    python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --start 2020-01-01

    # fetch the newest bars and process whatever has not been seen
    python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --live

    # replay your own file: drop data/raw/XAUUSD_1D.csv, then
    python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --provider csv

    # look at a session without advancing it
    python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --status

The only difference between replay and live is where the frame comes from. Both
call the same :meth:`PaperTradingEngine.process_bar`, so a behaviour that holds
in one holds in the other.

State persists to data/paper/<SYMBOL>_<TIMEFRAME>.json and is reloaded on the
next run, so a session survives a restart and re-running overlapping bars is a
no-op rather than a duplicate trade.

This is simulation. No broker is contacted, no order leaves the process, and
there is no flag here or anywhere else that makes it place a real trade.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.loader import get_config  # noqa: E402
from app.data.loaders.registry import available_providers, get_provider  # noqa: E402
from app.data.service import MarketDataService  # noqa: E402
from app.data.validators.quality import DataQualityError  # noqa: E402
from app.paper_trading.engine import PaperTradingEngine  # noqa: E402
from app.utils.logging import setup_logging  # noqa: E402
from app.utils.paths import paper_state_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a paper-trading session (simulation only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbol", "-s", default="XAUUSD", help="Canonical symbol")
    parser.add_argument("--timeframe", "-t", default="1D")

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--replay",
        action="store_true",
        help="Replay cached/historical bars deterministically (default)",
    )
    source.add_argument(
        "--live",
        action="store_true",
        help="Fetch the newest bars from the configured provider and process them",
    )
    source.add_argument(
        "--status",
        action="store_true",
        help="Print the saved session state and exit without processing any bar",
    )

    parser.add_argument(
        "--provider",
        "-p",
        default=None,
        help="Data provider for this run. Available: "
        f"{', '.join(available_providers())}. Defaults to data.default_provider. "
        "Use 'csv' to replay your own data/raw/<SYMBOL>_<TIMEFRAME>.csv.",
    )

    parser.add_argument("--start", default="2020-01-01", help="Replay start date")
    parser.add_argument("--end", default=None, help="Replay end date")
    parser.add_argument(
        "--bars", type=int, default=500, help="Bars to request in --live mode"
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Override the state file path (defaults to data/paper/<SYMBOL>_<TF>.json)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the quality gate on a live fetch. The resulting grade is "
        "recorded as unstated, which the risk engine treats as a refusal.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Start from a flat book, archiving any existing state file",
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser


def _archive_existing(path: Path) -> Path | None:
    """Move an existing state file aside rather than deleting it.

    A paper session is research output. `--reset` should let you start over
    without destroying the record of what the previous run decided.
    """
    if not path.exists():
        return None
    index = 1
    while True:
        target = path.with_suffix(f"{path.suffix}.bak{index}")
        if not target.exists():
            path.rename(target)
            return target
        index += 1


def _print_status(engine: PaperTradingEngine) -> None:
    market = engine.market_summary()
    portfolio = engine.portfolio_snapshot()
    health = engine.system_health()
    state = engine.state

    print(f"\n{'=' * 64}")
    print(f"{engine.symbol} {engine.timeframe} — paper trading status")
    print(f"{'=' * 64}")

    if market.get("is_proxy"):
        print(f"! PROXY DATA: {market.get('data_caveat')}")

    freshness = market.get("freshness") or {}
    print(
        f"Market      : status={market.get('status')} bars={market.get('rows')} "
        f"source={market.get('source')}"
    )
    print(
        f"              last bar={market.get('last_bar')} "
        f"close={market.get('last_close')}"
    )
    print(
        f"              quality={market.get('quality_status')} "
        f"freshness={freshness.get('status')} ({freshness.get('detail')})"
    )
    print(
        f"Bars        : processed={state.get('bars_processed')} "
        f"decided={state.get('bars_decided')}"
    )
    print(f"Regime      : {state.get('last_regime')}")
    print(f"Decision    : {state.get('latest_decision')}")
    print(
        f"Portfolio   : cash={portfolio.get('cash'):.2f} "
        f"equity={portfolio.get('equity'):.2f} "
        f"unrealised={portfolio.get('unrealised_pnl'):.2f} "
        f"realised={portfolio.get('realised_pnl'):.2f}"
    )
    print(
        f"              drawdown={portfolio.get('drawdown'):.2%} "
        f"open={portfolio.get('open_positions')} "
        f"marked={portfolio.get('marked_to_market')}"
    )
    print(
        f"Trades      : closed={len(state.get('closed_trades') or [])} "
        f"fills={len(state.get('fills') or [])} "
        f"pending orders={len(state.get('pending_orders') or [])}"
    )
    performance = engine.performance_view()
    overall = performance.get("overall") or {}
    print(f"Realised    : {overall.get('verdict')}")
    for row in (performance.get("by_contributor") or [])[:5]:
        expectancy = row.get("expectancy_r")
        print(
            f"              {row['strategy']}: {row['trades']} trade(s), "
            + ("no defined R" if expectancy is None else f"mean {expectancy:+.2f}R")
            + ("  [sample only]" if row.get("insufficient_evidence") else "")
        )
    print(f"Health      : {health.get('status')}")
    for reason in health.get("reasons") or []:
        print(f"              - {reason}")

    errors = list(state.get("recent_errors") or [])[-5:]
    if errors:
        print("Errors      :")
        for error in errors:
            print(f"              - [{error.get('node')}] {error.get('error')}")

    print(f"\nState file  : {engine.state_path}")
    print("Live trading: DISABLED (simulation only, no broker integration exists)")


def build_service(args: argparse.Namespace) -> MarketDataService | None:
    """The data service for this run, or None to use the configured default.

    A paper session must be runnable on the operator's own bars: the Yahoo feeds
    are free but caveated, and "drop a CSV in data/raw and point the run at it"
    is the documented escape hatch. Without ``--provider`` the paper runner was
    the one entry point that could not take it.

    Returning None rather than a default service keeps the engine's lazy
    construction intact - ``--status`` must not reach for provider configuration
    it never uses.
    """
    if not args.provider:
        return None
    config = get_config()
    provider = get_provider(
        args.provider,
        assets=config.assets,
        config=config.data.provider(args.provider),
    )
    return MarketDataService(provider=provider)


def run_status(engine: PaperTradingEngine) -> int:
    """Report the saved session without touching it."""
    if engine.state_path is None or not engine.state_path.exists():
        print(
            f"No saved session at {engine.state_path}. Start one with --replay or --live; "
            "nothing is reported for a session that has not run."
        )
        return 1
    _print_status(engine)
    return 0


def run_replay(engine: PaperTradingEngine, args: argparse.Namespace) -> int:
    # The engine's own service, so replay and live read the same source and a
    # --provider override cannot apply to only one of them.
    try:
        result = engine.market_data_service.get_historical_data(
            args.symbol, args.timeframe, start=args.start, end=args.end
        )
    except DataQualityError as exc:
        print(f"Data rejected by the quality gate; nothing was replayed:\n{exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - report the real cause and stop
        print(f"Market data unavailable: {type(exc).__name__}: {exc}")
        return 1

    quality = str(result.quality.status)
    print(
        f"Replaying {len(result.data)} bars of {args.symbol} {args.timeframe} "
        f"(quality={quality})"
    )
    # The full report, not just the grade: the dashboard shows the gate's
    # findings, and the grade alone cannot say what was wrong with the feed.
    outputs = engine.catch_up(result.data, data_quality=result.quality)
    print(f"Processed {len(outputs)} new bar(s).")
    _print_status(engine)
    return 0


def run_live(engine: PaperTradingEngine, args: argparse.Namespace) -> int:
    print(f"Fetching the newest {args.bars} bars of {args.symbol} {args.timeframe}...")
    outcome = engine.run_live_tick(bars=args.bars, validate=not args.no_validate)
    print(f"Tick status: {outcome['status']}")
    if outcome["status"] == "failed":
        print(f"  {outcome['reason']}")
    else:
        print(f"Processed {outcome.get('bars_processed', 0)} new bar(s).")
    _print_status(engine)
    # A failed refresh is a real failure, but the session state is intact, so the
    # exit code reflects the fetch rather than the engine.
    return 1 if outcome["status"] == "failed" else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level)

    config = get_config()
    symbol = args.symbol.upper()

    if symbol not in config.assets.assets:
        known = ", ".join(sorted(config.assets.assets))
        print(f"Unknown symbol {symbol!r}. Configured assets: {known}")
        return 1

    if args.provider and args.provider not in available_providers():
        known = ", ".join(available_providers())
        print(f"Unknown provider {args.provider!r}. Available: {known}")
        return 1

    state_path = (
        Path(args.state) if args.state else paper_state_path(symbol, args.timeframe)
    )
    # --status reports; it never mutates. Archiving the file it was asked to
    # read would leave nothing to report and destroy the session in passing.
    if args.reset and not args.status:
        archived = _archive_existing(state_path)
        if archived is not None:
            print(f"Previous state archived to {archived}")

    engine = PaperTradingEngine(
        config=config,
        symbol=symbol,
        timeframe=args.timeframe,
        state_path=state_path,
        market_data_service=build_service(args),
    )

    if not engine.trading_enabled and not args.status:
        print(
            "Note: platform.trading_enabled is false, so no paper orders will be "
            "created. Signals, regimes and risk decisions are still recorded."
        )

    if args.status:
        return run_status(engine)
    return run_live(engine, args) if args.live else run_replay(engine, args)


if __name__ == "__main__":
    raise SystemExit(main())
