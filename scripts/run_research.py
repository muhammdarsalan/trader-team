"""Run a validation study and write an interpretable report.

Examples:
    python scripts/run_research.py --symbol XAUUSD --timeframe 1D --start 2012-01-01
    python scripts/run_research.py --symbol XAUUSD --no-walk-forward --simulations 500
    python scripts/run_research.py --list-experiments

This is the part of the platform that tries to break a configuration rather
than to show it off. A study that concludes "this does not work" has succeeded;
that outcome is the expected one and costs far less than finding out later.

No result printed by this script is a forecast, and none of it is a claim of
profitability.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.loader import get_config, override_config  # noqa: E402
from app.data.service import MarketDataService  # noqa: E402
from app.data.validators.quality import DataQualityError  # noqa: E402
from app.research.experiments import ExperimentStore  # noqa: E402
from app.research.splits import SplitError  # noqa: E402
from app.research.study import ValidationStudy  # noqa: E402
from app.utils.logging import setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a configuration out of sample, walk forward and under resampling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbol", "-s", help="Canonical symbol, e.g. XAUUSD")
    parser.add_argument("--timeframe", "-t", default="1D")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default=None)

    parser.add_argument("--experiment-id", default=None,
                        help="Override the derived id. Rarely wanted: the derived id is "
                             "reproducible from the configuration and the data.")
    parser.add_argument("--no-walk-forward", action="store_true")
    parser.add_argument("--no-monte-carlo", action="store_true")
    parser.add_argument("--no-robustness", action="store_true")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--objective", default=None,
                        help="sortino, sharpe, expectancy_r, calmar or risk_adjusted")
    parser.add_argument("--no-store", action="store_true",
                        help="Skip the experiment database. The trial count then covers "
                             "this session only, which understates the search.")
    parser.add_argument("--list-experiments", action="store_true",
                        help="List recorded studies and exit")
    parser.add_argument("--log-level", default="WARNING")
    return parser


def apply_overrides(config, args):
    """Fold the command-line switches into the research configuration."""
    research = config.research
    walk_forward = research.walk_forward
    monte_carlo = research.monte_carlo
    robustness = research.robustness

    if args.no_walk_forward:
        walk_forward = walk_forward.model_copy(update={"enabled": False})
    if args.folds is not None:
        walk_forward = walk_forward.model_copy(update={"folds": args.folds})
    if args.no_monte_carlo:
        monte_carlo = monte_carlo.model_copy(update={"enabled": False})
    if args.simulations is not None:
        monte_carlo = monte_carlo.model_copy(update={"simulations": args.simulations})
    if args.no_robustness:
        robustness = robustness.model_copy(update={"enabled": False})

    update = {
        "walk_forward": walk_forward,
        "monte_carlo": monte_carlo,
        "robustness": robustness,
    }
    if args.objective:
        update["objective"] = args.objective

    return override_config(config, research=research.model_copy(update=update))


def list_experiments() -> int:
    store = ExperimentStore()
    rows = store.list_experiments()
    if not rows:
        print("No experiments recorded yet.")
        return 0

    print(f"{'experiment':<20}{'symbol':<10}{'verdict':<32}{'updated'}")
    print("-" * 88)
    for row in rows:
        print(
            f"{row['experiment_id']:<20}{row['symbol'] or '':<10}"
            f"{(row['verdict'] or '')[:30]:<32}{row['updated_at'][:19]}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level)

    if args.list_experiments:
        return list_experiments()

    if not args.symbol:
        print("--symbol is required (or use --list-experiments)")
        return 2

    config = apply_overrides(get_config(), args)
    service = MarketDataService()

    print(f"\n{'=' * 78}\nVALIDATION STUDY: {args.symbol} {args.timeframe}\n{'=' * 78}")

    try:
        requested = service.get_historical_data(
            args.symbol, args.timeframe, start=args.start, end=args.end
        )
    except DataQualityError as exc:
        print(f"Data rejected by the quality gate, so there is nothing to validate:\n{exc}")
        return 1

    store = None if args.no_store else ExperimentStore()

    study = ValidationStudy(
        config=config,
        data=requested.data,
        asset=config.assets.assets.get(args.symbol.upper()),
        quality_status=str(requested.quality.status),
        store=store,
        experiment_id=args.experiment_id,
    )

    try:
        result = study.run()
    except SplitError as exc:
        print(f"The history is too short for this study:\n{exc}")
        return 1

    print(result.report.render())
    print(
        f"\nExperiment {result.experiment_id} recorded. "
        f"{result.trials} distinct configuration(s) have now been evaluated against this data."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
