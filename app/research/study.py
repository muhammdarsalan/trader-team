"""The validation study.

One entry point that runs every analysis in this package in the order that
keeps them honest, and records what it did:

    1.  Split the history into in-sample, validation and out-of-sample windows.
    2.  Run the baseline on in-sample and validation.
    3.  Sweep parameter neighbourhoods - on the in-sample window, because a
        sweep is a search and a search consumes whatever it touches.
    4.  Measure strategy correlation and turn any redundancy into candidate
        configurations. Nothing is removed.
    5.  Walk forward, choosing between candidates on each fold's training half
        only.
    6.  Run the out-of-sample window. Once. Last.
    7.  Resample the out-of-sample trade sequence.
    8.  Count how much searching produced all of this, and assess it.
    9.  Write the report and record everything.

The ordering is the design. The out-of-sample window is touched at step 6 and
never before, because every earlier step is a decision, and a decision informed
by the final window would make that window in-sample without anybody noticing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.backtest.results import git_revision
from app.config.loader import AppConfig, override_config
from app.config.models import AssetConfig
from app.data.cache import frame_checksum
from app.data.schema import MarketData
from app.research.correlation import (
    analyse_correlations,
    summarise_variant_comparison,
    variant_row,
    weight_variants,
)
from app.research.experiments import (
    ExperimentRecord,
    ExperimentStore,
    config_fingerprint,
    reproducible_experiment_id,
    stable_hash,
)
from app.research.feedback import build_recommendations
from app.research.harness import ResearchHarness, SegmentRun
from app.research.monte_carlo import monte_carlo_trade_sequence
from app.research.objectives import get_objective
from app.research.overfitting import assess, bar_returns
from app.research.report import ValidationReport
from app.research.robustness import run_robustness_study
from app.research.splits import (
    Segment,
    SplitError,
    chronological_split,
    suggest_walk_forward_geometry,
    walk_forward_folds,
)
from app.research.walk_forward import WalkForwardAnalysis
from app.utils.logging import get_logger
from app.utils.timeutils import utcnow

logger = get_logger(__name__)


@dataclass
class StudyResult:
    """What a study produced, and where it was recorded."""

    report: ValidationReport
    experiment_id: str
    store: ExperimentStore | None = None
    trials: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


class ValidationStudy:
    """Runs the whole validation pipeline over one instrument."""

    def __init__(
        self,
        config: AppConfig,
        data: MarketData,
        asset: AssetConfig | None = None,
        *,
        quality_status: str | None = None,
        store: ExperimentStore | None = None,
        experiment_id: str | None = None,
    ) -> None:
        self.config = config
        self.data = data
        self.asset = asset
        self.quality_status = quality_status
        self.settings = config.research
        self.objective = get_objective(self.settings.objective)

        self.harness = ResearchHarness(
            config, data, asset, trading_enabled=True, quality_status=quality_status
        )
        self.store = store
        self.data_checksum = frame_checksum(data.df)
        self.git_revision = git_revision()

        self.spec = self._spec()
        self.experiment_id = experiment_id or reproducible_experiment_id(
            config_snapshot=self._config_snapshot(),
            data_checksum=self.data_checksum,
            git_revision=self.git_revision,
            random_seed=config.platform.random_seed,
            spec=self.spec,
        )
        #: Configurations evaluated during *this* study. The overfitting
        #: arithmetic uses the store's running total instead, which includes
        #: every earlier session.
        self._session_trials: set[str] = set()

    # ------------------------------------------------------------------- spec

    def _spec(self) -> dict[str, Any]:
        s = self.settings
        return {
            # Which configuration this study describes. Recorded in the report
            # itself, not only in the experiment store, so a surface reading
            # validation_report.json can tell whether the study still describes
            # the system it is being shown next to.
            "config_fingerprint": config_fingerprint(self._config_snapshot()),
            "split_fractions": list(s.split_fractions),
            "embargo_bars": s.embargo_bars,
            "objective": s.objective,
            "walk_forward": s.walk_forward.model_dump(),
            "monte_carlo": s.monte_carlo.model_dump(),
            "robustness": s.robustness.model_dump(),
            "correlation_threshold": s.correlation_threshold,
            "test_weight_variants": s.test_weight_variants,
        }

    def _config_snapshot(self) -> dict[str, Any]:
        return {
            "features": self.config.features.model_dump(),
            "regimes": self.config.regimes.model_dump(),
            "strategies": self.config.strategies.model_dump(),
            "risk": self.config.risk.model_dump(),
            "execution": self.config.execution.model_dump(),
            "backtest": self.config.backtest.model_dump(),
        }

    # -------------------------------------------------------------------- run

    def run(self) -> StudyResult:
        """Execute the study end to end."""
        warmup = self.harness.required_warmup()
        segments = self._split(warmup)
        by_name = {s.name: s for s in segments}

        # Register before anything runs. Trials reference this row, and a study
        # that crashes half way through should still leave a record of what was
        # attempted rather than vanishing.
        self._register(verdict="IN PROGRESS")

        report = ValidationReport(
            experiment_id=self.experiment_id,
            symbol=self.data.symbol,
            timeframe=self.data.timeframe.code,
            period_start=str(self.data.start),
            period_end=str(self.data.end),
            bars=len(self.data),
            data_provider=self.data.provider,
            data_checksum=self.data_checksum,
            data_quality=(
                f"{self.quality_status} across the series; each window graded separately"
                if self.quality_status
                else "graded per window"
            ),
            git_revision=self.git_revision,
            random_seed=self.config.platform.random_seed,
            created_at=utcnow().isoformat(),
            spec=self.spec,
        )

        # --- 2. development windows -----------------------------------------
        in_sample = self._run_segment(by_name["in_sample"])
        validation = self._run_segment(by_name["validation"])
        report.segments.extend([in_sample, validation])

        # --- 3. parameter sensitivity, on in-sample data only ----------------
        if self.settings.robustness.enabled and self.settings.robustness.parameters:
            report.robustness = run_robustness_study(
                self.harness,
                by_name["in_sample"],
                list(self.settings.robustness.parameters),
                objective=self.objective,
                objective_name=self.settings.objective,
                span=self.settings.robustness.span,
                points=self.settings.robustness.points,
                on_trial=lambda variant, cfg, metrics: self._count_trial(
                    variant, cfg, "in_sample", metrics
                ),
            )

        # --- 4. correlation and the variants it suggests ---------------------
        isolated = self._isolate_strategies(by_name["in_sample"])
        report.correlation = analyse_correlations(
            in_sample.result.trades,
            in_sample.result.decisions,
            segment="in_sample",
            threshold=self.settings.correlation_threshold,
            isolated_equity={name: run.equity_curve for name, run in isolated.items()},
            isolated_trades={
                name: run.metrics.total_trades for name, run in isolated.items()
            },
        )
        candidates = self._candidates(report)

        # --- 5. walk forward --------------------------------------------------
        if self.settings.walk_forward.enabled:
            report.walk_forward = self._walk_forward(warmup, candidates)

        # --- 6. the out-of-sample window, once --------------------------------
        out_of_sample = self._run_segment(by_name["out_of_sample"])
        report.segments.append(out_of_sample)

        # --- the weight variants, on data that did not suggest them -----------
        if candidates and self.settings.test_weight_variants:
            report.variant_rows, report.variant_summary = self._compare_variants(
                by_name["validation"], candidates
            )

        # --- 7. Monte Carlo on the out-of-sample trades ------------------------
        if self.settings.monte_carlo.enabled:
            mc = self.settings.monte_carlo
            report.monte_carlo = monte_carlo_trade_sequence(
                out_of_sample.trades,
                initial_equity=self.config.backtest.initial_balance,
                method=mc.method,
                simulations=mc.simulations,
                block_size=mc.block_size,
                seed=self.config.platform.random_seed,
                drawdown_limit=self.config.risk.max_drawdown_limit,
            )

        # --- regime performance -------------------------------------------------
        report.regime_performance = self._regime_table(report.segments)

        # --- 8. how much searching produced this --------------------------------
        trials = self._total_trials()
        report.overfitting = assess(
            trials=trials,
            in_sample_sharpe=in_sample.metrics.sharpe_ratio,
            out_of_sample_sharpe=out_of_sample.metrics.sharpe_ratio,
            out_of_sample_returns=bar_returns(out_of_sample.equity_curve),
            in_sample_trades=in_sample.metrics.total_trades,
            out_of_sample_trades=out_of_sample.metrics.total_trades,
            walk_forward_efficiency=(
                report.walk_forward.efficiency if report.walk_forward else None
            ),
            profitable_folds=(
                (report.walk_forward.profitable_folds, report.walk_forward.fold_count)
                if report.walk_forward and report.walk_forward.fold_count
                else None
            ),
            sensitivity_fragile=(
                report.robustness.fragile_parameters if report.robustness else ()
            ),
            degraded_segments=report.degraded_segments,
        )

        # --- what the findings would imply -------------------------------------
        # Last, and after the out-of-sample window has been scored: a
        # recommendation reads the other sections, so building it earlier would
        # mean reading results that did not exist yet.
        report.recommendations = build_recommendations(
            report,
            min_out_of_sample_trades=self.settings.feedback.min_out_of_sample_trades,
            min_walk_forward_folds=self.settings.feedback.min_walk_forward_folds,
        )

        # --- 9. record ------------------------------------------------------------
        self._persist(report)

        return StudyResult(
            report=report, experiment_id=self.experiment_id, store=self.store, trials=trials
        )

    # -------------------------------------------------------------- the steps

    def _split(self, warmup: int) -> list[Segment]:
        try:
            return chronological_split(
                self.data.df.index,
                fractions=tuple(self.settings.split_fractions),
                warmup_bars=warmup,
                embargo_bars=self.settings.embargo_bars,
                # A window shorter than a third of the indicators' own memory
                # cannot produce a result that stands on its own, and a study
                # built from three such windows would spend real time computing
                # statistics that describe nothing.
                min_segment_bars=max(30, warmup // 3),
            )
        except SplitError as exc:
            raise SplitError(
                f"{exc}\n\nThe study needs enough history for three windows plus a "
                f"{warmup:,}-bar feature warm-up. Request a longer period, or a finer "
                "timeframe over the same period."
            ) from exc

    def _run_segment(self, segment: Segment) -> SegmentRun:
        run = self.harness.run(segment, variant="baseline")
        self._count_trial("baseline", self.config, segment.name, run.metrics)
        logger.info(
            "Segment complete",
            extra={
                "segment": segment.name, "trades": run.metrics.total_trades,
                "quality": run.data_quality,
            },
        )
        return run

    def _isolate_strategies(self, segment: Segment) -> dict[str, SegmentRun]:
        """Run each enabled strategy on its own over ``segment``.

        The graph aggregates every strategy into a single signal before any
        position opens, so the trade log records one ensemble rather than which
        strategy earned what. Running them separately is the only way to see
        their return series apart from each other.

        These runs are not counted as trials. They are a measurement of the
        existing configuration's internal structure, not a search for a better
        one, and inflating the trial count with them would distort the
        overfitting arithmetic in the direction of pessimism.
        """
        names = self.config.strategies.enabled_names()
        if not self.settings.isolate_strategies or len(names) < 2:
            return {}

        runs: dict[str, SegmentRun] = {}
        for name in names:
            blocks = {
                key: block.model_copy(update={"enabled": key == name})
                for key, block in self.config.strategies.strategies.items()
            }
            solo = override_config(
                self.config,
                strategies=self.config.strategies.model_copy(update={"strategies": blocks}),
            )
            try:
                runs[name] = self.harness.run(segment, config=solo, variant=f"only_{name}")
            except Exception as exc:  # noqa: BLE001 - one strategy failing is data
                logger.warning(
                    "Could not run a strategy in isolation",
                    extra={"strategy": name, "error": str(exc)},
                )
        return runs

    def _candidates(self, report: ValidationReport) -> dict[str, AppConfig]:
        """Configuration variants suggested by the correlation findings.

        Empty when nothing was flagged, in which case walk-forward runs the
        baseline everywhere and adds no selection - which is the cleaner
        experiment when there is no hypothesis to test.
        """
        if not (self.settings.test_weight_variants and report.correlation):
            return {}
        if not report.correlation.findings:
            return {}
        return weight_variants(
            self.config, report.correlation, down_weight=self.settings.variant_down_weight
        )

    def _walk_forward(self, warmup: int, candidates: dict[str, AppConfig]):
        settings = self.settings.walk_forward
        train_bars, test_bars = settings.train_bars, settings.test_bars

        if train_bars is None or test_bars is None:
            train_bars, test_bars = suggest_walk_forward_geometry(
                len(self.data.df), warmup, folds=settings.folds,
                test_fraction=settings.test_fraction,
            )

        folds = walk_forward_folds(
            self.data.df.index,
            train_bars=train_bars,
            test_bars=test_bars,
            step_bars=settings.step_bars,
            warmup_bars=warmup,
            embargo_bars=self.settings.embargo_bars,
            anchored=settings.anchored,
            max_folds=settings.folds,
        )

        analysis = WalkForwardAnalysis(
            self.harness,
            folds,
            candidates=candidates or None,
            objective=self.objective,
            objective_name=self.settings.objective,
            on_trial=lambda name, cfg, segment, metrics: self._count_trial(
                name, cfg, segment.name, metrics
            ),
        )
        return analysis.run()

    def _compare_variants(
        self, segment: Segment, candidates: dict[str, AppConfig]
    ) -> tuple[list[dict[str, Any]], str]:
        """Test the redundancy hypotheses on the validation window.

        The correlation was measured in-sample, so the variants it suggested are
        tested somewhere else. Not on the out-of-sample window: comparing
        several configurations there and reporting the best would consume the
        one window whose result is supposed to mean something.
        """
        rows: list[dict[str, Any]] = []
        for name, candidate in candidates.items():
            run = self.harness.run(segment, config=candidate, variant=name)
            self._count_trial(name, candidate, segment.name, run.metrics)
            rows.append(variant_row(name, run.metrics))
        return rows, summarise_variant_comparison(rows)

    def _regime_table(self, runs: list[SegmentRun]) -> pd.DataFrame:
        """Per-regime performance, labelled by the window it came from."""
        frames = []
        for run in runs:
            table = run.result.regime_performance
            if table is None or table.empty:
                continue
            labelled = table.copy()
            labelled.insert(0, "window", run.segment.name)
            frames.append(labelled)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------- accounting

    def _count_trial(
        self, variant: str, config: AppConfig, segment: str, metrics: Any
    ) -> None:
        """Record that one configuration was evaluated against this data.

        Counted per (configuration, window): the same configuration on the same
        window twice is one trial. Overstating the search is as wrong as
        understating it, and both distort the deflated Sharpe.
        """
        fingerprint = stable_hash(
            {
                "features": config.features.model_dump(),
                "regimes": config.regimes.model_dump(),
                "strategies": config.strategies.model_dump(),
                "risk": config.risk.model_dump(),
                "execution": config.execution.model_dump(),
            }
        )
        self._session_trials.add(f"{fingerprint}:{segment}")

        if self.store is None:
            return
        try:
            self.store.record_trial(
                self.experiment_id,
                data_checksum=self.data_checksum,
                variant=variant,
                config_hash=fingerprint,
                segment=segment,
                objective=self.settings.objective,
                value=(
                    float(self.objective(metrics))
                    if metrics is not None and pd.notna(self.objective(metrics))
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail a study
            logger.warning("Could not record trial", extra={"error": str(exc)})

    def _total_trials(self) -> int:
        """Distinct configurations ever evaluated against this data.

        Taken from the store when there is one, because last week's variants
        still happened and the expected best result of a search does not reset
        when the process restarts. Falls back to this session's count when
        running without a store, which understates it - and the report says so.
        """
        distinct_session = len({key.split(":")[0] for key in self._session_trials})
        if self.store is None:
            return max(1, distinct_session)
        try:
            return max(1, self.store.count_trials(self.data_checksum), distinct_session)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read the trial count", extra={"error": str(exc)})
            return max(1, distinct_session)

    def _register(self, verdict: str, data_quality: str | None = None) -> None:
        """Insert or refresh this study's metadata row."""
        if self.store is None:
            return
        try:
            self.store.record_experiment(
                ExperimentRecord(
                    experiment_id=self.experiment_id,
                    kind="validation_study",
                    symbol=self.data.symbol,
                    timeframe=self.data.timeframe.code,
                    period_start=str(self.data.start),
                    period_end=str(self.data.end),
                    bars=len(self.data),
                    data_provider=self.data.provider,
                    data_checksum=self.data_checksum,
                    data_quality=data_quality or self.quality_status or "UNKNOWN",
                    git_revision=self.git_revision,
                    random_seed=self.config.platform.random_seed,
                    config_hash=config_fingerprint(self._config_snapshot()),
                    config_snapshot=self._config_snapshot(),
                    spec=self.spec,
                    verdict=verdict,
                )
            )
        except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail a study
            logger.error("Could not register the experiment", extra={"error": str(exc)})

    def _persist(self, report: ValidationReport) -> None:
        """Write the report to disk and the metadata to the experiment store."""
        try:
            path = report.save()
        except OSError as exc:
            logger.error("Could not write the validation report", extra={"error": str(exc)})
            path = None

        if self.store is None:
            return

        verdict, _ = report.verdict()
        self._register(verdict=verdict, data_quality=report.data_quality)

        try:
            for run in report.segments:
                self.store.record_segment(
                    self.experiment_id,
                    name=run.segment.name,
                    role=str(run.segment.role),
                    start_time=str(run.segment.start_time),
                    end_time=str(run.segment.end_time),
                    bars=run.segment.bars,
                    trades=run.metrics.total_trades,
                    data_quality=run.data_quality,
                    variant=run.variant,
                )
                self.store.record_metrics(
                    self.experiment_id, run.segment.name, run.metrics.to_dict()
                )

            if report.walk_forward is not None:
                self.store.record_metrics(
                    self.experiment_id,
                    "walk_forward",
                    report.walk_forward.stitched_metrics.to_dict(),
                )

            if report.overfitting is not None:
                self.store.record_findings(
                    self.experiment_id,
                    [
                        {
                            "code": f.code,
                            "severity": str(f.severity),
                            "message": f.message,
                            "evidence": f.evidence,
                        }
                        for f in report.overfitting.findings
                    ],
                )

            if path is not None:
                self.store.record_artifact(self.experiment_id, "validation_report", path)
        except Exception as exc:  # noqa: BLE001 - a failed record must not lose the study
            logger.error("Could not record the experiment", extra={"error": str(exc)})
