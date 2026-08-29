"""The validation report.

The point of this report is that someone can read it and disagree with it. A
single headline number cannot be argued with; it can only be believed or not.
So every section states what was measured, on which bars, and what would have
to be true for the number to mean what it appears to mean.

The report never says a configuration is profitable. It cannot: profitability
is a claim about the future, and everything here is a measurement of the past
with known biases attached. What it can say is whether a configuration survived
a specific set of attempts to break it, and it says that in those words.

"This does not work" is a successful outcome and is reported as plainly as any
other. Most configurations should reach it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.research.correlation import CorrelationReport
from app.research.feedback import RecommendationSet
from app.research.harness import SegmentRun
from app.research.monte_carlo import MonteCarloReport
from app.research.overfitting import OverfittingAssessment
from app.research.robustness import RobustnessReport
from app.research.walk_forward import WalkForwardReport
from app.utils.logging import get_logger
from app.utils.paths import ensure_dir, reports_dir

logger = get_logger(__name__)

WIDTH = 78

#: Verdicts. None of them is "profitable", and none of them ever will be.
VERDICT_INSUFFICIENT = "INSUFFICIENT EVIDENCE"
VERDICT_NEGATIVE = "EVIDENCE AGAINST"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_SURVIVED = "SURVIVED THIS ROUND OF TESTING"


@dataclass
class ValidationReport:
    """Every result from one study, assembled into something readable."""

    experiment_id: str
    symbol: str
    timeframe: str
    period_start: str | None = None
    period_end: str | None = None
    bars: int = 0
    data_provider: str = "unknown"
    data_checksum: str = ""
    data_quality: str = "UNKNOWN"
    git_revision: str = ""
    random_seed: int = 42
    created_at: str = ""

    segments: list[SegmentRun] = field(default_factory=list)
    walk_forward: WalkForwardReport | None = None
    monte_carlo: MonteCarloReport | None = None
    robustness: RobustnessReport | None = None
    correlation: CorrelationReport | None = None
    overfitting: OverfittingAssessment | None = None
    variant_rows: list[dict[str, Any]] = field(default_factory=list)
    variant_summary: str = ""
    #: What the findings would imply, and what evidence stands behind each.
    #: Built after every other section, because a recommendation is a reading of
    #: the others rather than a measurement of its own.
    recommendations: RecommendationSet | None = None
    regime_performance: pd.DataFrame = field(default_factory=pd.DataFrame)
    spec: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # -------------------------------------------------------------- accessors

    def segment(self, name: str) -> SegmentRun | None:
        return next((s for s in self.segments if s.segment.name == name), None)

    @property
    def in_sample(self) -> SegmentRun | None:
        return self.segment("in_sample")

    @property
    def validation(self) -> SegmentRun | None:
        return self.segment("validation")

    @property
    def out_of_sample(self) -> SegmentRun | None:
        return self.segment("out_of_sample")

    @property
    def degraded_segments(self) -> tuple[str, ...]:
        return tuple(s.segment.name for s in self.segments if s.degraded_data)

    # ---------------------------------------------------------------- verdict

    def verdict(self) -> tuple[str, list[str]]:
        """A qualitative conclusion and the reasons behind it.

        The thresholds here are conventions, not discoveries, and they are
        stated in the report so a reader can substitute their own.
        """
        reasons: list[str] = []
        oos = self.out_of_sample

        if oos is None or oos.metrics.total_trades == 0:
            reasons.append(
                "The out-of-sample window produced no trades, so nothing was tested there."
            )
            return VERDICT_INSUFFICIENT, reasons

        if oos.metrics.total_trades < 30:
            reasons.append(
                f"The out-of-sample window contains {oos.metrics.total_trades} trades. Below "
                "roughly 30 the statistics are not stable enough to support any conclusion."
            )
            return VERDICT_INSUFFICIENT, reasons

        if oos.metrics.expectancy_r <= 0:
            reasons.append(
                f"Out-of-sample expectancy is {oos.metrics.expectancy_r:+.3f}R per trade. On "
                "data it had never seen, this configuration lost money per unit of risk taken."
            )
            if self.in_sample and self.in_sample.metrics.expectancy_r > 0:
                reasons.append(
                    f"In-sample expectancy was {self.in_sample.metrics.expectancy_r:+.3f}R, so "
                    "the in-sample result did not carry over."
                )
            return VERDICT_NEGATIVE, reasons

        severe = self.overfitting.severe if self.overfitting else ()
        if severe:
            reasons.extend(f.message for f in severe)
            return VERDICT_INCONCLUSIVE, reasons

        if self.walk_forward and self.walk_forward.fold_count >= 3:
            good = self.walk_forward.profitable_folds
            total = self.walk_forward.fold_count
            if good <= total / 2:
                reasons.append(
                    f"Only {good} of {total} walk-forward folds had positive expectancy. "
                    "Out-of-sample survival did not repeat as the window rolled forward."
                )
                return VERDICT_INCONCLUSIVE, reasons

        if self.robustness and self.robustness.fragile_parameters:
            reasons.append(
                "These parameters change the outcome sharply for a small change in value: "
                f"{', '.join(self.robustness.fragile_parameters)}."
            )
            return VERDICT_INCONCLUSIVE, reasons

        reasons.append(
            f"Out-of-sample expectancy is {oos.metrics.expectancy_r:+.3f}R over "
            f"{oos.metrics.total_trades} trades, no severe overfitting diagnostic fired, and "
            "the parameter neighbourhood is flat."
        )
        reasons.append(
            "This means the configuration was not broken by the tests that were run. It is "
            "not a finding that it works, and it is not a prediction about anything."
        )
        return VERDICT_SURVIVED, reasons

    # ----------------------------------------------------------------- render

    def render(self) -> str:
        blocks = [
            self._header(),
            self._preamble(),
            self._segments_section(),
            self._walk_forward_section(),
            self._drawdown_section(),
            self._monte_carlo_section(),
            self._robustness_section(),
            self._correlation_section(),
            self._recommendation_section(),
            self._regime_section(),
            self._overfitting_section(),
            self._evidence_section(),
            self._verdict_section(),
            self._footer(),
        ]
        return "\n\n".join(b for b in blocks if b)

    def __str__(self) -> str:
        return self.render()

    # --- sections ------------------------------------------------------------

    def _header(self) -> str:
        return "\n".join(
            [
                "=" * WIDTH,
                "VALIDATION REPORT",
                "=" * WIDTH,
                f"Experiment:    {self.experiment_id}",
                f"Asset:         {self.symbol} {self.timeframe}",
                f"Period:        {self.period_start} -> {self.period_end}  ({self.bars:,} bars)",
                f"Data:          {self.data_provider}, quality {self.data_quality}, "
                f"checksum {self.data_checksum}",
                f"Code:          {self.git_revision}   seed {self.random_seed}",
                f"Generated:     {self.created_at}",
            ]
        )

    def _preamble(self) -> str:
        return "\n".join(
            [
                "-" * WIDTH,
                "HOW TO READ THIS",
                "-" * WIDTH,
                "  The in-sample numbers are description, not evidence. The configuration",
                "  exists because of those bars; it would be surprising if it did badly on",
                "  them, and it means nothing when it does well.",
                "",
                "  The validation numbers are weak evidence, and weaker every time they are",
                "  looked at. Each glance moves that window closer to being in-sample.",
                "",
                "  The out-of-sample and walk-forward numbers are the ones that carry weight,",
                "  and they are still weak. They are one path through one history of one",
                "  instrument, with execution assumptions that are estimates.",
                "",
                "  No section of this report claims a configuration is profitable. Nothing",
                "  here forecasts anything.",
            ]
        )

    def _segments_section(self) -> str:
        lines = ["-" * WIDTH, "IN-SAMPLE, VALIDATION AND OUT-OF-SAMPLE", "-" * WIDTH]
        if not self.segments:
            lines.append("  No segments were run.")
            return "\n".join(lines)

        frame = pd.DataFrame([s.summary_row() for s in self.segments])
        display = frame[
            ["segment", "bars", "trades", "total_return", "max_drawdown", "sharpe",
             "sortino", "expectancy_r", "win_rate", "data_quality"]
        ].copy()
        lines.append(display.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

        lines.append("")
        for run in self.segments:
            window = (
                f"{run.segment.start_time:%Y-%m-%d} -> {run.segment.end_time:%Y-%m-%d}"
            )
            lines.append(f"  {run.segment.name} ({window}, {run.variant}):")
            for note in run.notes:
                lines.append(f"    - {note}")
            for warning in run.metrics.warnings:
                lines.append(f"    - {warning}")
            if not run.notes and not run.metrics.warnings:
                lines.append("    - No caveats raised for this window.")

        degraded = self.degraded_segments
        if degraded:
            lines += [
                "",
                f"  Windows that ran on less than clean data: {', '.join(degraded)}. The risk",
                "  engine was told the grade of each window and refused to size positions",
                "  where the configuration says it should; a window that shows few trades and",
                "  a data-quality note is reporting that refusal, not a quiet strategy.",
            ]
        return "\n".join(lines)

    def _walk_forward_section(self) -> str:
        if self.walk_forward is None:
            return ""
        return self.walk_forward.render().replace("-" * 60, "-" * WIDTH)

    def _drawdown_section(self) -> str:
        """Drawdown deserves its own section because it is what ends accounts.

        A return figure is an outcome; a drawdown is an experience, and it is the
        one that decides whether a system is still being run when the outcome
        arrives.
        """
        lines = ["-" * WIDTH, "DRAWDOWN BEHAVIOUR", "-" * WIDTH]
        rows = []

        for run in self.segments:
            m = run.metrics
            rows.append(
                {
                    "window": run.segment.name,
                    "max_dd": m.max_drawdown,
                    "avg_dd": m.average_drawdown,
                    "longest_dd_bars": m.max_drawdown_duration_bars,
                    "worst_loss": m.largest_loss,
                    "max_consec_losses": m.max_consecutive_losses,
                    "calmar": m.calmar_ratio,
                }
            )

        if self.walk_forward is not None and self.walk_forward.fold_count:
            m = self.walk_forward.stitched_metrics
            rows.append(
                {
                    "window": "walk_forward",
                    "max_dd": m.max_drawdown,
                    "avg_dd": m.average_drawdown,
                    "longest_dd_bars": m.max_drawdown_duration_bars,
                    "worst_loss": m.largest_loss,
                    "max_consec_losses": m.max_consecutive_losses,
                    "calmar": m.calmar_ratio,
                }
            )

        if not rows:
            lines.append("  Nothing to report.")
            return "\n".join(lines)

        lines.append(
            pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )

        oos = self.out_of_sample
        if oos is not None and oos.metrics.max_drawdown_duration_bars:
            lines += [
                "",
                f"  The longest out-of-sample stretch below a previous peak lasted "
                f"{oos.metrics.max_drawdown_duration_bars:,} bars. That is the period a",
                "  person would have had to keep running this while it was not working, which",
                "  is a harder question than whether the arithmetic is favourable.",
            ]

        if self.monte_carlo is not None and self.monte_carlo.simulations:
            p95 = self.monte_carlo.max_drawdown.get("p95")
            observed = self.monte_carlo.observed_max_drawdown
            if p95 is not None:
                lines += [
                    "",
                    f"  Resampling the trade order puts the 95th-percentile drawdown at "
                    f"{p95:.1%}, against {observed:.1%} observed. The observed figure is one",
                    "  draw; sizing decisions should respect the distribution, not the draw.",
                ]
        return "\n".join(lines)

    def _monte_carlo_section(self) -> str:
        if self.monte_carlo is None:
            return ""
        return self.monte_carlo.render().replace("-" * 60, "-" * WIDTH)

    def _robustness_section(self) -> str:
        if self.robustness is None:
            return ""
        return self.robustness.render().replace("-" * 60, "-" * WIDTH)

    def _correlation_section(self) -> str:
        if self.correlation is None:
            return ""
        block = self.correlation.render().replace("-" * 60, "-" * WIDTH)
        if self.variant_summary:
            block += "\n\n" + self.variant_summary
        return block

    def _recommendation_section(self) -> str:
        if self.recommendations is None:
            return ""
        return self.recommendations.render()

    def _regime_section(self) -> str:
        if self.regime_performance is None or self.regime_performance.empty:
            return ""
        lines = ["-" * WIDTH, "PERFORMANCE BY MARKET REGIME", "-" * WIDTH]
        lines.append(
            self.regime_performance.to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )
        lines += [
            "",
            "  Regime labels come from the rule-based detector, so this table inherits its",
            "  thresholds. A strategy looking strong in one regime on a handful of trades is",
            "  a hypothesis about that regime and nothing more - the selector deliberately",
            "  ignores samples below its own minimum for the same reason.",
        ]
        return "\n".join(lines)

    def _overfitting_section(self) -> str:
        if self.overfitting is None:
            return ""
        return self.overfitting.render().replace("-" * 60, "-" * WIDTH)

    def evidence_summary(self) -> list[dict[str, Any]]:
        """What has and has not been established, by category.

        The categories are separated because they fail independently and because
        conflating them is the single most common way a research result gets
        overstated. "The tests pass" and "the strategy has an edge" are answers
        to different questions; a reader who sees only a green suite and a
        verdict line can merge them without noticing, so each category states
        its own status and its own limitation.

        ``status`` is one of ESTABLISHED, PARTIAL, NOT_ESTABLISHED or
        NOT_ASSESSED. None of them is ever "proven".
        """
        oos = self.out_of_sample
        rows: list[dict[str, Any]] = []

        # 1. Code. Deliberately first, and deliberately caveated: this is the
        #    category most likely to be mistaken for the others.
        rows.append(
            {
                "category": "Code and tests",
                "status": "SEPARATE QUESTION",
                "detail": (
                    "Whether the implementation does what it was written to do is "
                    "established by the test suite, not by this report."
                ),
                "limitation": (
                    "A passing suite on a losing strategy is a correctly implemented "
                    "losing strategy. Nothing about test results bears on whether an "
                    "edge exists."
                ),
            }
        )

        # 2. Out-of-sample evidence.
        if oos is None or oos.metrics.total_trades == 0:
            oos_status, oos_detail = "NOT_ESTABLISHED", (
                "The out-of-sample window produced no trades, so nothing was tested "
                "on unseen data."
            )
        elif oos.metrics.total_trades < 30:
            oos_status, oos_detail = "NOT_ESTABLISHED", (
                f"{oos.metrics.total_trades} out-of-sample trades. Below roughly 30 the "
                "sample is too small for the statistics to mean anything."
            )
        elif oos.metrics.expectancy_r <= 0:
            oos_status, oos_detail = "NOT_ESTABLISHED", (
                f"Out-of-sample expectancy is {oos.metrics.expectancy_r:+.3f}R over "
                f"{oos.metrics.total_trades} trades. On unseen data this configuration "
                "lost money per unit of risk."
            )
        else:
            oos_status, oos_detail = "PARTIAL", (
                f"Out-of-sample expectancy is {oos.metrics.expectancy_r:+.3f}R over "
                f"{oos.metrics.total_trades} trades, on one window scored once."
            )
        rows.append(
            {
                "category": "Out-of-sample evidence",
                "status": oos_status,
                "detail": oos_detail,
                "limitation": (
                    "One window of one instrument. A positive result here is the "
                    "beginning of evidence, not a conclusion."
                ),
            }
        )

        # 3. Statistical uncertainty.
        if self.monte_carlo is None:
            mc_status, mc_detail = "NOT_ASSESSED", "No resampling was run."
        else:
            mc_status, mc_detail = "PARTIAL", (
                "The out-of-sample trade sequence was resampled; the spread of outcomes "
                "is in the Monte Carlo section."
            )
        rows.append(
            {
                "category": "Statistical uncertainty",
                "status": mc_status,
                "detail": mc_detail,
                "limitation": (
                    "Resampling explores orderings of the trades that happened. It "
                    "cannot produce a trade the strategy never took."
                ),
            }
        )

        # 4. Robustness.
        if self.robustness is None or not self.robustness.sensitivities:
            rb_status, rb_detail = "NOT_ASSESSED", "No parameter neighbourhood was swept."
        elif self.robustness.fragile_parameters:
            rb_status, rb_detail = "NOT_ESTABLISHED", (
                "These parameters change the outcome sharply for a small change in "
                f"value: {', '.join(self.robustness.fragile_parameters)}."
            )
        else:
            rb_status, rb_detail = "PARTIAL", (
                f"{len(self.robustness.sensitivities)} parameter neighbourhood(s) swept "
                "and none was fragile."
            )
        rows.append(
            {
                "category": "Robustness",
                "status": rb_status,
                "detail": rb_detail,
                "limitation": (
                    "Swept one parameter at a time on the in-sample window. Interactions "
                    "between parameters are not explored."
                ),
            }
        )

        # 5. Overfitting risk.
        if self.overfitting is None:
            of_status, of_detail = "NOT_ASSESSED", "No overfitting diagnostics were run."
        elif self.overfitting.severe:
            of_status, of_detail = "NOT_ESTABLISHED", (
                f"{len(self.overfitting.severe)} severe finding(s): "
                + "; ".join(f.message for f in self.overfitting.severe)
            )
        else:
            of_status, of_detail = "PARTIAL", (
                "No severe diagnostic fired for the number of configurations tried."
            )
        rows.append(
            {
                "category": "Overfitting risk",
                "status": of_status,
                "detail": of_detail,
                "limitation": (
                    "The trial count covers searches this platform recorded. Choices "
                    "made before the first study - which strategies exist at all - are "
                    "not counted and cannot be."
                ),
            }
        )

        # 6. Correlation and redundancy.
        if self.correlation is None:
            corr_status, corr_detail = "NOT_ASSESSED", "Correlation was not measured."
        elif self.correlation.findings:
            corr_status, corr_detail = "PARTIAL", (
                f"{len(self.correlation.findings)} redundancy hypothesis(es) raised: "
                + "; ".join(str(f) for f in self.correlation.findings)
            )
        else:
            corr_status, corr_detail = "PARTIAL", (
                "No strategy pair exceeded the correlation threshold on this window."
            )
        rows.append(
            {
                "category": "Correlation and redundancy",
                "status": corr_status,
                "detail": corr_detail,
                "limitation": (
                    "Measured on one window. No strategy is disabled on the strength of "
                    "a correlation; candidates are tested instead."
                ),
            }
        )

        # 7. Data quality.
        degraded = self.degraded_segments
        rows.append(
            {
                "category": "Data quality",
                "status": "NOT_ESTABLISHED" if degraded else "PARTIAL",
                "detail": (
                    f"Windows that ran on less than clean data: {', '.join(degraded)}."
                    if degraded
                    else f"Every window graded clean. Series grade: {self.data_quality}."
                ),
                "limitation": (
                    "The feed is a free vendor series. Where the instrument is a proxy "
                    "for something else, the proxy's tracking error is not modelled."
                ),
            }
        )

        # 8. Execution assumptions.
        rows.append(
            {
                "category": "Execution assumptions",
                "status": "NOT_ESTABLISHED",
                "detail": (
                    "Fills are simulated: next-bar open, configured spread, modelled "
                    "slippage, commission. No order in this study met a real venue."
                ),
                "limitation": (
                    "Spread comes from configuration, not from quotes. On a real venue "
                    "it widens exactly when it matters most, and no result here reflects "
                    "that."
                ),
            }
        )
        return rows

    def _evidence_section(self) -> str:
        lines = ["-" * WIDTH, "WHAT HAS AND HAS NOT BEEN ESTABLISHED", "-" * WIDTH]
        lines.append(
            "  These fail independently. Merging them is the most common way a research"
        )
        lines.append("  result gets overstated, so each is stated separately.")
        lines.append("")
        for row in self.evidence_summary():
            lines.append(f"  {row['category']}: {row['status']}")
            lines.append(f"    {row['detail']}")
            lines.append(f"    limitation: {row['limitation']}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _verdict_section(self) -> str:
        verdict, reasons = self.verdict()
        lines = ["=" * WIDTH, f"VERDICT: {verdict}", "=" * WIDTH]
        lines += [f"  - {reason}" for reason in reasons]

        if verdict == VERDICT_SURVIVED:
            lines += [
                "",
                "  Surviving is the weakest positive result available here. It means the",
                "  tests that were run did not break this configuration. Other tests exist,",
                "  this is one instrument, and the future is not a sample from the past.",
            ]
        elif verdict == VERDICT_NEGATIVE:
            lines += [
                "",
                "  This is a successful research outcome. Knowing a configuration does not",
                "  work on unseen data is worth more than not knowing, and it was reached",
                "  cheaply.",
            ]
        return "\n".join(lines)

    def _footer(self) -> str:
        return "\n".join(
            [
                "=" * WIDTH,
                "Every figure above describes historical samples with realistic but estimated",
                "execution costs. Backtested and walk-forward performance is hypothetical and",
                "carries inherent limitations. Nothing here is financial advice, and no part",
                "of this platform executes real-money trades.",
                "=" * WIDTH,
            ]
        )

    # ------------------------------------------------------------------ output

    #: Dataclass fields whose serialised key differs from the attribute name.
    #: Kept explicit so :func:`test_to_dict_carries_every_section_the_report_holds`
    #: can prove that every field the report holds reaches the dict, rather than
    #: matching on names and quietly passing when a section is dropped.
    _DICT_ALIASES = {"variant_rows": "variants"}

    def to_dict(self) -> dict[str, Any]:
        """The machine-readable report.

        Every section the object holds appears here. That is not a stylistic
        preference: the dashboard and any other consumer read this dict rather
        than the rendered text, so a section missing from it is a section that
        does not exist as far as they are concerned. ``robustness``,
        ``variant_summary`` and ``regime_performance`` were absent for exactly
        that reason - the text report rendered parameter sensitivity and
        ``save()`` wrote it to CSV, while the JSON a reader would actually parse
        carried no trace of it.
        """
        verdict, reasons = self.verdict()
        return {
            "experiment_id": self.experiment_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "bars": self.bars,
            "data_provider": self.data_provider,
            "data_checksum": self.data_checksum,
            "data_quality": self.data_quality,
            "git_revision": self.git_revision,
            "random_seed": self.random_seed,
            "created_at": self.created_at,
            "spec": self.spec,
            "verdict": verdict,
            "verdict_reasons": reasons,
            "evidence_summary": self.evidence_summary(),
            "segments": [s.summary_row() for s in self.segments],
            "walk_forward": (
                {
                    "folds": self.walk_forward.fold_count,
                    "profitable_folds": self.walk_forward.profitable_folds,
                    "efficiency": self.walk_forward.efficiency,
                    "stitched": self.walk_forward.stitched_metrics.to_dict(),
                    "warnings": list(self.walk_forward.warnings),
                }
                if self.walk_forward
                else None
            ),
            "monte_carlo": self.monte_carlo.to_dict() if self.monte_carlo else None,
            "robustness": self.robustness.to_dict() if self.robustness else None,
            "correlation": self.correlation.to_dict() if self.correlation else None,
            "overfitting": self.overfitting.to_dict() if self.overfitting else None,
            "variants": self.variant_rows,
            "variant_summary": self.variant_summary,
            "recommendations": (
                self.recommendations.to_dict() if self.recommendations else None
            ),
            "regime_performance": (
                self.regime_performance.to_dict(orient="records")
                if not self.regime_performance.empty
                else []
            ),
            "notes": self.notes,
        }

    def save(self, directory: Path | None = None) -> Path:
        """Write the report and its machine-readable form to disk."""
        base = ensure_dir((directory or reports_dir() / "validation") / self.experiment_id)

        (base / "validation_report.txt").write_text(self.render(), encoding="utf-8")
        (base / "validation_report.json").write_text(
            json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8"
        )

        segment_rows = [s.summary_row() for s in self.segments]
        if segment_rows:
            pd.DataFrame(segment_rows).to_csv(base / "segments.csv", index=False)
        if self.walk_forward is not None and self.walk_forward.fold_count:
            self.walk_forward.to_frame().to_csv(base / "walk_forward_folds.csv", index=False)
            if not self.walk_forward.stitched_equity.empty:
                self.walk_forward.stitched_equity.to_csv(base / "walk_forward_equity.csv")
        if self.robustness is not None:
            frame = self.robustness.to_frame()
            if not frame.empty:
                frame.to_csv(base / "parameter_sensitivity.csv", index=False)
        if self.correlation is not None and not self.correlation.return_correlations.empty:
            self.correlation.return_correlations.to_csv(base / "strategy_correlation.csv")
        if self.variant_rows:
            pd.DataFrame(self.variant_rows).to_csv(base / "weight_variants.csv", index=False)

        logger.info(
            "Validation report saved",
            extra={"experiment_id": self.experiment_id, "path": str(base)},
        )
        return base
