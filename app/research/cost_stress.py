"""Execution-cost stress testing.

Every backtest in this project pays costs the configuration assumes: a spread
scaled from the asset's typical value, slippage modelled as a fraction of ATR,
and any commission set in ``configs/execution.yaml``. Those assumptions are
optimistic by construction - the spread is a constant, when on a real venue it
widens exactly when a stop is being hit; slippage is smooth, when in practice it
arrives in the worst tick of a fast move. A result that only holds at the
assumed costs is describing the assumption, not the market.

So this asks a single question: **does whatever the configuration does at
baseline survive paying more to trade?** It runs the *same fixed configuration*
over one development window under a handful of named, declared cost scenarios -
baseline plus adverse - and reports how the outcome degrades. It is not a
search:

- The scenarios are fixed and named. Nothing here picks the cost assumption
  that makes the result look best; that would be the overfitting machine wearing
  a different hat.
- The baseline is always the headline. An adverse scenario can only ever be
  reported as degradation *from* it.
- Nothing concludes a configuration is profitable. The strongest thing this can
  say is that an apparent edge was not destroyed by modest realistic costs, and
  that is only ever "survived this round", never "works".

Every scenario is applied through :func:`app.research.robustness.set_parameter`,
which re-validates the whole configuration, and then run through the ordinary
:class:`~app.research.harness.ResearchHarness` - the same backtester and the
same :class:`~app.execution.simulator.ExecutionSimulator` a real study uses. A
scenario changes ``execution.*`` and nothing else, so the difference in the
numbers is the difference the cost assumption made, computed on real fills.

These runs are deliberately **not** counted as data-snooping trials. The
overfitting arithmetic counts distinct configurations a search *could have
selected*; an adverse-cost scenario is never a candidate to ship, and every one
of them is designed to make the result worse, so folding them into the trial
count would misdescribe the search rather than protect against it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.backtest.metrics import PerformanceMetrics
from app.config.loader import AppConfig
from app.config.models import CostStressSettings
from app.research.harness import ResearchHarness
from app.research.objectives import Objective, sortino
from app.research.robustness import set_parameter
from app.research.splits import Segment
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: The scenario every other one is measured against.
BASELINE = "baseline"

#: The single scenario the survival flag keys on: a modest, realistic
#: combination of adverse costs applied together. Named as a constant so the
#: report and the tests agree on which scenario is decisive.
MODEST_ADVERSE = "adverse_combined"


@dataclass(frozen=True)
class CostScenario:
    """One declared set of execution-cost overrides.

    ``overrides`` maps dotted configuration paths - always under ``execution`` -
    to the values this scenario forces. An empty mapping is the baseline: the
    configuration exactly as shipped.
    """

    name: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def is_baseline(self) -> bool:
        return not self.overrides


@dataclass(frozen=True)
class ScenarioResult:
    """What one scenario produced on the stress window.

    Two different kinds of "not usable" are kept apart on purpose. A scenario
    can *error* - its configuration was invalid, or the backtest raised - and
    then it has no numbers at all. Or it can run cleanly but produce too few
    trades for the ranking objective to be defined, in which case the objective
    is ``NaN`` while expectancy, return and costs are perfectly real. Collapsing
    the two would hide a valid degradation result behind an "error" label
    whenever the window happened to be thin.
    """

    scenario: CostScenario
    objective: float
    metrics: PerformanceMetrics
    error: str | None = None

    @property
    def errored(self) -> bool:
        return self.error is not None

    @property
    def has_metrics(self) -> bool:
        """The run completed and produced trades, so its metrics mean something."""
        return self.error is None and self.metrics.total_trades > 0

    @property
    def objective_finite(self) -> bool:
        """The ranking objective could be computed (enough trades to rank on)."""
        return self.error is None and math.isfinite(self.objective)

    @property
    def name(self) -> str:
        return self.scenario.name


def default_scenarios(config: AppConfig, settings: CostStressSettings) -> list[CostScenario]:
    """Build the declared scenario set from configuration.

    The magnitudes live in ``configs/research.yaml`` so they are auditable and
    so a reviewer can see exactly how hard the costs were pushed. They are
    applied to the configuration's *current* execution settings:

    - a spread scenario scales ``execution.spread_multiplier``;
    - a slippage scenario scales ``execution.slippage_value``;
    - the commission scenario sets ``execution.commission_pct`` to an absolute
      floor, because the shipped configuration charges no commission at all and
      a multiple of zero would leave it untested.

    ``adverse_combined`` applies the *smallest* of each together - the modest,
    realistic case - because that is the one a survival claim has to clear. The
    larger single-factor scenarios exist to show the gradient, not to set the
    bar.
    """
    execution = config.execution
    scenarios: list[CostScenario] = [
        CostScenario(BASELINE, "The configuration exactly as shipped.")
    ]

    for mult in settings.spread_multipliers:
        scenarios.append(
            CostScenario(
                f"spread_x{mult:g}",
                f"Spread widened to {mult:g}x the configured value.",
                {"execution.spread_multiplier": float(execution.spread_multiplier) * mult},
            )
        )

    for mult in settings.slippage_multipliers:
        scenarios.append(
            CostScenario(
                f"slippage_x{mult:g}",
                f"Slippage widened to {mult:g}x the configured value.",
                {"execution.slippage_value": float(execution.slippage_value) * mult},
            )
        )

    if settings.commission_pct > 0:
        scenarios.append(
            CostScenario(
                "commission_added",
                f"A commission of {settings.commission_pct:.4%} of notional per fill, "
                "where the baseline charges none.",
                {"execution.commission_pct": settings.commission_pct},
            )
        )

    # The decisive scenario: the mildest adverse spread and slippage together,
    # plus the commission. Realistic retail costs, all at once.
    combined: dict[str, Any] = {}
    if settings.spread_multipliers:
        combined["execution.spread_multiplier"] = (
            float(execution.spread_multiplier) * min(settings.spread_multipliers)
        )
    if settings.slippage_multipliers:
        combined["execution.slippage_value"] = (
            float(execution.slippage_value) * min(settings.slippage_multipliers)
        )
    if settings.commission_pct > 0:
        combined["execution.commission_pct"] = settings.commission_pct
    if combined:
        scenarios.append(
            CostScenario(
                MODEST_ADVERSE,
                "Modest, realistic adverse costs applied together: the mildest spread "
                "and slippage widening plus commission. The scenario a survival claim "
                "must clear.",
                combined,
            )
        )

    return scenarios


@dataclass
class CostStressReport:
    """How the configuration held up as costs were made worse."""

    segment: str
    objective_name: str
    results: list[ScenarioResult] = field(default_factory=list)

    # -------------------------------------------------------------- accessors

    def result(self, name: str) -> ScenarioResult | None:
        return next((r for r in self.results if r.name == name), None)

    @property
    def baseline(self) -> ScenarioResult | None:
        return self.result(BASELINE)

    @property
    def adverse(self) -> list[ScenarioResult]:
        return [r for r in self.results if not r.scenario.is_baseline]

    # ------------------------------------------------------------- judgements

    @property
    def baseline_has_edge(self) -> bool:
        """Whether the baseline made money per unit of risk before extra stress.

        Expectancy per R is the report's headline per-trade figure, so it is the
        one the survival question is asked in. A configuration that does not
        clear zero here has no edge for costs to take away, which is a different
        statement from "the edge did not survive".
        """
        base = self.baseline
        return base is not None and base.has_metrics and base.metrics.expectancy_r > 0

    @property
    def survives(self) -> bool:
        """Whether an apparent baseline edge survived the modest adverse scenario.

        ``False`` when there was no edge to begin with, when the decisive
        scenario could not be run, or when it drove expectancy to zero or below.
        A ``True`` here is never a profitability finding - see :meth:`survival`.
        """
        if not self.baseline_has_edge:
            return False
        decisive = self.result(MODEST_ADVERSE) or self._worst_adverse()
        if decisive is None or not decisive.has_metrics:
            return False
        return decisive.metrics.expectancy_r > 0

    def _worst_adverse(self) -> ScenarioResult | None:
        scored = [r for r in self.adverse if r.has_metrics]
        return min(scored, key=lambda r: r.metrics.expectancy_r, default=None)

    def survival(self) -> tuple[str, str]:
        """A one-word status and a sentence, both honest.

        Status is ``NO_BASELINE_EDGE``, ``DID_NOT_SURVIVE``, ``SURVIVED`` or
        ``NOT_ASSESSED``. None of them means profitable.
        """
        base = self.baseline
        if base is None or not base.has_metrics:
            return "NOT_ASSESSED", (
                "The baseline scenario produced no usable result (it errored or opened no "
                "trades on the stress window), so cost survival could not be judged."
            )

        if not self.baseline_has_edge:
            return "NO_BASELINE_EDGE", (
                f"Baseline expectancy is {base.metrics.expectancy_r:+.3f}R before any added "
                "cost stress, so there is no apparent edge for costs to remove. Cost "
                "sensitivity is reported below, but survival is not a meaningful frame for a "
                "configuration that does not make money at its own assumed costs."
            )

        decisive = self.result(MODEST_ADVERSE) or self._worst_adverse()
        if decisive is None or not decisive.has_metrics:
            return "NOT_ASSESSED", (
                "The baseline showed an apparent edge, but the decisive adverse scenario "
                "could not be evaluated, so survival could not be judged."
            )

        base_e = base.metrics.expectancy_r
        stressed_e = decisive.metrics.expectancy_r
        if stressed_e > 0:
            return "SURVIVED", (
                f"Apparent baseline edge of {base_e:+.3f}R held at {stressed_e:+.3f}R under "
                f"'{decisive.name}'. This is survival of one round of cost stress on one "
                "window, not evidence the configuration is profitable."
            )
        return "DID_NOT_SURVIVE", (
            f"An apparent baseline edge of {base_e:+.3f}R fell to {stressed_e:+.3f}R under "
            f"'{decisive.name}' - modest, realistic costs. The edge is an artefact of the "
            "optimistic baseline cost assumptions, not something the strategy would keep on a "
            "real venue."
        )

    def cost_drag_range(self) -> tuple[float, float]:
        """Smallest and largest cost drag across scenarios that produced trades."""
        drags = [r.metrics.cost_drag for r in self.results if r.has_metrics]
        return (min(drags), max(drags)) if drags else (0.0, 0.0)

    # ------------------------------------------------------------ serialising

    def to_frame(self) -> pd.DataFrame:
        base = self.baseline
        base_e = base.metrics.expectancy_r if base and base.has_metrics else float("nan")
        rows = []
        for r in self.results:
            m = r.metrics
            rows.append(
                {
                    "scenario": r.name,
                    "is_baseline": r.scenario.is_baseline,
                    "objective": r.objective if r.objective_finite else float("nan"),
                    "trades": m.total_trades,
                    "total_return": m.total_return if r.has_metrics else float("nan"),
                    "expectancy_r": m.expectancy_r if r.has_metrics else float("nan"),
                    "expectancy_delta": (
                        m.expectancy_r - base_e
                        if r.has_metrics and math.isfinite(base_e)
                        else float("nan")
                    ),
                    "total_costs": m.total_costs if r.has_metrics else float("nan"),
                    "cost_drag": m.cost_drag if r.has_metrics else float("nan"),
                    "max_drawdown": m.max_drawdown if r.has_metrics else float("nan"),
                    "error": r.error,
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        status, detail = self.survival()
        base = self.baseline
        return {
            "segment": self.segment,
            "objective_name": self.objective_name,
            "baseline_has_edge": self.baseline_has_edge,
            "baseline_expectancy_r": (
                base.metrics.expectancy_r if base and base.has_metrics else None
            ),
            "survives": self.survives,
            "survival_status": status,
            "survival_detail": detail,
            "cost_drag_range": list(self.cost_drag_range()),
            "scenarios": [
                {
                    "scenario": r.name,
                    "description": r.scenario.description,
                    "overrides": r.scenario.overrides,
                    "is_baseline": r.scenario.is_baseline,
                    "objective": r.objective if r.objective_finite else None,
                    "trades": r.metrics.total_trades,
                    "total_return": r.metrics.total_return if r.has_metrics else None,
                    "expectancy_r": r.metrics.expectancy_r if r.has_metrics else None,
                    "total_costs": r.metrics.total_costs if r.has_metrics else None,
                    "cost_drag": r.metrics.cost_drag if r.has_metrics else None,
                    "max_drawdown": r.metrics.max_drawdown if r.has_metrics else None,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

    def render(self) -> str:
        width = 68
        lines = [
            "-" * width,
            f"EXECUTION-COST STRESS (objective: {self.objective_name}, "
            f"segment: {self.segment})",
            "-" * width,
        ]
        if not self.results:
            lines.append("  No cost scenarios were run.")
            return "\n".join(lines)

        base = self.baseline
        base_e = base.metrics.expectancy_r if base and base.has_metrics else float("nan")
        lines.append(
            f"  {'scenario':<22}{'trades':>8}{'return':>10}{'expectancy':>12}{'Δexp':>9}{'cost drag':>11}"
        )
        for r in self.results:
            m = r.metrics
            if r.errored:
                lines.append(f"  {r.name:<22}{'—':>8}{'—':>10}{'error':>12}{'':>9}{'':>11}")
                continue
            if not r.has_metrics:
                lines.append(f"  {r.name:<22}{0:>8,}{'—':>10}{'no trades':>12}{'':>9}{'':>11}")
                continue
            delta = m.expectancy_r - base_e if math.isfinite(base_e) else float("nan")
            delta_s = f"{delta:+.3f}" if math.isfinite(delta) else "  —"
            lines.append(
                f"  {r.name:<22}{m.total_trades:>8,}{m.total_return:>9.1%}"
                f"{m.expectancy_r:>+12.3f}{delta_s:>9}{m.cost_drag:>10.2%}"
            )

        status, detail = self.survival()
        lines += ["", f"  Survival: {status}", f"  {detail}"]
        lines += [
            "",
            "  Scenarios are fixed and declared, never selected. The baseline is the",
            "  headline; adverse scenarios are only ever degradation from it. Surviving",
            "  cost stress is not evidence of profitability.",
        ]
        return "\n".join(lines)


def run_cost_stress_study(
    harness: ResearchHarness,
    segment: Segment,
    scenarios: list[CostScenario],
    *,
    config: AppConfig | None = None,
    objective: Objective = sortino,
    objective_name: str = "sortino",
) -> CostStressReport:
    """Run every scenario over ``segment`` and describe the degradation.

    Args:
        harness: the segment runner. Its backtester and execution simulator are
            the ones a real study uses, so a scenario's cost overrides reach the
            actual fills rather than a parallel calculation.
        segment: which window to stress on. Must be a development window
            (in-sample or validation) - stressing on the out-of-sample window
            would consume it, and this is a development-time question.
        scenarios: the declared scenario set, baseline first. Built by
            :func:`default_scenarios` from configuration.
        config: configuration to stress. Defaults to the harness baseline. Each
            scenario is applied on top of this via ``set_parameter``.
        objective: how each run is scored, for the shape column.
        objective_name: label for the objective in the report.
    """
    base_config = config or harness.config
    results: list[ScenarioResult] = []

    for scenario in scenarios:
        try:
            scenario_config = base_config
            for path, value in scenario.overrides.items():
                if not path.startswith("execution."):
                    raise ValueError(
                        f"Cost scenario {scenario.name!r} overrides {path!r}, which is not an "
                        "execution setting. A cost scenario must only change execution costs."
                    )
                scenario_config = set_parameter(scenario_config, path, value)

            run = harness.run(segment, config=scenario_config, variant=f"cost:{scenario.name}")
            results.append(
                ScenarioResult(
                    scenario=scenario,
                    objective=objective(run.metrics),
                    metrics=run.metrics,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad scenario is data, not a crash
            logger.warning(
                "Cost scenario failed",
                extra={"scenario": scenario.name, "error": str(exc)},
            )
            results.append(
                ScenarioResult(
                    scenario=scenario,
                    objective=float("nan"),
                    metrics=PerformanceMetrics(),
                    error=str(exc),
                )
            )

    return CostStressReport(
        segment=segment.name, objective_name=objective_name, results=results
    )
