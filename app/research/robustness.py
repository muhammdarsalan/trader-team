"""Parameter robustness and sensitivity.

Every threshold in ``configs/`` is a guess. RSI(14) is a number Wilder picked
in 1978; a 2.0 ATR stop is round because humans like round numbers. If a
configuration only works at exactly those values and falls apart at 13 or 15,
what has been discovered is a property of the sample, not of the market - the
parameter was fitted to noise that happened to sit at that value.

So the test is not "which value is best". That question is the overfitting
machine itself, and asking it is how a research process talks itself into a
peak. The test is: **is the neighbourhood flat?** A parameter whose
neighbouring values all behave similarly is describing something real, however
mediocre. A parameter with a sharp peak at the shipped value is describing the
sample.

Two things follow from that, and both are implemented:

- The report never recommends the best value found. It reports the shape.
- A peak *at the current setting* is treated as a red flag rather than as
  confirmation, because that is the shape produced by having tuned it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from app.backtest.metrics import PerformanceMetrics
from app.config.loader import AppConfig, ConfigError, override_config
from app.research.harness import ResearchHarness
from app.research.objectives import Objective, sortino
from app.research.splits import Segment
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ParameterPathError(ConfigError):
    """Raised when a dotted parameter path does not exist in the configuration."""


# ----------------------------------------------------------- config surgery


def get_parameter(config: AppConfig, path: str) -> Any:
    """Read a configuration value by dotted path.

    ``"risk.risk_per_trade"``, ``"features.rsi_period"`` and
    ``"strategies.strategies.trend_following.params.fast_ema"`` all work.
    """
    parts = path.split(".")
    node: Any = config
    for depth, part in enumerate(parts):
        if isinstance(node, BaseModel):
            if part not in type(node).model_fields:
                raise ParameterPathError(
                    f"{path!r}: {type(node).__name__} has no field {part!r}. "
                    f"Available: {sorted(type(node).model_fields)}"
                )
            node = getattr(node, part)
        elif isinstance(node, dict):
            if part not in node:
                raise ParameterPathError(
                    f"{path!r}: no key {part!r}. Available: {sorted(map(str, node))}"
                )
            node = node[part]
        elif hasattr(node, part):
            node = getattr(node, part)
        else:
            consumed = ".".join(parts[:depth])
            raise ParameterPathError(
                f"{path!r}: {consumed or 'the configuration'} is a "
                f"{type(node).__name__} and cannot be indexed by {part!r}"
            )
    return node


def set_parameter(config: AppConfig, path: str, value: Any) -> AppConfig:
    """A copy of ``config`` with one dotted-path value replaced.

    The replacement is re-validated, so a sweep cannot quietly produce a
    configuration the platform would have rejected at startup - a MACD fast
    period above its slow period, a negative risk fraction. An invalid grid
    point is an error worth seeing, not a data point.
    """
    section, _, remainder = path.partition(".")
    if not remainder:
        raise ParameterPathError(
            f"{path!r} names a whole configuration section. Sweep a field inside it."
        )
    if section not in AppConfig.__dataclass_fields__:
        raise ParameterPathError(
            f"Unknown configuration section {section!r} in {path!r}. "
            f"Available: {sorted(AppConfig.__dataclass_fields__)}"
        )

    current = getattr(config, section)
    updated = _replace_in(current, remainder.split("."), value, path)

    try:
        revalidated = type(updated).model_validate(updated.model_dump())
    except ValidationError as exc:
        raise ParameterPathError(
            f"Setting {path} = {value!r} produces an invalid {type(updated).__name__}:\n{exc}"
        ) from exc

    return override_config(config, **{section: revalidated})


def _replace_in(node: Any, parts: list[str], value: Any, path: str) -> Any:
    """Rebuild ``node`` with the leaf at ``parts`` replaced."""
    if not parts:
        return value

    head, rest = parts[0], parts[1:]

    if isinstance(node, BaseModel):
        if head not in type(node).model_fields:
            raise ParameterPathError(
                f"{path!r}: {type(node).__name__} has no field {head!r}. "
                f"Available: {sorted(type(node).model_fields)}"
            )
        return node.model_copy(
            update={head: _replace_in(getattr(node, head), rest, value, path)}
        )

    if isinstance(node, dict):
        if head not in node:
            raise ParameterPathError(
                f"{path!r}: no key {head!r}. Available: {sorted(map(str, node))}"
            )
        return {**node, head: _replace_in(node[head], rest, value, path)}

    raise ParameterPathError(
        f"{path!r}: cannot descend into a {type(node).__name__} looking for {head!r}"
    )


def neighbourhood(baseline: Any, *, span: float = 0.3, points: int = 5) -> list[Any]:
    """A grid around ``baseline``, preserving its type.

    Deliberately narrow and centred. A wide grid turns a sensitivity test into a
    parameter search, which is the thing this module exists to avoid: the
    question is whether the immediate neighbourhood is flat, not what the best
    value in a broad range would be.
    """
    if isinstance(baseline, bool):
        return [False, True]
    if not isinstance(baseline, (int, float)):
        raise TypeError(f"Cannot build a numeric neighbourhood around {baseline!r}")

    points = max(3, int(points) | 1)          # odd, so the baseline is included
    factors = np.linspace(1 - span, 1 + span, points)
    values = [baseline * f for f in factors]

    if isinstance(baseline, int):
        deduped = sorted({max(1, int(round(v))) for v in values})
        return deduped
    return [float(v) for v in values]


# --------------------------------------------------------------------- report


@dataclass(frozen=True)
class SweepPoint:
    """One grid point: a parameter value and what it produced."""

    parameter: str
    value: Any
    is_baseline: bool
    objective: float
    metrics: PerformanceMetrics
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.error is None and math.isfinite(self.objective)


@dataclass(frozen=True)
class ParameterSensitivity:
    """How one parameter behaves across its neighbourhood."""

    parameter: str
    baseline_value: Any
    points: tuple[SweepPoint, ...]
    objective_name: str

    # --- shape ---
    mean: float = 0.0
    std: float = 0.0
    coefficient_of_variation: float = float("inf")
    positive_fraction: float = 0.0
    worst_adjacent_drop: float = 0.0
    baseline_is_peak: bool = False
    fragile: bool = False
    notes: tuple[str, ...] = ()

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "parameter": p.parameter,
                    "value": p.value,
                    "baseline": p.is_baseline,
                    "objective": p.objective if p.usable else float("nan"),
                    "trades": p.metrics.total_trades,
                    "total_return": p.metrics.total_return,
                    "max_drawdown": p.metrics.max_drawdown,
                    "expectancy_r": p.metrics.expectancy_r,
                    "error": p.error,
                }
                for p in self.points
            ]
        )


@dataclass
class RobustnessReport:
    """Sensitivity across every parameter swept."""

    segment: str
    objective_name: str
    sensitivities: list[ParameterSensitivity] = field(default_factory=list)

    @property
    def fragile_parameters(self) -> tuple[str, ...]:
        return tuple(s.parameter for s in self.sensitivities if s.fragile)

    def to_frame(self) -> pd.DataFrame:
        if not self.sensitivities:
            return pd.DataFrame()
        return pd.concat([s.to_frame() for s in self.sensitivities], ignore_index=True)

    def render(self) -> str:
        width = 60
        lines = [
            "-" * width,
            f"PARAMETER SENSITIVITY (objective: {self.objective_name}, "
            f"segment: {self.segment})",
            "-" * width,
        ]
        if not self.sensitivities:
            lines.append("  No parameters were swept.")
            return "\n".join(lines)

        lines.append(
            f"  {'parameter':<44}{'spread':>8}{'flat?':>8}"
        )
        for s in self.sensitivities:
            verdict = "FRAGILE" if s.fragile else "stable"
            lines.append(
                f"  {s.parameter:<44}{s.coefficient_of_variation:>8.2f}{verdict:>8}"
            )

        for s in self.sensitivities:
            if s.notes:
                lines.append("")
                lines.append(f"  {s.parameter}:")
                lines += [f"    - {note}" for note in s.notes]

        lines += [
            "",
            "  A flat neighbourhood is the good outcome, even at a mediocre level. A peak",
            "  at the shipped value is the signature of a parameter that was tuned to this",
            "  sample. No best value is recommended here on purpose.",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------- sweep


def sweep_parameter(
    harness: ResearchHarness,
    segment: Segment,
    parameter: str,
    values: list[Any] | None = None,
    *,
    config: AppConfig | None = None,
    objective: Objective = sortino,
    objective_name: str = "sortino",
    span: float = 0.3,
    points: int = 5,
    on_trial: Any = None,
) -> ParameterSensitivity:
    """Run ``segment`` once per value of ``parameter`` and describe the shape.

    Args:
        harness: the segment runner.
        segment: which window to sweep on. Should be an in-sample or validation
            window - sweeping on the out-of-sample window consumes it.
        parameter: dotted path, e.g. ``"features.rsi_period"``.
        values: grid. Defaults to a narrow neighbourhood of the current value.
        objective: how each run is scored.
        span, points: shape of the default neighbourhood.
        on_trial: optional callback ``(variant, config, metrics)`` invoked per
            grid point, so the caller can count trials for the data-snooping
            arithmetic.
    """
    base_config = config or harness.config
    baseline = get_parameter(base_config, parameter)
    grid = values if values is not None else neighbourhood(baseline, span=span, points=points)
    if baseline not in grid:
        grid = sorted([*grid, baseline], key=lambda v: (v is None, v))

    swept: list[SweepPoint] = []
    for value in grid:
        variant = f"{parameter}={value}"
        try:
            variant_config = set_parameter(base_config, parameter, value)
            run = harness.run(segment, config=variant_config, variant=variant)
            score = objective(run.metrics)
            swept.append(
                SweepPoint(
                    parameter=parameter,
                    value=value,
                    is_baseline=value == baseline,
                    objective=score,
                    metrics=run.metrics,
                )
            )
            if on_trial is not None:
                on_trial(variant, variant_config, run.metrics)
        except Exception as exc:  # noqa: BLE001 - one bad grid point is data, not a crash
            logger.warning(
                "Sweep point failed",
                extra={"parameter": parameter, "value": value, "error": str(exc)},
            )
            swept.append(
                SweepPoint(
                    parameter=parameter, value=value, is_baseline=value == baseline,
                    objective=float("nan"), metrics=PerformanceMetrics(), error=str(exc),
                )
            )

    return _describe(parameter, baseline, swept, objective_name)


def _describe(
    parameter: str, baseline: Any, points: list[SweepPoint], objective_name: str
) -> ParameterSensitivity:
    """Turn a grid of results into a statement about shape."""
    notes: list[str] = []
    usable = [p for p in points if p.usable]

    if len(usable) < 3:
        notes.append(
            f"Only {len(usable)} of {len(points)} grid points produced a usable result "
            "(most often too few trades to score). The shape cannot be read from this."
        )
        return ParameterSensitivity(
            parameter=parameter, baseline_value=baseline, points=tuple(points),
            objective_name=objective_name, notes=tuple(notes),
        )

    scores = np.array([p.objective for p in usable], dtype=float)
    mean = float(scores.mean())
    std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
    cv = float(std / abs(mean)) if abs(mean) > 1e-9 else float("inf")
    positive = float((scores > 0).mean())

    # A cliff is one step between neighbouring values that is both large in
    # absolute terms and much larger than the other steps. Measuring it against
    # the range instead would call any flat neighbourhood a cliff, since on a
    # flat surface the largest of several tiny steps is still most of a tiny
    # range - the shape would be read from rounding error.
    ordered = sorted(usable, key=lambda p: (p.value is None, p.value))
    adjacent = np.array([p.objective for p in ordered], dtype=float)
    steps = np.abs(np.diff(adjacent))

    level = max(abs(mean), std, 1e-9)
    worst_drop = float(steps.max() / level) if len(steps) else 0.0
    concentrated = bool(len(steps) > 2 and steps.max() > 2.5 * float(np.median(steps)))

    baseline_point = next((p for p in usable if p.is_baseline), None)
    is_peak = bool(
        baseline_point is not None
        and baseline_point.objective >= scores.max() - 1e-12
        and len(usable) >= 3
    )

    fragile = False

    if cv > 1.0:
        fragile = True
        notes.append(
            f"The objective varies more than its own mean across the neighbourhood "
            f"(spread {cv:.2f}). Small changes to this parameter change the conclusion."
        )
    if 0.0 < positive < 1.0:
        notes.append(
            f"Only {positive:.0%} of neighbouring values keep the objective positive. The "
            "sign of the result depends on the parameter, not only its magnitude."
        )
    if worst_drop > 0.6 and concentrated:
        fragile = True
        notes.append(
            f"One step between adjacent values moves the objective by {worst_drop:.0%} of "
            "its own typical size, far more than the other steps. That is a cliff rather "
            "than a slope, and nothing about a market changes discontinuously at a round "
            "parameter value."
        )
    if is_peak:
        fragile = True
        notes.append(
            f"The shipped value ({baseline!r}) is the best in its own neighbourhood. That "
            "is what a parameter tuned to this sample looks like; it is a reason for "
            "suspicion rather than reassurance."
        )
    if not notes:
        notes.append(
            "The neighbourhood is flat: neighbouring values behave much like the shipped "
            "one. This says the result is not an artefact of this exact number. It says "
            "nothing about whether the result is any good."
        )

    return ParameterSensitivity(
        parameter=parameter,
        baseline_value=baseline,
        points=tuple(points),
        objective_name=objective_name,
        mean=mean,
        std=std,
        coefficient_of_variation=cv,
        positive_fraction=positive,
        worst_adjacent_drop=worst_drop,
        baseline_is_peak=is_peak,
        fragile=fragile,
        notes=tuple(notes),
    )


def run_robustness_study(
    harness: ResearchHarness,
    segment: Segment,
    parameters: list[str],
    *,
    config: AppConfig | None = None,
    objective: Objective = sortino,
    objective_name: str = "sortino",
    span: float = 0.3,
    points: int = 5,
    on_trial: Any = None,
) -> RobustnessReport:
    """Sweep several parameters over the same segment."""
    report = RobustnessReport(segment=segment.name, objective_name=objective_name)
    for parameter in parameters:
        try:
            report.sensitivities.append(
                sweep_parameter(
                    harness, segment, parameter, config=config, objective=objective,
                    objective_name=objective_name, span=span, points=points,
                    on_trial=on_trial,
                )
            )
        except ParameterPathError as exc:
            logger.error("Skipping unknown parameter", extra={"parameter": parameter,
                                                              "error": str(exc)})
    return report
