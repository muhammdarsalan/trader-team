"""Overfitting and data-snooping diagnostics.

The uncomfortable arithmetic behind this module: if you test twenty random
configurations on the same history, the best of them has an expected Sharpe
ratio well above zero *even when none of them has any edge at all*. That number
is not a discovery. It is the maximum of twenty draws from a distribution
centred on nothing, and it is exactly what a research process produces if it
keeps searching and reports only the winner.

So the honest question is never "is this Sharpe good?" but "is this Sharpe
better than the best I would expect to see by chance, given how many things I
tried?". Answering it requires knowing the number of trials, which is why the
experiment store counts them against the data checksum rather than trusting
anyone to remember.

Two standard tools are implemented here:

**Expected maximum Sharpe under the null** - what the best of *N* worthless
strategies scores, from the extreme-value approximation in Bailey and López de
Prado's work on backtest overfitting.

**The deflated Sharpe ratio** - the probability that an observed Sharpe exceeds
that benchmark, after adjusting for how many trials produced it, for the sample
length, and for the skew and fat tails of the returns. Trading returns are
negatively skewed and heavy-tailed, both of which flatter a naive Sharpe.

Neither is a verdict. A high deflated Sharpe does not mean a strategy works; it
means one specific way of being fooled has been checked for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

#: Euler-Mascheroni constant, from the Gumbel approximation to the maximum of
#: independent draws.
EULER_MASCHERONI = 0.577_215_664_901_532_9


class Severity(StrEnum):
    """How hard a finding should push back on believing the result."""

    INFO = "INFO"
    CAUTION = "CAUTION"
    SEVERE = "SEVERE"


@dataclass(frozen=True)
class Finding:
    """One diagnostic, with the reason it was raised."""

    code: str
    severity: Severity
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


@dataclass(frozen=True)
class OverfittingAssessment:
    """Everything known about how much of a result could be selection."""

    trials: int
    findings: tuple[Finding, ...] = ()
    observed_sharpe: float | None = None
    #: The same Sharpe, per bar rather than annualised. Kept separately because
    #: the benchmark below is a per-bar figure, and comparing an annualised
    #: number against a per-bar one is off by a factor of sixteen on daily data.
    observed_sharpe_per_bar: float | None = None
    expected_max_sharpe_under_null: float | None = None
    deflated_sharpe: float | None = None
    in_sample_sharpe: float | None = None
    out_of_sample_sharpe: float | None = None
    degradation: float | None = None
    walk_forward_efficiency: float | None = None

    @property
    def severe(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.SEVERE)

    @property
    def worst_severity(self) -> Severity:
        for level in (Severity.SEVERE, Severity.CAUTION, Severity.INFO):
            if any(f.severity is level for f in self.findings):
                return level
        return Severity.INFO

    def render(self) -> str:
        width = 60
        lines = ["-" * width, "OVERFITTING AND DATA-SNOOPING", "-" * width]
        lines.append(f"  Configurations tested against this data   {self.trials:>10,}")
        if self.observed_sharpe is not None:
            lines.append(
                f"  Observed Sharpe, annualised (out of sample){self.observed_sharpe:>10.2f}"
            )
        if self.observed_sharpe_per_bar is not None:
            lines.append(
                f"  Observed Sharpe per bar                   {self.observed_sharpe_per_bar:>10.4f}"
            )
        if self.expected_max_sharpe_under_null is not None:
            lines.append(
                "  Best per-bar Sharpe expected from chance  "
                f"{self.expected_max_sharpe_under_null:>10.4f}"
            )
        if self.deflated_sharpe is not None:
            lines.append(f"  Deflated Sharpe (probability)             {self.deflated_sharpe:>10.2%}")
        if self.in_sample_sharpe is not None and self.out_of_sample_sharpe is not None:
            lines.append(
                f"  In-sample -> out-of-sample Sharpe         "
                f"{self.in_sample_sharpe:>5.2f} -> {self.out_of_sample_sharpe:.2f}"
            )
        if self.walk_forward_efficiency is not None:
            lines.append(f"  Walk-forward efficiency                   {self.walk_forward_efficiency:>10.2f}")

        lines.append("")
        if self.findings:
            for finding in self.findings:
                lines.append(f"  {finding}")
        else:
            lines.append("  No diagnostic fired. That is not the same as a validated result.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trials": self.trials,
            "observed_sharpe": self.observed_sharpe,
            "observed_sharpe_per_bar": self.observed_sharpe_per_bar,
            "expected_max_sharpe_under_null": self.expected_max_sharpe_under_null,
            "deflated_sharpe": self.deflated_sharpe,
            "in_sample_sharpe": self.in_sample_sharpe,
            "out_of_sample_sharpe": self.out_of_sample_sharpe,
            "degradation": self.degradation,
            "walk_forward_efficiency": self.walk_forward_efficiency,
            "findings": [
                {
                    "code": f.code,
                    "severity": str(f.severity),
                    "message": f.message,
                    "evidence": f.evidence,
                }
                for f in self.findings
            ],
        }


# ------------------------------------------------------------ normal helpers


def normal_cdf(x: float) -> float:
    """Standard normal CDF, from the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Standard normal quantile function.

    Acklam's rational approximation, accurate to about 1.15e-9 across the whole
    range. Implemented here rather than pulled from scipy so that the
    diagnostics carry no dependency the rest of the platform does not already
    need.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"normal_ppf expects 0 < p < 1, got {p}")

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)

    low, high = 0.02425, 1 - 0.02425

    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


# ------------------------------------------------------------------ the maths


def expected_max_sharpe(trials: int, sharpe_std: float = 1.0) -> float:
    """Best per-observation Sharpe expected from ``trials`` worthless strategies.

    The maximum of many independent draws follows a Gumbel distribution, and
    this is its mean. The practical consequence is worth stating plainly: with
    a hundred configurations tried, a Sharpe around 2.6 standard deviations
    above zero is the *expected* best result when every configuration is
    worthless. Beating zero was never the bar.

    Args:
        trials: how many configurations were evaluated.
        sharpe_std: the spread of Sharpe ratios across those trials. Defaults
            to 1.0, which expresses the answer in standard deviations.
    """
    n = max(1, int(trials))
    if n == 1:
        return 0.0
    gamma = EULER_MASCHERONI
    return float(
        sharpe_std
        * ((1 - gamma) * normal_ppf(1 - 1 / n) + gamma * normal_ppf(1 - 1 / (n * math.e)))
    )


def deflated_sharpe_ratio(
    returns: pd.Series,
    trials: int = 1,
    *,
    sharpe_std: float | None = None,
) -> tuple[float | None, dict[str, float]]:
    """Probability the observed Sharpe beats what selection alone would produce.

    Args:
        returns: per-bar returns of the equity curve being judged.
        trials: number of configurations evaluated to arrive at this one.
        sharpe_std: spread of per-observation Sharpe across those trials. When
            omitted, the theoretical value ``1/sqrt(T)`` is used, which assumes
            the trials were independent - optimistic, since variants of one
            strategy are anything but.

    Returns:
        ``(probability, diagnostics)``. The probability is None when the sample
        is too short or too degenerate to compute anything meaningful, which is
        itself worth reporting.
    """
    series = pd.to_numeric(returns, errors="coerce").dropna()
    n = len(series)
    diagnostics: dict[str, float] = {"observations": float(n)}

    if n < 30:
        return None, diagnostics

    values = series.to_numpy(dtype=float)
    std = float(values.std(ddof=1))
    if std <= 0:
        return None, diagnostics

    sharpe = float(values.mean() / std)          # per observation, not annualised
    skew = float(pd.Series(values).skew())
    kurtosis = float(pd.Series(values).kurtosis()) + 3.0   # pandas reports excess

    spread = sharpe_std if sharpe_std is not None else 1.0 / math.sqrt(n)
    benchmark = expected_max_sharpe(trials, spread)

    diagnostics.update(
        {
            "sharpe_per_observation": sharpe,
            "skew": skew,
            "kurtosis": kurtosis,
            "benchmark_sharpe": benchmark,
            "sharpe_std": spread,
        }
    )

    # The denominator is the standard error of the Sharpe estimate under
    # non-normal returns: negative skew and fat tails both make a given Sharpe
    # less impressive than it looks.
    variance_term = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if variance_term <= 0:
        return None, diagnostics

    statistic = (sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(variance_term)
    probability = normal_cdf(statistic)
    diagnostics["statistic"] = statistic
    return float(probability), diagnostics


# ---------------------------------------------------------------- assessment


def assess(
    *,
    trials: int,
    in_sample_sharpe: float | None = None,
    out_of_sample_sharpe: float | None = None,
    out_of_sample_returns: pd.Series | None = None,
    in_sample_trades: int = 0,
    out_of_sample_trades: int = 0,
    walk_forward_efficiency: float | None = None,
    profitable_folds: tuple[int, int] | None = None,
    sensitivity_fragile: tuple[str, ...] = (),
    degraded_segments: tuple[str, ...] = (),
) -> OverfittingAssessment:
    """Collect every overfitting signal into one assessment.

    Nothing here concludes that a configuration works. Every finding is a
    reason to trust the numbers less; the absence of findings is the absence of
    one specific kind of evidence against, which is not evidence for.
    """
    findings: list[Finding] = []

    # --- how much searching produced this -------------------------------------
    if trials >= 100:
        findings.append(
            Finding(
                "HEAVY_SEARCH", Severity.SEVERE,
                f"{trials:,} configurations have been evaluated against this data. The best "
                "of that many is expected to look good even if none of them has an edge; "
                "treat the leader as a hypothesis, not a finding.",
                {"trials": trials},
            )
        )
    elif trials >= 20:
        findings.append(
            Finding(
                "REPEATED_SEARCH", Severity.CAUTION,
                f"{trials:,} configurations have been evaluated against this data. Selection "
                "is now a material part of whatever the best one scores.",
                {"trials": trials},
            )
        )

    # --- degradation from development to unseen data --------------------------
    degradation: float | None = None
    if in_sample_sharpe is not None and out_of_sample_sharpe is not None:
        if abs(in_sample_sharpe) > 1e-9:
            degradation = out_of_sample_sharpe / in_sample_sharpe

        if out_of_sample_sharpe <= 0 < in_sample_sharpe:
            findings.append(
                Finding(
                    "OOS_SIGN_FLIP", Severity.SEVERE,
                    f"Sharpe was {in_sample_sharpe:.2f} in sample and {out_of_sample_sharpe:.2f} "
                    "out of sample. A result that reverses on unseen data is the signature "
                    "of fitting the sample rather than the market.",
                    {"in_sample": in_sample_sharpe, "out_of_sample": out_of_sample_sharpe},
                )
            )
        elif degradation is not None and degradation < 0.5 and in_sample_sharpe > 0:
            findings.append(
                Finding(
                    "OOS_DEGRADATION", Severity.CAUTION,
                    f"Out-of-sample Sharpe retains {degradation:.0%} of the in-sample figure. "
                    "Some decay is normal and expected; this much suggests the in-sample "
                    "number was substantially selection.",
                    {"degradation": degradation},
                )
            )

    # --- walk-forward behaviour ------------------------------------------------
    if walk_forward_efficiency is not None and walk_forward_efficiency < 0.5:
        findings.append(
            Finding(
                "LOW_WALK_FORWARD_EFFICIENCY", Severity.CAUTION,
                f"Walk-forward efficiency is {walk_forward_efficiency:.2f}: test windows keep "
                "less than half of what their training windows promised.",
                {"efficiency": walk_forward_efficiency},
            )
        )

    if profitable_folds is not None:
        good, total = profitable_folds
        if total >= 3 and good <= total / 2:
            findings.append(
                Finding(
                    "INCONSISTENT_FOLDS", Severity.CAUTION,
                    f"Only {good} of {total} walk-forward folds had positive expectancy. A "
                    "result carried by one or two windows is a result about those windows.",
                    {"profitable_folds": good, "total_folds": total},
                )
            )

    # --- sample sizes -----------------------------------------------------------
    if out_of_sample_trades < 30:
        findings.append(
            Finding(
                "THIN_OOS_SAMPLE", Severity.SEVERE if out_of_sample_trades < 10 else Severity.CAUTION,
                f"The out-of-sample window contains {out_of_sample_trades} trades. Below "
                "roughly 30, the statistics move materially on a single trade and nothing "
                "computed from them is stable.",
                {"trades": out_of_sample_trades},
            )
        )
    if in_sample_trades and out_of_sample_trades:
        ratio = out_of_sample_trades / in_sample_trades
        if ratio < 0.1:
            findings.append(
                Finding(
                    "OOS_TRADE_FREQUENCY_COLLAPSE", Severity.CAUTION,
                    f"The out-of-sample window produced {ratio:.0%} as many trades per the "
                    "in-sample rate. The system largely stopped trading rather than trading "
                    "badly, which is a different failure and needs a different fix.",
                    {"ratio": ratio},
                )
            )

    # --- the deflated Sharpe ------------------------------------------------------
    deflated: float | None = None
    benchmark: float | None = None
    per_bar: float | None = None
    if out_of_sample_returns is not None and not out_of_sample_returns.empty:
        deflated, diagnostics = deflated_sharpe_ratio(out_of_sample_returns, trials)
        benchmark = diagnostics.get("benchmark_sharpe")
        per_bar = diagnostics.get("sharpe_per_observation")
        if deflated is None:
            findings.append(
                Finding(
                    "DEFLATED_SHARPE_UNAVAILABLE", Severity.INFO,
                    "The out-of-sample sample is too short or too degenerate to compute a "
                    "deflated Sharpe ratio.",
                    diagnostics,
                )
            )
        elif deflated < 0.95:
            findings.append(
                Finding(
                    "DEFLATED_SHARPE_LOW", Severity.CAUTION if deflated > 0.5 else Severity.SEVERE,
                    f"The deflated Sharpe ratio is {deflated:.1%}. After adjusting for the "
                    f"{trials:,} configurations tried, the sample length and the shape of the "
                    "return distribution, the observed Sharpe is not distinguishable from what "
                    "selection alone would produce.",
                    diagnostics,
                )
            )

    # --- fragility and data quality --------------------------------------------
    if sensitivity_fragile:
        findings.append(
            Finding(
                "PARAMETER_FRAGILITY", Severity.SEVERE,
                "These parameters change the outcome sharply for a small change in value: "
                f"{', '.join(sensitivity_fragile)}. A result that lives on a narrow ledge "
                "was fitted to the ledge.",
                {"parameters": list(sensitivity_fragile)},
            )
        )

    if degraded_segments:
        findings.append(
            Finding(
                "DEGRADED_SEGMENT_DATA", Severity.CAUTION,
                "These windows ran on data the quality gate did not pass clean: "
                f"{', '.join(degraded_segments)}. Their numbers describe the data as much "
                "as the strategy.",
                {"segments": list(degraded_segments)},
            )
        )

    return OverfittingAssessment(
        trials=trials,
        findings=tuple(findings),
        observed_sharpe=out_of_sample_sharpe,
        observed_sharpe_per_bar=per_bar,
        expected_max_sharpe_under_null=benchmark,
        deflated_sharpe=deflated,
        in_sample_sharpe=in_sample_sharpe,
        out_of_sample_sharpe=out_of_sample_sharpe,
        degradation=degradation,
        walk_forward_efficiency=walk_forward_efficiency,
    )


def bar_returns(equity_curve: pd.Series) -> pd.Series:
    """Per-bar returns of an equity curve, ready for the Sharpe diagnostics."""
    if equity_curve is None or equity_curve.empty:
        return pd.Series(dtype="float64")
    return equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
