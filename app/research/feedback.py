"""The research-to-strategy feedback loop.

A validation study measures things. This module is what turns those
measurements into a *proposal* a human can read, argue with, and — only if they
explicitly choose to — apply to the running configuration.

Three rules shape everything here.

**Correlation never disables anything.** Two strategies behaving alike on one
window is a hypothesis about redundancy, not a finding about it. The study
already turns each such pair into candidate configurations and tests them; this
module reads what that testing produced. A pair that correlates at 0.95 and
whose down-weighted variant performed *worse* out of sample yields
:attr:`RecommendedAction.NO_ACTION`, and says why. There is no code path from a
correlation coefficient to a disabled strategy, and
:func:`apply_recommendations` refuses to disable a strategy at all.

**Evidence is tiered, and the tier travels with the recommendation.** A variant
that won on the validation window has been observed once, on a window that has
already been looked at repeatedly; a variant that folds chose from their own
training halves and that then held up on their test halves has been observed
several times on data that was not consulted. Those are not the same claim and
are never rendered as the same claim. :class:`EvidenceTier` is what keeps them
apart, and :meth:`Recommendation.is_applicable` gates on it.

**Applying a recommendation is opt-in, and refusing is loud.** The gate is
``research.feedback.enabled``, which ships **false**. With it off, nothing is
applied and the recommendations are reporting output only. With it on, a
recommendation that does not meet the configured evidence bar raises
:class:`FeedbackRefusedError` rather than quietly leaving the weight at its default —
a silent no-op would be indistinguishable from a recommendation that was applied
and happened to change nothing.

Nothing in this module predicts anything. A recommendation says what was
measured and what it would imply; it does not say what will happen next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from app.config.loader import AppConfig, override_config
from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from app.research.report import ValidationReport

logger = get_logger(__name__)

__all__ = [
    "EvidenceTier",
    "FeedbackRefusedError",
    "Recommendation",
    "RecommendationSet",
    "RecommendedAction",
    "apply_recommendations",
    "build_recommendations",
]


class EvidenceTier(StrEnum):
    """How much the evidence behind a recommendation is worth.

    Ordered weakest to strongest. The names describe *which bars produced the
    number*, because that is the only thing that distinguishes them - the
    arithmetic is identical at every tier.
    """

    #: Nothing was measured. No variant ran, or none produced a trade.
    NONE = "NONE"
    #: Measured only on bars the configuration was derived from. Describes the
    #: sample; carries no weight about anything else.
    IN_SAMPLE = "IN_SAMPLE"
    #: One held-out window, already consulted during development. Weak, and
    #: weaker every time it is looked at.
    VALIDATION_WINDOW = "VALIDATION_WINDOW"
    #: Chosen on each fold's training half and scored on its test half, several
    #: times. The strongest tier this platform produces, and still weak.
    WALK_FORWARD = "WALK_FORWARD"


#: Tiers in ascending order of weight, for comparison.
_TIER_ORDER = (
    EvidenceTier.NONE,
    EvidenceTier.IN_SAMPLE,
    EvidenceTier.VALIDATION_WINDOW,
    EvidenceTier.WALK_FORWARD,
)


def tier_rank(tier: EvidenceTier) -> int:
    return _TIER_ORDER.index(tier)


class RecommendedAction(StrEnum):
    """What a recommendation proposes.

    There is deliberately no ``DISABLE``. A study can show that dropping a
    strategy scored better on the windows it was tested on; it cannot show that
    the strategy should not exist, and the difference matters because a disabled
    strategy stops producing the evidence that would overturn the decision.
    """

    #: Change nothing. The evidence did not support acting.
    NO_ACTION = "NO_ACTION"
    #: The evidence supports considering a lower standing weight. Still requires
    #: an explicit configuration change to take effect.
    CONSIDER_DOWN_WEIGHT = "CONSIDER_DOWN_WEIGHT"
    #: Something was flagged but nothing measurable came back.
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class FeedbackRefusedError(RuntimeError):
    """Raised when a recommendation is asked to be applied without the evidence.

    Carries the refusals so a caller can report every one rather than only the
    first.
    """

    def __init__(self, refusals: list[str]) -> None:
        self.refusals = list(refusals)
        super().__init__(
            "The research feedback loop refused to change the configuration:\n"
            + "\n".join(f"  - {r}" for r in refusals)
        )


@dataclass(frozen=True)
class Recommendation:
    """One proposal, the evidence behind it, and what it would take to apply it."""

    subject: str
    action: RecommendedAction
    evidence_tier: EvidenceTier
    strategy: str | None = None
    proposed_weight: float | None = None
    rationale: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    #: Reasons this recommendation cannot be applied as things stand. Empty does
    #: not mean "apply it" - it means the evidence bar was met.
    blockers: tuple[str, ...] = ()

    def is_applicable(self, *, min_tier: EvidenceTier = EvidenceTier.WALK_FORWARD) -> bool:
        """Whether this may be applied to a running configuration.

        Requires an actionable proposal, a weight to apply, no blockers, and
        evidence at or above ``min_tier``. All four, because each of them has a
        way of being true while the others are not.
        """
        return (
            self.action is RecommendedAction.CONSIDER_DOWN_WEIGHT
            and self.proposed_weight is not None
            and not self.blockers
            and tier_rank(self.evidence_tier) >= tier_rank(min_tier)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "action": str(self.action),
            "evidence_tier": str(self.evidence_tier),
            "strategy": self.strategy,
            "proposed_weight": self.proposed_weight,
            "rationale": list(self.rationale),
            "evidence": dict(self.evidence),
            "blockers": list(self.blockers),
            "applicable": self.is_applicable(),
        }

    def render(self) -> str:
        lines = [f"  {self.subject}", f"    action: {self.action}  ({self.evidence_tier})"]
        if self.strategy and self.proposed_weight is not None:
            lines.append(
                f"    proposal: set {self.strategy} standing weight to "
                f"{self.proposed_weight:g}"
            )
        lines += [f"    - {r}" for r in self.rationale]
        if self.blockers:
            lines.append("    not applicable because:")
            lines += [f"      - {b}" for b in self.blockers]
        return "\n".join(lines)


@dataclass
class RecommendationSet:
    """Every recommendation from one study, plus how they were derived."""

    experiment_id: str
    recommendations: list[Recommendation] = field(default_factory=list)
    notes: tuple[str, ...] = ()

    def __iter__(self):
        return iter(self.recommendations)

    def __len__(self) -> int:
        return len(self.recommendations)

    @property
    def applicable(self) -> list[Recommendation]:
        return [r for r in self.recommendations if r.is_applicable()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "notes": list(self.notes),
            "applicable_count": len(self.applicable),
        }

    def render(self) -> str:
        width = 78
        lines = ["-" * width, "RECOMMENDATIONS", "-" * width]
        if not self.recommendations:
            lines.append("  Nothing was flagged, so there is nothing to recommend.")
        else:
            lines.extend(r.render() for r in self.recommendations)
        lines += [
            "",
            "  A recommendation is a reading of what was measured, not an instruction and",
            "  not a forecast. None of these is applied to any running configuration by",
            "  this report; research.feedback.enabled governs that and ships false.",
        ]
        if self.notes:
            lines += ["", "  Notes:"] + [f"    - {n}" for n in self.notes]
        return "\n".join(lines)


# ------------------------------------------------------------------- building


def _variant_target(variant: str) -> tuple[str | None, str | None]:
    """Split a variant name into (kind, strategy). ``weight_variants`` names them."""
    for prefix, kind in (("halve_", "halve"), ("drop_", "drop")):
        if variant.startswith(prefix):
            return kind, variant[len(prefix) :]
    return None, None


def _walk_forward_evidence(report: ValidationReport) -> dict[str, dict[str, Any]]:
    """Per-variant out-of-sample record, from the folds that chose each one.

    A fold's test half is out of sample *for that fold's choice*, which is what
    makes this the strongest tier available. Folds are grouped by the variant
    their training half selected; a variant no fold ever chose has no record
    here, which is itself informative.
    """
    if report.walk_forward is None or not report.walk_forward.folds:
        return {}

    grouped: dict[str, dict[str, Any]] = {}
    for fold in report.walk_forward.folds:
        row = grouped.setdefault(
            fold.selected_variant,
            {"folds": 0, "trades": 0, "expectancy_r": [], "profitable_folds": 0},
        )
        row["folds"] += 1
        row["trades"] += fold.test.metrics.total_trades
        row["expectancy_r"].append(fold.test.metrics.expectancy_r)
        if fold.test.metrics.expectancy_r > 0:
            row["profitable_folds"] += 1

    for row in grouped.values():
        values = row.pop("expectancy_r")
        row["mean_expectancy_r"] = sum(values) / len(values) if values else 0.0
    return grouped


def _validation_evidence(report: ValidationReport) -> dict[str, dict[str, Any]]:
    """Per-variant record from the one-window comparison, keyed by variant name."""
    return {
        str(row.get("variant")): dict(row)
        for row in report.variant_rows
        if row.get("variant")
    }


def build_recommendations(
    report: ValidationReport,
    *,
    min_out_of_sample_trades: int = 30,
    min_walk_forward_folds: int = 3,
) -> RecommendationSet:
    """Turn a study's findings into interpretable proposals.

    Walks every redundancy finding the correlation analysis produced and asks,
    for each, what the candidate configurations actually did on data that did
    not suggest them. The answer is usually "not enough happened to tell", and
    that is reported as such rather than resolved in favour of acting.

    Args:
        report: a completed :class:`~app.research.report.ValidationReport`.
        min_out_of_sample_trades: below this many out-of-sample trades the study
            declines to conclude anything, matching the report's own threshold.
        min_walk_forward_folds: folds needed before fold agreement counts as
            repetition rather than coincidence.
    """
    recommendations: list[Recommendation] = []
    notes: list[str] = []

    oos = report.out_of_sample
    oos_trades = oos.metrics.total_trades if oos else 0
    wf_evidence = _walk_forward_evidence(report)
    validation_evidence = _validation_evidence(report)
    baseline_wf = wf_evidence.get("baseline")

    correlation = report.correlation
    if correlation is None or not correlation.findings:
        notes.append(
            "No strategy pair exceeded the correlation threshold, so no redundancy "
            "hypothesis was raised and there is nothing to weigh."
        )
        return RecommendationSet(report.experiment_id, [], tuple(notes))

    for finding in correlation.findings:
        a, b = finding.strategy_a, finding.strategy_b
        counts = correlation.trades_per_strategy
        target = a if counts.get(a, 0) <= counts.get(b, 0) else b
        subject = f"{a} / {b} redundancy"

        base_rationale = [
            str(finding),
            (
                f"{target} has the thinner evidence base of the two "
                f"({counts.get(target, 0):,} trades), so the candidate configurations "
                "tilt away from it. That is not a judgement that it is the worse "
                "strategy."
            ),
        ]
        evidence: dict[str, Any] = {
            "return_correlation": finding.return_correlation,
            "signal_agreement": finding.signal_agreement,
            "observations": finding.overlapping_observations,
            "trades_per_strategy": {a: counts.get(a, 0), b: counts.get(b, 0)},
            "out_of_sample_trades": oos_trades,
            "measured_on": correlation.segment,
        }

        # --- what did the candidates actually do? ----------------------------
        halve = f"halve_{target}"
        wf_row = wf_evidence.get(halve)
        val_row = validation_evidence.get(halve)
        base_val = validation_evidence.get("baseline")

        blockers: list[str] = []

        # A pair where neither strategy traded is agreeing on inaction, not
        # duplicating a bet. The detector counts a shared WAIT as agreement,
        # which is right in general - two strategies that wait together nine
        # bars in ten are one opinion with two names - but at zero trades each
        # there is no exposure to reduce and no weight change that could be
        # tested. Reported as its own outcome rather than resolved into a
        # variant comparison that would compare nothing with nothing.
        traded = counts.get(a, 0) + counts.get(b, 0)
        if traded == 0:
            recommendations.append(
                Recommendation(
                    subject=subject,
                    action=RecommendedAction.INSUFFICIENT_EVIDENCE,
                    evidence_tier=EvidenceTier.NONE,
                    strategy=None,
                    proposed_weight=None,
                    rationale=(
                        str(finding),
                        f"Neither {a} nor {b} opened a position in this window, so their "
                        "agreement is agreement on inaction. There is no exposure to "
                        "reduce and no weight change that could be measured.",
                        "This is a limitation of the window, not a finding about either "
                        "strategy.",
                    ),
                    evidence={**evidence, "both_strategies_idle": True},
                    blockers=(
                        "Neither strategy traded in the window the pair was measured on.",
                    ),
                )
            )
            continue

        if oos_trades < min_out_of_sample_trades:
            blockers.append(
                f"The out-of-sample window produced {oos_trades} trades, below the "
                f"{min_out_of_sample_trades} this study requires before concluding "
                "anything. Nothing measured on it is stable enough to act on."
            )

        # Tier 1: walk-forward, if any fold ever chose the variant.
        if wf_row and wf_row["folds"] >= min_walk_forward_folds:
            tier = EvidenceTier.WALK_FORWARD
            improved = (
                baseline_wf is None
                or wf_row["mean_expectancy_r"] > baseline_wf["mean_expectancy_r"]
            )
            evidence["walk_forward"] = dict(wf_row)
            if baseline_wf:
                evidence["walk_forward_baseline"] = dict(baseline_wf)
            rationale = base_rationale + [
                f"{wf_row['folds']} of the walk-forward folds selected {halve} from their "
                f"own training half; those folds averaged "
                f"{wf_row['mean_expectancy_r']:+.3f}R on their test halves."
            ]
            if improved and wf_row["trades"] > 0:
                action = RecommendedAction.CONSIDER_DOWN_WEIGHT
                rationale.append(
                    "That is better than the folds that kept the baseline, on data those "
                    "folds had not seen when they chose. It is the strongest evidence "
                    "this platform produces and it is still one instrument, one history."
                )
            else:
                action = RecommendedAction.NO_ACTION
                rationale.append(
                    "That is not better than the folds that kept the baseline, so the "
                    "redundancy did not cost anything measurable. Nothing is changed."
                )
                blockers.append("The variant did not outperform the baseline out of sample.")

        # Tier 2: the one-window comparison.
        elif val_row is not None and base_val is not None:
            tier = EvidenceTier.VALIDATION_WINDOW
            delta = float(val_row.get("expectancy_r", 0)) - float(
                base_val.get("expectancy_r", 0)
            )
            delta_dd = float(val_row.get("max_drawdown", 0)) - float(
                base_val.get("max_drawdown", 0)
            )
            evidence["validation_window"] = {
                "variant": dict(val_row),
                "baseline": dict(base_val),
                "expectancy_delta": delta,
                "drawdown_delta": delta_dd,
            }
            action = RecommendedAction.NO_ACTION
            rationale = base_rationale + [
                f"On the validation window {halve} changed expectancy by {delta:+.3f}R and "
                f"drawdown by {delta_dd:+.1%} against the baseline.",
                "One window, and one that has already been consulted during development. "
                "That is not enough to move a weight, whichever way it points.",
            ]
            blockers.append(
                "The only comparison available is a single validation window, which is "
                "below the walk-forward evidence bar."
            )

        # Tier 3: nothing ran.
        else:
            tier = EvidenceTier.NONE
            action = RecommendedAction.INSUFFICIENT_EVIDENCE
            rationale = base_rationale + [
                f"No comparison of {halve} against the baseline is available - either the "
                "variant was never evaluated, or it produced no trades where it was."
            ]
            blockers.append("No candidate comparison was produced for this pair.")

        recommendations.append(
            Recommendation(
                subject=subject,
                action=action,
                evidence_tier=tier,
                strategy=target,
                proposed_weight=(
                    0.5 if action is RecommendedAction.CONSIDER_DOWN_WEIGHT else None
                ),
                rationale=tuple(rationale),
                evidence=evidence,
                blockers=tuple(blockers),
            )
        )

    notes.append(
        "No strategy is disabled on the strength of a correlation. The candidates a "
        "redundancy hypothesis suggests are evaluated, and the outcome of that "
        "evaluation is what appears above."
    )
    return RecommendationSet(report.experiment_id, recommendations, tuple(notes))


# -------------------------------------------------------------------- applying


def apply_recommendations(
    config: AppConfig,
    recommendations: RecommendationSet,
    *,
    enabled: bool | None = None,
    min_tier: EvidenceTier = EvidenceTier.WALK_FORWARD,
    max_weight_reduction: float = 0.5,
    strict: bool = True,
) -> tuple[AppConfig, list[str]]:
    """Apply validated recommendations to ``config``, or refuse and say why.

    This is the only route from a research finding to a running configuration,
    and it is shut by default.

    Args:
        config: the configuration to derive from. Never mutated.
        recommendations: the set produced by :func:`build_recommendations`.
        enabled: override the ``research.feedback.enabled`` gate. Reading it from
            configuration is the normal path; the argument exists so a test can
            be explicit about which gate it is exercising.
        min_tier: the evidence tier a recommendation must reach.
        max_weight_reduction: the most a standing weight may be reduced, as a
            fraction. A recommendation proposing a larger cut is refused rather
            than clamped - clamping would apply a change nobody authorised.
        strict: raise :class:`FeedbackRefusedError` when an actionable recommendation
            cannot be applied. With ``False`` the refusals are returned and
            logged instead, for a caller that wants to report them all.

    Returns:
        ``(config, applied)`` where ``applied`` describes each change made. An
        empty list means nothing was applied, which is the expected outcome.

    Raises:
        FeedbackRefusedError: under ``strict`` when the gate is off but changes were
            asked for, or when a recommendation fails the evidence bar.
    """
    settings = getattr(config.research, "feedback", None)
    if enabled is None:
        enabled = bool(getattr(settings, "enabled", False))

    candidates = [
        r for r in recommendations if r.action is RecommendedAction.CONSIDER_DOWN_WEIGHT
    ]

    if not enabled:
        message = (
            "research.feedback.enabled is false, so no research finding was applied to "
            f"the configuration. {len(candidates)} recommendation(s) would have been "
            "considered had it been on."
        )
        logger.info("Research feedback gate is closed", extra={"candidates": len(candidates)})
        return config, [] if not candidates else [message]

    refusals: list[str] = []
    applied: list[str] = []
    updated = config

    for rec in recommendations:
        if rec.action is not RecommendedAction.CONSIDER_DOWN_WEIGHT:
            continue

        if not rec.is_applicable(min_tier=min_tier):
            reasons = ", ".join(rec.blockers) or (
                f"evidence tier {rec.evidence_tier} is below the required {min_tier}"
            )
            refusals.append(f"{rec.subject}: {reasons}")
            continue

        strategy = rec.strategy
        if strategy not in updated.strategies.strategies:
            refusals.append(
                f"{rec.subject}: {strategy!r} is not in the configured strategy roster."
            )
            continue

        block = updated.strategies.strategies[strategy]
        proposed = float(rec.proposed_weight)
        if proposed <= 0:
            refusals.append(
                f"{rec.subject}: a proposed weight of {proposed:g} would disable "
                f"{strategy}. This loop does not disable strategies - a disabled strategy "
                "stops producing the evidence that would overturn the decision."
            )
            continue

        reduction = 1.0 - (proposed / block.weight) if block.weight else 1.0
        if reduction > max_weight_reduction + 1e-9:
            refusals.append(
                f"{rec.subject}: reducing {strategy} from {block.weight:g} to "
                f"{proposed:g} is a {reduction:.0%} cut, beyond the "
                f"{max_weight_reduction:.0%} this loop will apply. Refused rather than "
                "clamped, because a clamped value is a change nobody authorised."
            )
            continue

        strategies = dict(updated.strategies.strategies)
        strategies[strategy] = block.model_copy(update={"weight": proposed})
        updated = override_config(
            updated,
            strategies=updated.strategies.model_copy(update={"strategies": strategies}),
        )
        applied.append(
            f"{strategy}: standing weight {block.weight:g} -> {proposed:g} "
            f"({rec.evidence_tier} evidence, experiment {recommendations.experiment_id})"
        )
        logger.info(
            "Applied a validated research recommendation",
            extra={
                "strategy": strategy,
                "weight": proposed,
                "tier": str(rec.evidence_tier),
                "experiment_id": recommendations.experiment_id,
            },
        )

    if refusals:
        # Visible either way. A silent no-op here is indistinguishable from a
        # change that was applied and happened to alter nothing.
        for refusal in refusals:
            logger.warning("Research feedback refused", extra={"detail": refusal})
        if strict:
            raise FeedbackRefusedError(refusals)

    return updated, applied + [f"REFUSED: {r}" for r in refusals]
