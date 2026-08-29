"""The research-to-strategy feedback loop.

The invariants under test are the ones whose violation would look like ordinary
behaviour rather than like a bug.

**A correlation must never, by itself, change anything.** The failure mode is a
loop that quietly disables a strategy because it looked like another one on a
window, and then never gathers the evidence that would have overturned that. The
tests below drive a 0.98 correlation through the whole path and assert the
configuration comes out identical.

**In-sample, validation-window and per-fold out-of-sample evidence must never be
conflated.** They are the same arithmetic on different bars, so nothing about a
number reveals which it is. :class:`EvidenceTier` carries that, and a
recommendation resting on one window must not be applicable at the walk-forward
bar.

**Refusing must be loud.** A gate that silently declines is indistinguishable
from one that applied a change which happened to alter nothing, so the refusal
path is asserted to raise and to name the reason.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.backtest.metrics import PerformanceMetrics
from app.backtest.results import BacktestResult, RunProvenance
from app.config.loader import get_config, override_config
from app.research.correlation import CorrelationReport, RedundancyFinding
from app.research.feedback import (
    EvidenceTier,
    FeedbackRefusedError,
    Recommendation,
    RecommendedAction,
    apply_recommendations,
    build_recommendations,
    tier_rank,
)
from app.research.harness import SegmentRun
from app.research.report import ValidationReport
from app.research.splits import Segment, SegmentRole
from app.research.walk_forward import FoldResult, WalkForwardReport

# --------------------------------------------------------------------- fixtures


def _segment(name: str, role: SegmentRole, start: str = "2016-01-01") -> Segment:
    index = pd.bdate_range(start=start, periods=300, tz="UTC")
    return Segment(
        name=name,
        role=role,
        start=0,
        end=len(index),
        data_start=0,
        start_time=index[0],
        end_time=index[-1],
    )


def _run(
    name: str,
    role: SegmentRole,
    *,
    trades: int,
    expectancy: float,
    variant: str = "baseline",
) -> SegmentRun:
    metrics = PerformanceMetrics(
        total_trades=trades, expectancy_r=expectancy, sortino_ratio=expectancy * 2
    )
    return SegmentRun(
        segment=_segment(name, role),
        variant=variant,
        metrics=metrics,
        trades=pd.DataFrame(),
        equity_curve=pd.Series([10_000.0, 10_100.0]),
        data_quality="PASS",
        data_quality_source="caller",
        result=BacktestResult(
            provenance=RunProvenance(
                experiment_id="RES-feedbacktest",
                symbol="XAUUSD",
                timeframe="1D",
                start=None,
                end=None,
                bars=300,
                data_provider="synthetic",
                data_checksum="abc123",
                data_quality="PASS",
            ),
            metrics=metrics,
            trades=pd.DataFrame(),
            equity_curve=pd.Series([10_000.0, 10_100.0]),
            snapshots=pd.DataFrame(),
        ),
    )


def _correlation(pair=("momentum", "trend_following"), *, correlation=0.98) -> CorrelationReport:
    a, b = pair
    return CorrelationReport(
        segment="in_sample",
        trades_per_strategy={a: 40, b: 90},
        findings=(
            RedundancyFinding(
                strategy_a=a,
                strategy_b=b,
                return_correlation=correlation,
                signal_agreement=0.93,
                overlapping_observations=520,
                hypothesis=f"{a} may be redundant with {b}",
            ),
        ),
        threshold=0.7,
    )


def _report(
    *,
    oos_trades: int = 60,
    correlation: CorrelationReport | None = None,
    walk_forward: WalkForwardReport | None = None,
    variant_rows: list[dict] | None = None,
) -> ValidationReport:
    report = ValidationReport(
        experiment_id="RES-feedbacktest", symbol="XAUUSD", timeframe="1D"
    )
    report.segments = [
        _run("in_sample", SegmentRole.TRAIN, trades=120, expectancy=0.30),
        _run("validation", SegmentRole.VALIDATION, trades=40, expectancy=0.10),
        _run("out_of_sample", SegmentRole.TEST, trades=oos_trades, expectancy=0.05),
    ]
    report.correlation = correlation if correlation is not None else _correlation()
    report.walk_forward = walk_forward
    report.variant_rows = variant_rows or []
    return report


def _folds(selected: dict[str, tuple[int, float]]) -> WalkForwardReport:
    """Walk-forward folds, ``{variant: (fold_count, test expectancy)}``."""
    folds: list[FoldResult] = []
    index = 0
    for variant, (count, expectancy) in selected.items():
        for _ in range(count):
            folds.append(
                FoldResult(
                    index=index,
                    train=_run("train", SegmentRole.TRAIN, trades=50, expectancy=0.2),
                    test=_run("test", SegmentRole.TEST, trades=20, expectancy=expectancy),
                    selected_variant=variant,
                    selection_score=0.4,
                    candidates_considered=3,
                    selection_note="chosen on the training half",
                )
            )
            index += 1
    return WalkForwardReport(folds=folds)


# ------------------------------------------- correlation never disables anything


def test_a_strong_correlation_alone_recommends_no_action():
    """0.98 correlation, no candidate evidence: nothing is proposed."""
    recommendations = build_recommendations(_report(walk_forward=None, variant_rows=[]))

    assert len(recommendations) == 1
    rec = recommendations.recommendations[0]
    assert rec.action is RecommendedAction.INSUFFICIENT_EVIDENCE
    assert rec.evidence_tier is EvidenceTier.NONE
    assert rec.proposed_weight is None
    assert not rec.is_applicable()
    assert recommendations.applicable == []
    assert any("No comparison" in r for r in rec.rationale)


def test_no_recommendation_ever_disables_a_strategy():
    """There is no action that removes a strategy, at any evidence tier."""
    assert not hasattr(RecommendedAction, "DISABLE")
    assert {str(a) for a in RecommendedAction} == {
        "NO_ACTION",
        "CONSIDER_DOWN_WEIGHT",
        "INSUFFICIENT_EVIDENCE",
    }

    # Even the strongest possible evidence yields a weight change, never removal.
    report = _report(walk_forward=_folds({"halve_momentum": (5, 0.40), "baseline": (2, 0.01)}))
    rec = build_recommendations(report).recommendations[0]
    assert rec.action is RecommendedAction.CONSIDER_DOWN_WEIGHT
    assert rec.proposed_weight is not None and rec.proposed_weight > 0


def test_applying_a_zero_weight_is_refused_as_a_disable(config_dir):
    """A proposal that would zero a weight is refused, not clamped or applied."""
    config = get_config(config_dir)
    rec = Recommendation(
        subject="momentum / trend_following redundancy",
        action=RecommendedAction.CONSIDER_DOWN_WEIGHT,
        evidence_tier=EvidenceTier.WALK_FORWARD,
        strategy="momentum",
        proposed_weight=0.0,
    )
    from app.research.feedback import RecommendationSet

    with pytest.raises(FeedbackRefusedError) as excinfo:
        apply_recommendations(
            config, RecommendationSet("RES-x", [rec]), enabled=True
        )

    assert "does not disable strategies" in str(excinfo.value)
    assert config.strategies.strategies["momentum"].weight == 1.0


def test_a_worse_variant_produces_no_action_despite_high_correlation():
    """Redundancy that cost nothing measurable leaves the configuration alone."""
    report = _report(
        walk_forward=_folds({"halve_momentum": (4, -0.20), "baseline": (3, 0.30)})
    )
    rec = build_recommendations(report).recommendations[0]

    assert rec.action is RecommendedAction.NO_ACTION
    assert rec.evidence_tier is EvidenceTier.WALK_FORWARD, (
        "the tier records where the evidence came from, not whether it was favourable"
    )
    assert not rec.is_applicable()
    assert any("did not outperform" in b for b in rec.blockers)
    assert any("Nothing is changed" in r for r in rec.rationale)


# ----------------------------------------------- IS / validation / OOS separation


def test_evidence_tiers_are_ordered_weakest_to_strongest():
    assert tier_rank(EvidenceTier.NONE) < tier_rank(EvidenceTier.IN_SAMPLE)
    assert tier_rank(EvidenceTier.IN_SAMPLE) < tier_rank(EvidenceTier.VALIDATION_WINDOW)
    assert tier_rank(EvidenceTier.VALIDATION_WINDOW) < tier_rank(EvidenceTier.WALK_FORWARD)


def test_a_validation_window_win_does_not_reach_the_walk_forward_bar():
    """One window is recorded as one window and cannot be applied as more.

    This is the invariant that stops a validation-window result being read as
    out-of-sample evidence. The numbers are identical in kind; only the tier
    distinguishes them.
    """
    report = _report(
        walk_forward=None,
        variant_rows=[
            {"variant": "baseline", "expectancy_r": 0.05, "max_drawdown": 0.10, "trades": 40},
            # A large apparent improvement, on one already-consulted window.
            {"variant": "halve_momentum", "expectancy_r": 0.60, "max_drawdown": 0.04,
             "trades": 38},
        ],
    )
    rec = build_recommendations(report).recommendations[0]

    assert rec.evidence_tier is EvidenceTier.VALIDATION_WINDOW
    assert rec.action is RecommendedAction.NO_ACTION, (
        "a single validation window must not produce an actionable recommendation "
        "however large the apparent improvement"
    )
    assert not rec.is_applicable(min_tier=EvidenceTier.WALK_FORWARD)
    assert any("single validation window" in b for b in rec.blockers)
    assert rec.evidence["validation_window"]["expectancy_delta"] == pytest.approx(0.55)


def test_walk_forward_evidence_is_labelled_as_per_fold_out_of_sample():
    report = _report(
        walk_forward=_folds({"halve_momentum": (4, 0.35), "baseline": (3, 0.02)})
    )
    rec = build_recommendations(report).recommendations[0]

    assert rec.evidence_tier is EvidenceTier.WALK_FORWARD
    assert rec.action is RecommendedAction.CONSIDER_DOWN_WEIGHT
    assert rec.is_applicable()
    assert rec.evidence["walk_forward"]["folds"] == 4
    assert rec.evidence["walk_forward_baseline"]["folds"] == 3
    assert rec.evidence["walk_forward"]["mean_expectancy_r"] == pytest.approx(0.35)
    assert any("their own training half" in r for r in rec.rationale)


def test_too_few_folds_does_not_qualify_as_walk_forward_evidence():
    """Two folds agreeing is coincidence, and is treated as such."""
    report = _report(walk_forward=_folds({"halve_momentum": (2, 0.50)}))
    rec = build_recommendations(report).recommendations[0]

    assert rec.evidence_tier is not EvidenceTier.WALK_FORWARD
    assert not rec.is_applicable()


def test_thin_out_of_sample_evidence_blocks_every_recommendation():
    """Below the trade threshold nothing is actionable, whatever the folds say."""
    report = _report(
        oos_trades=7,  # the number the real study produced
        walk_forward=_folds({"halve_momentum": (5, 0.40), "baseline": (2, 0.01)}),
    )
    rec = build_recommendations(report, min_out_of_sample_trades=30).recommendations[0]

    assert rec.action is RecommendedAction.CONSIDER_DOWN_WEIGHT
    assert not rec.is_applicable(), "7 out-of-sample trades cannot support a config change"
    assert any("below the 30" in b for b in rec.blockers)
    assert rec.evidence["out_of_sample_trades"] == 7


# --------------------------------------------------------------- the gate itself


def test_the_gate_ships_closed(config_dir):
    assert get_config(config_dir).research.feedback.enabled is False


def test_nothing_is_applied_while_the_gate_is_closed(config_dir):
    """With the gate shut the configuration is returned untouched, and it says so."""
    config = get_config(config_dir)
    report = _report(walk_forward=_folds({"halve_momentum": (5, 0.40), "baseline": (2, 0.01)}))
    recommendations = build_recommendations(report)
    assert recommendations.applicable, "the fixture should offer something to refuse"

    updated, applied = apply_recommendations(config, recommendations)

    assert updated is config
    assert updated.strategies.strategies["momentum"].weight == 1.0
    assert any("research.feedback.enabled is false" in a for a in applied), (
        "a closed gate must announce itself, not return an empty list that looks "
        "like there was nothing to apply"
    )


def test_an_open_gate_applies_only_walk_forward_backed_recommendations(config_dir):
    config = get_config(config_dir)
    report = _report(walk_forward=_folds({"halve_momentum": (5, 0.40), "baseline": (2, 0.01)}))

    updated, applied = apply_recommendations(
        config, build_recommendations(report), enabled=True
    )

    assert updated is not config
    assert updated.strategies.strategies["momentum"].weight == 0.5
    assert config.strategies.strategies["momentum"].weight == 1.0, "input config mutated"
    assert len(applied) == 1
    assert "standing weight 1 -> 0.5" in applied[0]
    assert "WALK_FORWARD" in applied[0]
    assert "RES-feedbacktest" in applied[0], "provenance must travel with the change"


def test_an_open_gate_refuses_loudly_rather_than_silently_skipping(config_dir):
    """Insufficient evidence raises; it does not quietly leave the default in place."""
    config = get_config(config_dir)
    report = _report(
        oos_trades=7,
        walk_forward=_folds({"halve_momentum": (5, 0.40), "baseline": (2, 0.01)}),
    )

    with pytest.raises(FeedbackRefusedError) as excinfo:
        apply_recommendations(config, build_recommendations(report), enabled=True)

    assert excinfo.value.refusals, "the refusal must name its reason"
    assert "below the 30" in str(excinfo.value)
    assert config.strategies.strategies["momentum"].weight == 1.0


def test_non_strict_mode_reports_refusals_instead_of_raising(config_dir):
    config = get_config(config_dir)
    report = _report(oos_trades=7, walk_forward=_folds({"halve_momentum": (5, 0.4)}))

    updated, applied = apply_recommendations(
        config, build_recommendations(report), enabled=True, strict=False
    )

    assert updated.strategies.strategies["momentum"].weight == 1.0
    assert any(a.startswith("REFUSED:") for a in applied), (
        "non-strict must still surface the refusal, just without raising"
    )


def test_a_cut_beyond_the_configured_maximum_is_refused_not_clamped(config_dir):
    """Clamping would apply a change nobody authorised."""
    from app.research.feedback import RecommendationSet

    config = get_config(config_dir)
    rec = Recommendation(
        subject="momentum / trend_following redundancy",
        action=RecommendedAction.CONSIDER_DOWN_WEIGHT,
        evidence_tier=EvidenceTier.WALK_FORWARD,
        strategy="momentum",
        proposed_weight=0.1,  # a 90% cut
    )

    with pytest.raises(FeedbackRefusedError) as excinfo:
        apply_recommendations(
            config,
            RecommendationSet("RES-x", [rec]),
            enabled=True,
            max_weight_reduction=0.5,
        )

    assert "Refused rather than clamped" in str(excinfo.value)
    assert config.strategies.strategies["momentum"].weight == 1.0


def test_a_recommendation_for_an_unknown_strategy_is_refused(config_dir):
    from app.research.feedback import RecommendationSet

    config = get_config(config_dir)
    rec = Recommendation(
        subject="ghost / momentum redundancy",
        action=RecommendedAction.CONSIDER_DOWN_WEIGHT,
        evidence_tier=EvidenceTier.WALK_FORWARD,
        strategy="not_a_strategy",
        proposed_weight=0.5,
    )

    with pytest.raises(FeedbackRefusedError) as excinfo:
        apply_recommendations(config, RecommendationSet("RES-x", [rec]), enabled=True)
    assert "not in the configured strategy roster" in str(excinfo.value)


# ------------------------------------------------------------------- provenance


def test_recommendations_carry_their_experiment_and_evidence():
    report = _report(walk_forward=_folds({"halve_momentum": (4, 0.35), "baseline": (3, 0.02)}))
    recommendations = build_recommendations(report)
    payload = recommendations.to_dict()

    assert payload["experiment_id"] == "RES-feedbacktest"
    assert payload["applicable_count"] == 1
    rec = payload["recommendations"][0]
    assert rec["evidence_tier"] == "WALK_FORWARD"
    assert rec["evidence"]["return_correlation"] == pytest.approx(0.98)
    assert rec["evidence"]["measured_on"] == "in_sample", (
        "the window the correlation was measured on must travel with the finding"
    )
    assert rec["evidence"]["trades_per_strategy"] == {"momentum": 40, "trend_following": 90}
    assert rec["rationale"], "a recommendation with no stated reasoning is not interpretable"


def test_no_findings_yields_an_explicit_empty_set():
    report = _report(correlation=CorrelationReport(segment="in_sample", findings=()))
    recommendations = build_recommendations(report)

    assert len(recommendations) == 0
    assert any("no redundancy hypothesis was raised" in n for n in recommendations.notes)


def test_render_states_that_nothing_was_applied():
    report = _report(walk_forward=_folds({"halve_momentum": (4, 0.35), "baseline": (3, 0.02)}))
    text = build_recommendations(report).render()

    assert "RECOMMENDATIONS" in text
    assert "None of these is applied to any running configuration" in text
    assert "research.feedback.enabled" in text
    assert "not a forecast" in text


def test_applied_weight_actually_reaches_the_selector(config_dir):
    """A applied weight must change what the selector does, or it changed nothing.

    Closes the loop: the point of the bridge is the selector's ``base`` weight,
    so this asserts against the selector rather than against the config object.
    """
    from app.regimes.models import MarketRegime, RegimeType, VolatilityState
    from app.signals.selector import StrategySelector
    from app.strategies.registry import build_enabled_strategies

    config = get_config(config_dir)
    report = _report(walk_forward=_folds({"halve_momentum": (5, 0.40), "baseline": (2, 0.01)}))
    updated, applied = apply_recommendations(
        config, build_recommendations(report), enabled=True
    )
    assert applied

    regime = MarketRegime(
        regime=RegimeType.TRENDING_UP,
        confidence=1.0,
        volatility=VolatilityState.MEDIUM,
        trend_strength=0.8,
    )
    before = StrategySelector().select(build_enabled_strategies(config.strategies), regime)
    after = StrategySelector().select(build_enabled_strategies(updated.strategies), regime)

    assert before["momentum"].base_weight == 1.0
    assert after["momentum"].base_weight == 0.5
    assert after["momentum"].weight < before["momentum"].weight
    assert any("standing weight 0.5" in r for r in after["momentum"].reasoning), (
        "the selector must say that a non-default standing weight is in play"
    )


def test_disabled_gate_is_the_default_path_through_a_real_config(config_dir):
    """The end-to-end default: findings exist, nothing is applied, and it is stated."""
    config = get_config(config_dir)
    assert config.research.feedback.enabled is False

    report = _report(walk_forward=_folds({"halve_momentum": (5, 0.40), "baseline": (2, 0.01)}))
    recommendations = build_recommendations(report)
    updated, applied = apply_recommendations(config, recommendations)

    assert updated.strategies.strategies["momentum"].weight == 1.0
    assert recommendations.applicable, "there was something applicable, and it was not applied"
    assert applied and "false" in applied[0]


def test_override_config_does_not_leak_between_studies(config_dir):
    """Applying to one config must not affect another derived from the same base."""
    base = get_config(config_dir)
    report = _report(walk_forward=_folds({"halve_momentum": (5, 0.40), "baseline": (2, 0.01)}))
    recommendations = build_recommendations(report)

    first, _ = apply_recommendations(base, recommendations, enabled=True)
    second = override_config(base)

    assert first.strategies.strategies["momentum"].weight == 0.5
    assert second.strategies.strategies["momentum"].weight == 1.0
    assert base.strategies.strategies["momentum"].weight == 1.0


# ------------------------------------------ agreement on inaction is not redundancy


def test_a_pair_that_never_traded_is_not_treated_as_redundant():
    """Two strategies with zero trades agree because both always waited.

    A real study surfaced this: three pairs flagged at 100% signal agreement
    where two of the strategies had opened no position at all. The detector is
    right that a shared WAIT is agreement — two strategies waiting together nine
    bars in ten are one opinion with two names — but at zero trades each there is
    no exposure to reduce, and a variant comparison would be comparing nothing
    with nothing. The giveaway in that study was a variant that "changed
    expectancy by +0.000R".
    """
    correlation = CorrelationReport(
        segment="in_sample",
        trades_per_strategy={"mean_reversion": 0, "support_resistance": 0},
        findings=(
            RedundancyFinding(
                strategy_a="mean_reversion",
                strategy_b="support_resistance",
                return_correlation=None,
                signal_agreement=1.0,
                overlapping_observations=1851,
                hypothesis="they say the same thing on every bar",
            ),
        ),
        threshold=0.7,
    )
    report = _report(
        correlation=correlation,
        walk_forward=_folds({"halve_mean_reversion": (5, 0.9), "baseline": (2, 0.0)}),
        variant_rows=[
            {"variant": "baseline", "expectancy_r": 0.05, "max_drawdown": 0.1},
            {"variant": "halve_mean_reversion", "expectancy_r": 0.05, "max_drawdown": 0.1},
        ],
    )

    rec = build_recommendations(report).recommendations[0]

    assert rec.action is RecommendedAction.INSUFFICIENT_EVIDENCE, (
        "a pair that never traded must not produce an actionable proposal, however "
        "strong the walk-forward numbers attached to its variant name"
    )
    assert rec.evidence_tier is EvidenceTier.NONE
    assert rec.proposed_weight is None
    assert rec.strategy is None
    assert not rec.is_applicable()
    assert rec.evidence["both_strategies_idle"] is True
    assert any("agreement on inaction" in r for r in rec.rationale)
    assert any("limitation of the window" in r for r in rec.rationale)


def test_a_pair_where_one_side_traded_is_still_evaluated_normally():
    """The guard is for pairs with no trades at all, not for thin ones."""
    correlation = CorrelationReport(
        segment="in_sample",
        trades_per_strategy={"momentum": 0, "trend_following": 40},
        findings=(
            RedundancyFinding(
                strategy_a="momentum",
                strategy_b="trend_following",
                return_correlation=0.91,
                signal_agreement=0.95,
                overlapping_observations=500,
                hypothesis="h",
            ),
        ),
        threshold=0.7,
    )
    report = _report(
        correlation=correlation,
        walk_forward=_folds({"halve_momentum": (4, 0.30), "baseline": (3, 0.01)}),
    )
    rec = build_recommendations(report).recommendations[0]

    assert rec.action is RecommendedAction.CONSIDER_DOWN_WEIGHT
    assert rec.evidence.get("both_strategies_idle") is None


def test_the_correlation_report_notes_idle_pairs():
    """The caveat is stated where the finding is, not only where it is consumed."""
    from app.research.correlation import analyse_correlations

    decisions = pd.DataFrame(
        {
            "signal_mean_reversion": ["WAIT"] * 200,
            "signal_support_resistance": ["WAIT"] * 200,
        }
    )
    report = analyse_correlations(
        trades=pd.DataFrame(),
        decisions=decisions,
        segment="in_sample",
        isolated_trades={"mean_reversion": 0, "support_resistance": 0},
        isolated_equity={
            # Flat: neither strategy ever opened a position.
            "mean_reversion": pd.Series([10_000.0] * 200),
            "support_resistance": pd.Series([10_000.0] * 200),
        },
    )

    assert report.findings, "the fixture should still flag the pair"
    assert any("agreement on inaction" in n for n in report.notes)
    assert any("no exposure to reduce" in n for n in report.notes)
