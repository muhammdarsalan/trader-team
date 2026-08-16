"""Risk decision types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RiskVerdict(StrEnum):
    APPROVED = "APPROVED"
    REDUCED = "REDUCED"   # allowed, but at a smaller size than requested
    REJECTED = "REJECTED"


class RiskBlockReason(StrEnum):
    """Why a trade was refused. Recorded so no-trade decisions are auditable."""

    KILL_SWITCH = "KILL_SWITCH"
    TRADING_DISABLED = "TRADING_DISABLED"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_POSITIONS = "MAX_POSITIONS"
    MAX_POSITIONS_PER_SYMBOL = "MAX_POSITIONS_PER_SYMBOL"
    PORTFOLIO_RISK = "PORTFOLIO_RISK"
    STRATEGY_RISK = "STRATEGY_RISK"
    CORRELATED_RISK = "CORRELATED_RISK"
    PORTFOLIO_NOTIONAL = "PORTFOLIO_NOTIONAL"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    INVALID_STOP = "INVALID_STOP"
    POSITION_TOO_SMALL = "POSITION_TOO_SMALL"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass(frozen=True)
class RiskDecision:
    """The risk engine's verdict on one proposed trade.

    Carries the full arithmetic, not just the answer. Section 15 of the brief
    requires position sizing to be explicit and testable, and a number nobody
    can reproduce by hand is neither.
    """

    verdict: RiskVerdict
    quantity: float = 0.0
    risk_amount: float = 0.0
    risk_per_unit: float = 0.0
    notional: float = 0.0
    block_reason: RiskBlockReason | None = None
    reasoning: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.verdict in {RiskVerdict.APPROVED, RiskVerdict.REDUCED}

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": str(self.verdict),
            "quantity": self.quantity,
            "risk_amount": self.risk_amount,
            "risk_per_unit": self.risk_per_unit,
            "notional": self.notional,
            "block_reason": str(self.block_reason) if self.block_reason else None,
            "reasoning": list(self.reasoning),
            "metrics": dict(self.metrics),
        }

    def describe(self) -> str:
        if not self.approved:
            reasons = "; ".join(self.reasoning) or "no reason recorded"
            return f"REJECTED ({self.block_reason}): {reasons}"
        return (
            f"{self.verdict}: {self.quantity:.6g} units, "
            f"risking {self.risk_amount:.2f} ({self.risk_per_unit:.6g}/unit)"
        )

    @classmethod
    def reject(
        cls, reason: RiskBlockReason, *messages: str, **metrics: Any
    ) -> RiskDecision:
        return cls(
            verdict=RiskVerdict.REJECTED,
            block_reason=reason,
            reasoning=messages,
            metrics=metrics,
        )
