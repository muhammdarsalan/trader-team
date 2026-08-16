"""The risk engine.

Decides whether a signal becomes a position, and how large. Every check either
approves, reduces, or refuses with a recorded reason - a no-trade decision is
as much a result as a trade, and an unexplained one is a bug nobody can find.

The core arithmetic is deliberately simple enough to verify by hand:

    risk_amount   = equity * risk_per_trade
    risk_per_unit = |entry - stop|
    quantity      = risk_amount / risk_per_unit

Everything else is a cap applied on top. Caps only ever shrink the position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config.models import AssetConfig, RiskConfig
from app.data.validators.quality import QualityStatus
from app.portfolio.portfolio import Portfolio
from app.risk.models import RiskBlockReason, RiskDecision, RiskVerdict
from app.signals.models import Signal
from app.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_quality(value: QualityStatus | str | None) -> QualityStatus | None:
    """Coerce a data-quality grade to :class:`QualityStatus`, or None if unstated.

    ``None`` is returned for anything that does not name a real grade -
    including the literal string ``"UNKNOWN"``, which several call sites use as
    a placeholder. That placeholder must not be mistaken for a pass, so it maps
    to the same "not verified" state as no answer at all.
    """
    if value is None:
        return None
    if isinstance(value, QualityStatus):
        return value
    try:
        return QualityStatus[str(value).strip().upper()]
    except KeyError:
        return None


class RiskEngine:
    """Position sizing and exposure limits."""

    def __init__(
        self,
        config: RiskConfig | None = None,
        trading_enabled: bool = False,
        data_quality: QualityStatus | str | None = None,
    ) -> None:
        self.config = config or RiskConfig()
        # The global kill switch. While False no position opens, in any mode.
        self.trading_enabled = trading_enabled
        # The grade the quality engine gave the series this engine is sizing
        # against. Set once per run by whoever owns the data; overridable per
        # call for a feed whose grade changes bar to bar.
        self.data_quality = normalize_quality(data_quality)

    # ------------------------------------------------------------------- API

    def evaluate(
        self,
        signal: Signal,
        portfolio: Portfolio,
        equity: float,
        asset: AssetConfig | None = None,
        correlations: pd.DataFrame | None = None,
        data_quality: QualityStatus | str | None = None,
    ) -> RiskDecision:
        """Decide whether and how large to trade ``signal``.

        Args:
            signal: the proposed trade.
            portfolio: current positions and history.
            equity: current mark-to-market equity.
            asset: asset config, for notional and minimum-size rules.
            correlations: symbol correlation matrix, when available.
            data_quality: grade of the series this signal was derived from.
                Falls back to the grade the engine was constructed with. When
                neither is supplied the engine refuses rather than assuming.

        Returns:
            A :class:`RiskDecision` recording the arithmetic and every check.
        """
        cfg = self.config
        reasoning: list[str] = []

        # --- gates that block outright ---------------------------------------
        if not signal.is_actionable:
            return RiskDecision.reject(
                RiskBlockReason.NO_SIGNAL, "Signal is not actionable"
            )

        if not self.trading_enabled:
            return RiskDecision.reject(
                RiskBlockReason.KILL_SWITCH,
                "Trading is disabled by the global kill switch; the signal was analysed "
                "but no position was opened",
            )

        quality = self._effective_quality(data_quality)
        quality_block = self._data_quality_block(quality)
        if quality_block is not None:
            return quality_block

        risk_per_unit = signal.risk_per_unit
        if risk_per_unit is None or risk_per_unit <= 0:
            return RiskDecision.reject(
                RiskBlockReason.INVALID_STOP,
                f"Stop distance is {risk_per_unit}, so position size is undefined",
            )

        if equity <= 0:
            return RiskDecision.reject(
                RiskBlockReason.INSUFFICIENT_CASH, f"Equity is {equity:.2f}"
            )

        drawdown = portfolio.drawdown(equity)
        if drawdown >= cfg.max_drawdown_limit:
            return RiskDecision.reject(
                RiskBlockReason.MAX_DRAWDOWN,
                f"Drawdown {drawdown:.1%} has reached the {cfg.max_drawdown_limit:.1%} "
                "limit; no new positions until equity recovers",
                drawdown=drawdown,
            )

        daily_loss = portfolio.daily_loss(equity)
        if daily_loss >= cfg.daily_loss_limit:
            return RiskDecision.reject(
                RiskBlockReason.DAILY_LOSS_LIMIT,
                f"Today's loss {daily_loss:.1%} has reached the {cfg.daily_loss_limit:.1%} "
                "limit; trading halts for the day",
                daily_loss=daily_loss,
            )

        if portfolio.open_count >= cfg.max_concurrent_positions:
            return RiskDecision.reject(
                RiskBlockReason.MAX_POSITIONS,
                f"{portfolio.open_count} positions already open "
                f"(limit {cfg.max_concurrent_positions})",
            )

        existing = portfolio.positions_for(signal.symbol)
        if len(existing) >= cfg.max_positions_per_symbol:
            return RiskDecision.reject(
                RiskBlockReason.MAX_POSITIONS_PER_SYMBOL,
                f"{len(existing)} position(s) already open in {signal.symbol} "
                f"(limit {cfg.max_positions_per_symbol})",
            )

        # --- base sizing ------------------------------------------------------
        risk_amount = self._base_risk_amount(equity)
        reasoning.append(
            f"Base risk {risk_amount:.2f} "
            f"({cfg.sizing_method}, equity {equity:.2f})"
        )

        # --- drawdown scaling -------------------------------------------------
        reduced = False
        if drawdown >= cfg.drawdown_reduce_threshold:
            risk_amount *= cfg.drawdown_reduce_factor
            reduced = True
            reasoning.append(
                f"Drawdown {drawdown:.1%} exceeds {cfg.drawdown_reduce_threshold:.1%}; "
                f"risk scaled by {cfg.drawdown_reduce_factor:g} to {risk_amount:.2f}"
            )

        # --- portfolio-level risk budgets -------------------------------------
        budget_checks = [
            (
                portfolio.open_risk(),
                cfg.max_portfolio_risk * equity,
                RiskBlockReason.PORTFOLIO_RISK,
                "portfolio",
            ),
            (
                portfolio.risk_for_strategy(signal.strategy),
                cfg.max_risk_per_strategy * equity,
                RiskBlockReason.STRATEGY_RISK,
                f"strategy {signal.strategy}",
            ),
        ]
        for used, cap, block_reason, label in budget_checks:
            remaining = cap - used
            if remaining <= 0:
                return RiskDecision.reject(
                    block_reason,
                    f"The {label} risk budget is exhausted "
                    f"({used:.2f} of {cap:.2f} in use)",
                    used=used, cap=cap,
                )
            if risk_amount > remaining:
                risk_amount = remaining
                reduced = True
                reasoning.append(
                    f"Risk trimmed to {risk_amount:.2f} by the {label} budget "
                    f"({used:.2f} of {cap:.2f} already committed)"
                )

        # --- correlation-aware budget ------------------------------------------
        correlated_risk = self._correlated_risk(signal, portfolio, correlations)
        if correlated_risk > 0:
            cap = cfg.max_correlated_risk * equity
            remaining = cap - correlated_risk
            if remaining <= 0:
                return RiskDecision.reject(
                    RiskBlockReason.CORRELATED_RISK,
                    f"Existing positions correlated with {signal.symbol} already use "
                    f"{correlated_risk:.2f} of the {cap:.2f} correlated-risk budget. "
                    "Correlated positions are one bet, not several.",
                    correlated_risk=correlated_risk,
                )
            if risk_amount > remaining:
                risk_amount = remaining
                reduced = True
                reasoning.append(
                    f"Risk trimmed to {risk_amount:.2f} by the correlated-exposure budget"
                )

        # --- quantity ----------------------------------------------------------
        quantity = risk_amount / risk_per_unit
        entry = float(signal.entry_price)

        # --- notional caps -------------------------------------------------------
        max_notional = cfg.max_position_notional_pct * equity
        if quantity * entry > max_notional:
            quantity = max_notional / entry
            risk_amount = quantity * risk_per_unit
            reduced = True
            reasoning.append(
                f"Size capped by the {cfg.max_position_notional_pct:.0%} per-position "
                f"notional limit; risk falls to {risk_amount:.2f}"
            )

        current_notional = portfolio.exposure({p.symbol: p.entry_price for p in portfolio.positions})
        portfolio_notional_cap = cfg.max_portfolio_notional_pct * equity
        headroom = portfolio_notional_cap - current_notional
        if headroom <= 0:
            return RiskDecision.reject(
                RiskBlockReason.PORTFOLIO_NOTIONAL,
                f"Portfolio notional {current_notional:.2f} has reached its "
                f"{portfolio_notional_cap:.2f} cap",
            )
        if quantity * entry > headroom:
            quantity = headroom / entry
            risk_amount = quantity * risk_per_unit
            reduced = True
            reasoning.append(
                f"Size capped by remaining portfolio notional headroom ({headroom:.2f})"
            )

        # --- minimum viable size -------------------------------------------------
        if asset is not None and cfg.min_position_units > 0 and quantity < cfg.min_position_units:
            return RiskDecision.reject(
                RiskBlockReason.POSITION_TOO_SMALL,
                f"Computed size {quantity:.6g} is below the {cfg.min_position_units:g} "
                "minimum; the account is too small for this stop distance",
                quantity=quantity,
            )

        if quantity <= 0 or not np.isfinite(quantity):
            return RiskDecision.reject(
                RiskBlockReason.POSITION_TOO_SMALL,
                f"Computed size is {quantity}",
            )

        notional = quantity * entry
        reasoning.append(
            f"Final size {quantity:.6g} units = risk {risk_amount:.2f} / "
            f"{risk_per_unit:.6g} per unit"
        )

        decision = RiskDecision(
            verdict=RiskVerdict.REDUCED if reduced else RiskVerdict.APPROVED,
            quantity=quantity,
            risk_amount=risk_amount,
            risk_per_unit=risk_per_unit,
            notional=notional,
            reasoning=tuple(reasoning),
            metrics={
                "equity": equity,
                "drawdown": drawdown,
                "daily_loss": daily_loss,
                "open_positions": portfolio.open_count,
                "open_risk": portfolio.open_risk(),
                "risk_pct_of_equity": risk_amount / equity if equity else 0.0,
                "data_quality": str(quality),
            },
        )

        logger.debug(
            "Risk decision",
            extra={
                "symbol": signal.symbol, "strategy": signal.strategy,
                "verdict": str(decision.verdict), "quantity": quantity,
                "risk_amount": risk_amount,
            },
        )
        return decision

    # -------------------------------------------------------------- internals

    def _effective_quality(
        self, per_call: QualityStatus | str | None
    ) -> QualityStatus | None:
        """The grade in force for this decision.

        A per-call grade wins over the constructor's, and an unrecognised
        per-call value does not silently fall back to a friendlier
        constructor grade - it is the caller asserting something the engine
        cannot read, which is exactly the case that must not pass.
        """
        if per_call is not None:
            return normalize_quality(per_call)
        return self.data_quality

    def _data_quality_block(self, quality: QualityStatus | None) -> RiskDecision | None:
        """Refuse to size a position on data that was never verified, or failed.

        The quality engine already refuses to *hand over* FAIL-grade data on the
        normal path, but that path is not the only one: a caller can disable
        validation, load a frame straight from disk, or forget to pass the grade
        along. Repeating the check at the point where money is committed means a
        degraded series cannot reach a position size by taking a side entrance,
        and the refusal is recorded like every other no-trade decision.
        """
        cfg = self.config

        if quality is None:
            if not cfg.require_known_data_quality:
                return None
            return RiskDecision.reject(
                RiskBlockReason.DATA_QUALITY_UNKNOWN,
                "No data-quality grade was supplied with this signal, so the series "
                "backing it is unverified. An unstated grade is treated as a refusal: "
                "'nobody checked' must not size the same position as 'checked and passed'. "
                "Pass the quality report's status through, or set "
                "require_known_data_quality: false in configs/risk.yaml to accept the risk "
                "deliberately.",
                data_quality="UNVERIFIED",
            )

        if quality is QualityStatus.FAIL and cfg.block_on_data_quality_fail:
            return RiskDecision.reject(
                RiskBlockReason.DATA_QUALITY,
                "The data-quality gate graded this series FAIL. Every input to position "
                "sizing descends from that series, so no size computed from it is "
                "meaningful.",
                data_quality=str(quality),
            )

        if quality is QualityStatus.WARNING and cfg.block_on_data_quality_warning:
            return RiskDecision.reject(
                RiskBlockReason.DATA_QUALITY,
                "The data-quality gate graded this series WARNING and "
                "block_on_data_quality_warning is set, so no new position is opened.",
                data_quality=str(quality),
            )

        return None

    def _base_risk_amount(self, equity: float) -> float:
        cfg = self.config
        if cfg.sizing_method == "percent_risk":
            return equity * cfg.risk_per_trade
        if cfg.sizing_method == "fixed_risk":
            return cfg.fixed_risk_amount
        # fixed_units is handled by the caller multiplying out; expressed here as
        # the risk implied by a fixed unit count is not knowable without the stop,
        # so treat it as percent risk with the configured units applied later.
        return equity * cfg.risk_per_trade

    def _correlated_risk(
        self,
        signal: Signal,
        portfolio: Portfolio,
        correlations: pd.DataFrame | None,
    ) -> float:
        """Open risk in positions correlated with this signal's symbol.

        Same-direction exposure in correlated symbols is one bet. Three
        "independent" longs in EURUSD, GBPUSD and AUDUSD are a single short-dollar
        position wearing three hats, and sizing them as if independent triples
        the real risk.
        """
        if correlations is None or correlations.empty:
            return 0.0

        symbol = signal.symbol.upper()
        if symbol not in correlations.index:
            return 0.0

        threshold = self.config.correlation_threshold
        total = 0.0
        for position in portfolio.positions:
            other = position.symbol.upper()
            if other == symbol:
                total += position.risk_amount
                continue
            if other not in correlations.columns:
                continue

            corr = correlations.loc[symbol, other]
            if pd.isna(corr):
                continue

            same_direction = position.direction is signal.direction
            # A strong negative correlation between opposite-direction positions
            # is also concentration: short EURUSD and long USDCHF are one trade.
            aligned = (corr >= threshold and same_direction) or (
                corr <= -threshold and not same_direction
            )
            if aligned:
                total += position.risk_amount

        return total


def rolling_correlations(
    prices: dict[str, pd.Series], lookback: int = 60
) -> pd.DataFrame:
    """Correlation matrix of returns across symbols.

    Uses returns, not prices: two upward-trending price series correlate
    strongly whether or not their movements have anything to do with each other.

    Returns an empty frame when fewer than two symbols have enough history,
    which the risk engine treats as "no correlation information" rather than
    "no correlation".
    """
    usable = {
        symbol: series.pct_change().dropna().iloc[-lookback:]
        for symbol, series in prices.items()
        if len(series) > lookback // 2
    }
    usable = {s: r for s, r in usable.items() if len(r) >= max(10, lookback // 4)}

    if len(usable) < 2:
        return pd.DataFrame()

    return pd.DataFrame(usable).corr()
