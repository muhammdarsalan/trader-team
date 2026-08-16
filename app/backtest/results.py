"""Backtest results and the reproducibility record.

A result that cannot be regenerated is not a result. Every run carries the
configuration, the data fingerprint, the code revision and the random seed that
produced it, so the question "exactly which code and parameters produced this?"
has an answer that does not depend on anyone's memory.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.backtest.metrics import PerformanceMetrics
from app.utils.logging import get_logger
from app.utils.paths import ensure_dir
from app.utils.timeutils import utcnow

logger = get_logger(__name__)


def git_revision() -> str:
    """Current commit hash, or a marker when unavailable.

    Records ``dirty`` when the working tree has uncommitted changes, because a
    result produced from modified code is not reproducible from the commit
    alone and saying so is more useful than a hash that lies.
    """
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if rev.returncode != 0:
            return "not-a-git-repo"

        commit = rev.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return f"{commit}-dirty" if status.stdout.strip() else commit
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass
class RunProvenance:
    """Everything needed to reproduce a run."""

    experiment_id: str
    symbol: str
    timeframe: str
    start: str | None
    end: str | None
    bars: int
    data_provider: str
    data_checksum: str
    data_quality: str
    git_revision: str = field(default_factory=git_revision)
    random_seed: int = 42
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start": self.start,
            "end": self.end,
            "bars": self.bars,
            "data_provider": self.data_provider,
            "data_checksum": self.data_checksum,
            "data_quality": self.data_quality,
            "git_revision": self.git_revision,
            "random_seed": self.random_seed,
            "created_at": self.created_at,
            "config_snapshot": self.config_snapshot,
        }


@dataclass
class BacktestResult:
    """A completed backtest: metrics, trades, equity curve and provenance."""

    provenance: RunProvenance
    metrics: PerformanceMetrics
    trades: pd.DataFrame
    equity_curve: pd.Series
    snapshots: pd.DataFrame
    decisions: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_performance: pd.DataFrame = field(default_factory=pd.DataFrame)
    blocked_counts: dict[str, int] = field(default_factory=dict)
    node_errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def experiment_id(self) -> str:
        return self.provenance.experiment_id

    def render(self) -> str:
        """Full text report."""
        width = 60
        lines = [
            "=" * width,
            "BACKTEST REPORT",
            "=" * width,
            "",
            f"Experiment:   {self.provenance.experiment_id}",
            f"Asset:        {self.provenance.symbol} {self.provenance.timeframe}",
            f"Period:       {self.provenance.start} -> {self.provenance.end}",
            f"Bars:         {self.provenance.bars:,}",
            f"Data:         {self.provenance.data_provider} "
            f"(quality {self.provenance.data_quality}, checksum {self.provenance.data_checksum})",
            f"Git revision: {self.provenance.git_revision}",
            f"Random seed:  {self.provenance.random_seed}",
            "",
            self.metrics.render(),
        ]

        if self.blocked_counts:
            lines += ["", "-" * width, "NO-TRADE DECISIONS", "-" * width]
            total = sum(self.blocked_counts.values())
            for reason, count in sorted(
                self.blocked_counts.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"  {reason:<34}{count:>8,}  ({count / total:.1%})")

        if not self.regime_performance.empty:
            lines += ["", "-" * width, "PERFORMANCE BY REGIME", "-" * width]
            lines.append(self.regime_performance.to_string(index=False))

        if self.node_errors:
            lines += ["", "-" * width, f"NODE ERRORS ({len(self.node_errors)})", "-" * width]
            seen: dict[str, int] = {}
            for error in self.node_errors:
                key = f"{error['node']}: {error['error_type']}"
                seen[key] = seen.get(key, 0) + 1
            for key, count in sorted(seen.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {key} x{count}")

        lines.append("=" * width)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    # ------------------------------------------------------------------ output

    def save(self, directory: Path | None = None) -> Path:
        """Write the full result to disk, keyed by experiment id."""
        from app.utils.paths import experiments_dir

        base = ensure_dir((directory or experiments_dir()) / self.provenance.experiment_id)

        (base / "provenance.json").write_text(
            json.dumps(self.provenance.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        (base / "metrics.json").write_text(
            json.dumps(self.metrics.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        (base / "report.txt").write_text(self.render(), encoding="utf-8")

        if not self.trades.empty:
            self.trades.to_csv(base / "trades.csv", index=False)
        if not self.equity_curve.empty:
            self.equity_curve.to_csv(base / "equity_curve.csv")
        if not self.decisions.empty:
            self.decisions.to_csv(base / "decisions.csv", index=False)
        if not self.regime_performance.empty:
            self.regime_performance.to_csv(base / "regime_performance.csv", index=False)

        logger.info(
            "Backtest result saved",
            extra={"experiment_id": self.provenance.experiment_id, "path": str(base)},
        )
        return base
