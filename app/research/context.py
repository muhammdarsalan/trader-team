"""Research context for surfaces that are not the research module.

The paper-trading dashboard runs a configuration. Somewhere on disk there may
be a validation study of that same configuration. This module is the bridge: it
locates a study, reads its machine-readable report, and returns a payload the
dashboard can render — including, and especially, when there is no study to
read.

Two things it will not do.

**It never invents a result.** Every field is either read from a report on disk
or absent, and absence is reported as :attr:`ResearchAvailability.MISSING` with
the reason attached. A dashboard panel that filled in plausible zeros for a
study that was never run would be worse than an empty panel, because an empty
panel is obviously empty.

**It never presents a study of one configuration as evidence about another.**
A report carries the config fingerprint it was produced under. When that does
not match the running configuration the payload says
:attr:`ResearchAvailability.STALE` and names the mismatch, rather than showing
numbers that describe a system nobody is running any more.

The evidence hierarchy this exposes — code passing, in-sample, validation,
out-of-sample, walk-forward — is the whole point. A green test suite says the
code does what it was written to do. It says nothing whatsoever about whether
the strategy has an edge, and the payload keeps those two claims in separate
fields so a page cannot accidentally render one as the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.utils.logging import get_logger
from app.utils.paths import reports_dir

logger = get_logger(__name__)

__all__ = [
    "EVIDENCE_LADDER",
    "ResearchAvailability",
    "ResearchContext",
    "find_validation_reports",
    "load_research_context",
]


class ResearchAvailability(StrEnum):
    """Whether there is a study to show, and if not, why not."""

    #: A study was found for this instrument and configuration.
    AVAILABLE = "AVAILABLE"
    #: No study has been run for this instrument, or none could be found.
    MISSING = "MISSING"
    #: A study exists but describes a different configuration.
    STALE = "STALE"
    #: A report was found but could not be read.
    UNREADABLE = "UNREADABLE"


#: The rungs of evidence, weakest first, and what each does and does not support.
#: Rendered on the dashboard so the distinction between "the code runs" and "the
#: strategy works" is visible rather than assumed.
EVIDENCE_LADDER: tuple[dict[str, str], ...] = (
    {
        "rung": "Code and tests",
        "supports": "The implementation does what it was written to do.",
        "does_not_support": (
            "Nothing about whether the strategy has an edge. A green suite on a "
            "losing strategy is a correctly implemented losing strategy."
        ),
    },
    {
        "rung": "In-sample results",
        "supports": "A description of the bars the configuration was built from.",
        "does_not_support": (
            "Any claim at all. The configuration exists because of these bars; it "
            "would be surprising if it did badly on them."
        ),
    },
    {
        "rung": "Validation window",
        "supports": "One held-out window, used to choose between candidates.",
        "does_not_support": (
            "An out-of-sample claim. Every look moves this window closer to being "
            "in-sample."
        ),
    },
    {
        "rung": "Out-of-sample window",
        "supports": "One window the configuration had never seen, scored once.",
        "does_not_support": (
            "A general conclusion. One window of one instrument, with estimated "
            "execution costs."
        ),
    },
    {
        "rung": "Walk-forward folds",
        "supports": (
            "Repetition: a choice made on each fold's training half, scored on its "
            "test half, several times."
        ),
        "does_not_support": (
            "A forecast. It is the strongest evidence this platform produces and it "
            "remains weak."
        ),
    },
)


@dataclass
class ResearchContext:
    """What research says about the configuration a surface is running.

    ``availability`` is the field to check first. Every other field is empty or
    ``None`` unless it is :attr:`ResearchAvailability.AVAILABLE`, and no field is
    ever filled with a stand-in.
    """

    availability: ResearchAvailability
    reason: str
    symbol: str | None = None
    timeframe: str | None = None
    experiment_id: str | None = None
    report_path: str | None = None
    report: dict[str, Any] = field(default_factory=dict)
    #: Other studies found for this instrument, newest first, for provenance.
    other_experiments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_available(self) -> bool:
        return self.availability is ResearchAvailability.AVAILABLE

    # ------------------------------------------------------------- accessors

    def section(self, name: str) -> Any:
        """One section of the report, or None when absent or unavailable."""
        return self.report.get(name) if self.is_available else None

    def segment(self, name: str) -> dict[str, Any] | None:
        """One window's summary row, by name.

        The names matter: ``in_sample``, ``validation`` and ``out_of_sample`` are
        different claims, and a caller that wants out-of-sample numbers has to
        ask for out-of-sample numbers.
        """
        for row in self.report.get("segments") or []:
            if row.get("segment") == name:
                return row
        return None

    @property
    def verdict(self) -> str | None:
        return self.report.get("verdict") if self.is_available else None

    @property
    def verdict_reasons(self) -> list[str]:
        return list(self.report.get("verdict_reasons") or []) if self.is_available else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "availability": str(self.availability),
            "reason": self.reason,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "experiment_id": self.experiment_id,
            "report_path": self.report_path,
            "report": self.report,
            "other_experiments": self.other_experiments,
            "evidence_ladder": [dict(r) for r in EVIDENCE_LADDER],
        }


# ------------------------------------------------------------------- loading


def find_validation_reports(directory: Path | None = None) -> list[Path]:
    """Every validation report on disk, newest first.

    Sorted by modification time rather than by experiment id: ids are content
    hashes, so they carry no ordering.
    """
    base = directory or (reports_dir() / "validation")
    if not base.exists():
        return []
    found = [p / "validation_report.json" for p in base.iterdir() if p.is_dir()]
    existing = [p for p in found if p.exists()]
    return sorted(existing, key=lambda p: p.stat().st_mtime, reverse=True)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "Could not read a validation report", extra={"path": str(path), "error": str(exc)}
        )
        return None
    return payload if isinstance(payload, dict) else None


def load_research_context(
    symbol: str,
    timeframe: str,
    *,
    directory: Path | None = None,
    config_fingerprint: str | None = None,
) -> ResearchContext:
    """Find the most recent study for ``symbol``/``timeframe`` and load it.

    Args:
        symbol: canonical platform symbol the surface is running.
        timeframe: bar interval.
        directory: where to look. Defaults to ``reports/validation``.
        config_fingerprint: the running configuration's fingerprint. When given
            and a report's own fingerprint differs, the context is marked
            :attr:`ResearchAvailability.STALE` — the study describes a different
            system, and showing its numbers as though it described this one is
            the mistake this argument exists to prevent.

    Returns:
        A :class:`ResearchContext`. Never raises for a missing or malformed
        report; the availability field carries what went wrong.
    """
    symbol = symbol.upper()
    paths = find_validation_reports(directory)

    if not paths:
        return ResearchContext(
            availability=ResearchAvailability.MISSING,
            reason=(
                "No validation study has been run. Nothing on this page has been "
                "tested out of sample. Run: python scripts/run_research.py --symbol "
                f"{symbol} --timeframe {timeframe}"
            ),
            symbol=symbol,
            timeframe=timeframe,
        )

    unreadable = 0
    matches: list[tuple[Path, dict[str, Any]]] = []
    others: list[dict[str, Any]] = []

    for path in paths:
        payload = _read(path)
        if payload is None:
            unreadable += 1
            continue
        entry = {
            "experiment_id": payload.get("experiment_id"),
            "symbol": payload.get("symbol"),
            "timeframe": payload.get("timeframe"),
            "verdict": payload.get("verdict"),
            "created_at": payload.get("created_at"),
            "path": str(path.parent),
        }
        others.append(entry)
        if (
            str(payload.get("symbol", "")).upper() == symbol
            and str(payload.get("timeframe", "")) == timeframe
        ):
            matches.append((path, payload))

    if not matches:
        if unreadable and not others:
            return ResearchContext(
                availability=ResearchAvailability.UNREADABLE,
                reason=(
                    f"{unreadable} validation report(s) were found but none could be "
                    "parsed. The files may be truncated from an interrupted run."
                ),
                symbol=symbol,
                timeframe=timeframe,
                other_experiments=others,
            )
        return ResearchContext(
            availability=ResearchAvailability.MISSING,
            reason=(
                f"{len(others)} validation study(ies) exist, but none for {symbol} "
                f"{timeframe}. A study of a different instrument says nothing about "
                "this one."
            ),
            symbol=symbol,
            timeframe=timeframe,
            other_experiments=others,
        )

    path, payload = matches[0]
    experiment_id = payload.get("experiment_id")

    if config_fingerprint is not None:
        recorded = (payload.get("spec") or {}).get("config_fingerprint")
        if recorded is not None and recorded != config_fingerprint:
            return ResearchContext(
                availability=ResearchAvailability.STALE,
                reason=(
                    f"Study {experiment_id} was run under configuration {recorded}, and "
                    f"this session is running {config_fingerprint}. Its numbers describe "
                    "a different system. Re-run the study to validate what is actually "
                    "configured."
                ),
                symbol=symbol,
                timeframe=timeframe,
                experiment_id=experiment_id,
                report_path=str(path.parent),
                other_experiments=others,
            )

    return ResearchContext(
        availability=ResearchAvailability.AVAILABLE,
        reason=f"Study {experiment_id}, generated {payload.get('created_at')}.",
        symbol=symbol,
        timeframe=timeframe,
        experiment_id=experiment_id,
        report_path=str(path.parent),
        report=payload,
        other_experiments=[e for e in others if e["experiment_id"] != experiment_id],
    )
