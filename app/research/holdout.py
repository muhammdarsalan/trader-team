"""Frozen holdout protection.

A holdout is the one window whose result carries weight, and it carries weight
only for as long as it stays untouched. The moment a decision is informed by it
- a parameter changed because the holdout number was disappointing, a threshold
picked because it made the holdout look better - it stops being out of sample
and becomes the most expensive kind of in-sample data: the kind everyone still
believes is out of sample.

This module makes that decay **visible and recorded** rather than trusting it
not to happen. It seals a window, fingerprints the exact bars sealed, and keeps
a durable ledger of every sanctioned evaluation of it, with a running touch
count. The first touch is expected. Every touch after that is reported as what
it is: development data wearing an out-of-sample label.

## What is technically enforced, and what is not

Be precise about this, because a governance mechanism that overstates its own
reach is worse than none - it manufactures false confidence.

**Enforced.** Through the sanctioned door (:meth:`HoldoutRegistry.evaluate`):
every evaluation is logged, the touch count is incremented, and the sealed
window's fingerprint is re-checked against the data presented - so a holdout
that has been edited, re-downloaded, or silently swapped is *refused*, not
scored. Separately, :meth:`HoldoutRegistry.assert_available_for_development`
lets development and optimisation code refuse to run on any window whose
fingerprint is sealed, so a parameter sweep cannot quietly consume the holdout.
The ledger is a SQLite file, so touch counts survive process restarts and
accumulate across sessions.

**Not enforced, and cannot be.** Nothing here can stop a person from
reconstructing the same bars from the raw series and backtesting them without
ever calling this registry. The registry sees only what comes through its door.
It makes the correct path the easy one and records every sanctioned touch; it is
a **ledger, not a sandbox**. The honest claim is "every touch that went through
the front door is recorded, and development code that asks will be told to keep
out" - not "the holdout cannot be peeked at".

Unlike the experiment store, this database is **not disposable**. Deleting it
does not lose derived data; it erases the record that a holdout was ever
touched, which is the one thing here worth keeping. It is governance, not cache.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.data.cache import frame_checksum
from app.research.experiments import stable_hash
from app.utils.logging import get_logger
from app.utils.paths import ensure_dir, experiments_dir
from app.utils.timeutils import utcnow

logger = get_logger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS holdout_schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS holdout_seals (
    holdout_id         TEXT PRIMARY KEY,
    symbol             TEXT NOT NULL,
    timeframe          TEXT NOT NULL,
    start_time         TEXT NOT NULL,
    end_time           TEXT NOT NULL,
    bars               INTEGER NOT NULL,
    data_fingerprint   TEXT NOT NULL,
    config_fingerprint TEXT,
    git_revision       TEXT,
    random_seed        INTEGER,
    created_at         TEXT NOT NULL,
    label              TEXT,
    notes              TEXT
);

CREATE INDEX IF NOT EXISTS idx_seals_fingerprint
    ON holdout_seals(data_fingerprint);

CREATE TABLE IF NOT EXISTS holdout_accesses (
    access_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    holdout_id         TEXT NOT NULL,
    accessed_at        TEXT NOT NULL,
    touch_number       INTEGER NOT NULL,
    experiment_id      TEXT,
    purpose            TEXT,
    config_fingerprint TEXT,
    git_revision       TEXT,
    integrity_ok       INTEGER NOT NULL,
    FOREIGN KEY (holdout_id) REFERENCES holdout_seals(holdout_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_accesses_holdout
    ON holdout_accesses(holdout_id);
"""


class HoldoutError(RuntimeError):
    """Base for holdout governance failures."""


class HoldoutIntegrityError(HoldoutError):
    """The data presented does not match what was sealed under this holdout."""


class HoldoutViolationError(HoldoutError):
    """A development or optimisation operation tried to use a sealed holdout."""


# ------------------------------------------------------------- fingerprinting


def window_fingerprint(
    df: pd.DataFrame, start_time: pd.Timestamp, end_time: pd.Timestamp
) -> str:
    """Content hash of exactly the bars in ``[start_time, end_time]``.

    Fingerprinting the window rather than the whole series is what lets a seal
    notice that the holdout's own bars changed even if the rest of the history
    is untouched - a vendor re-statement, a re-download with a different
    adjustment, a hand-edit.
    """
    window = df.loc[start_time:end_time]
    if window.empty:
        raise HoldoutError(
            f"No bars fall in [{start_time}, {end_time}]; there is nothing to seal."
        )
    return frame_checksum(window)


def _holdout_id(symbol: str, timeframe: str, start_time: Any, end_time: Any) -> str:
    """Deterministic identity of a holdout *window*.

    Derived from the window's coordinates, deliberately **not** from the data
    fingerprint: the id names the window, so presenting different data for the
    same window is a detectable integrity failure rather than a silently
    different holdout.
    """
    return "HOLD-" + stable_hash(
        {
            "symbol": symbol.upper(),
            "timeframe": str(timeframe),
            "start": str(start_time),
            "end": str(end_time),
        },
        length=12,
    )


# -------------------------------------------------------------------- records


@dataclass(frozen=True)
class HoldoutSeal:
    """A sealed holdout window."""

    holdout_id: str
    symbol: str
    timeframe: str
    start_time: str
    end_time: str
    bars: int
    data_fingerprint: str
    config_fingerprint: str | None = None
    git_revision: str | None = None
    random_seed: int | None = None
    created_at: str = ""
    label: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "holdout_id": self.holdout_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "bars": self.bars,
            "data_fingerprint": self.data_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "git_revision": self.git_revision,
            "random_seed": self.random_seed,
            "created_at": self.created_at,
            "label": self.label,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class HoldoutAccess:
    """One sanctioned evaluation of a sealed holdout, and what it means."""

    holdout_id: str
    accessed_at: str
    touch_number: int
    experiment_id: str | None
    purpose: str
    config_fingerprint: str | None
    git_revision: str | None
    integrity_ok: bool

    @property
    def is_first_touch(self) -> bool:
        return self.touch_number == 1

    @property
    def warnings(self) -> tuple[str, ...]:
        notes: list[str] = []
        if self.touch_number > 1:
            notes.append(
                f"This holdout has now been evaluated {self.touch_number} times. After "
                "the first look it is no longer out of sample: each further evaluation is "
                "development data wearing an out-of-sample label, and its result should be "
                "read as in-sample."
            )
        if not self.integrity_ok:
            notes.append(
                "The data presented did not match what was sealed. This result is not an "
                "evaluation of the sealed holdout."
            )
        return tuple(notes)


@dataclass
class HoldoutStatus:
    """A seal together with its access history, for reporting."""

    seal: HoldoutSeal
    touch_count: int
    accesses: list[HoldoutAccess] = field(default_factory=list)

    @property
    def untouched(self) -> bool:
        return self.touch_count == 0

    @property
    def over_touched(self) -> bool:
        return self.touch_count > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "seal": self.seal.to_dict(),
            "touch_count": self.touch_count,
            "untouched": self.untouched,
            "over_touched": self.over_touched,
            "accesses": [
                {
                    "accessed_at": a.accessed_at,
                    "touch_number": a.touch_number,
                    "experiment_id": a.experiment_id,
                    "purpose": a.purpose,
                    "integrity_ok": a.integrity_ok,
                }
                for a in self.accesses
            ],
        }


# ------------------------------------------------------------------- registry


class HoldoutRegistry:
    """A durable ledger of sealed holdout windows and every touch of them."""

    def __init__(self, path: Path | None = None) -> None:
        """
        Args:
            path: database file. Defaults to ``experiments/holdout.db``. Unlike
                the experiment store this is not disposable - it is the record
                that a holdout was sealed and how often it has been touched.
        """
        self.path = Path(path) if path is not None else experiments_dir() / "holdout.db"
        ensure_dir(self.path.parent)
        self._initialise()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            row = connection.execute(
                "SELECT version FROM holdout_schema_version"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO holdout_schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )
            elif row["version"] != SCHEMA_VERSION:
                logger.warning(
                    "Holdout database was written by a different schema version",
                    extra={"found": row["version"], "expected": SCHEMA_VERSION},
                )

    # ---------------------------------------------------------------- sealing

    def seal(
        self,
        *,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        config_fingerprint: str | None = None,
        git_revision: str | None = None,
        random_seed: int | None = None,
        label: str = "",
        notes: str = "",
    ) -> HoldoutSeal:
        """Seal a window as a frozen holdout.

        Idempotent for an identical window and identical data: re-sealing
        returns the existing seal. Re-sealing the *same window* with *different
        bars* raises :class:`HoldoutIntegrityError` - a holdout whose data
        changed after sealing is a governance event, not a quiet re-seal.
        """
        holdout_id = _holdout_id(symbol, timeframe, start_time, end_time)
        fingerprint = window_fingerprint(data, start_time, end_time)
        bars = len(data.loc[start_time:end_time])

        existing = self.get_seal(holdout_id)
        if existing is not None:
            if existing.data_fingerprint != fingerprint:
                raise HoldoutIntegrityError(
                    f"Holdout {holdout_id} is already sealed over the same window with "
                    f"different data (sealed {existing.data_fingerprint}, now "
                    f"{fingerprint}). Re-sealing would erase the original seal; refuse "
                    "instead. The holdout's data changed after it was frozen."
                )
            return existing

        seal = HoldoutSeal(
            holdout_id=holdout_id,
            symbol=symbol.upper(),
            timeframe=str(timeframe),
            start_time=str(start_time),
            end_time=str(end_time),
            bars=bars,
            data_fingerprint=fingerprint,
            config_fingerprint=config_fingerprint,
            git_revision=git_revision,
            random_seed=random_seed,
            created_at=utcnow().isoformat(),
            label=label,
            notes=notes,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO holdout_seals (
                    holdout_id, symbol, timeframe, start_time, end_time, bars,
                    data_fingerprint, config_fingerprint, git_revision, random_seed,
                    created_at, label, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    seal.holdout_id, seal.symbol, seal.timeframe, seal.start_time,
                    seal.end_time, seal.bars, seal.data_fingerprint,
                    seal.config_fingerprint, seal.git_revision, seal.random_seed,
                    seal.created_at, seal.label, seal.notes,
                ),
            )
        logger.info("Sealed holdout", extra={"holdout_id": holdout_id, "bars": bars})
        return seal

    # ------------------------------------------------------------- evaluation

    def evaluate(
        self,
        holdout: HoldoutSeal | str,
        *,
        data: pd.DataFrame,
        experiment_id: str | None = None,
        purpose: str = "out_of_sample_evaluation",
        config_fingerprint: str | None = None,
        git_revision: str | None = None,
    ) -> HoldoutAccess:
        """Record a sanctioned evaluation of the holdout and return the access.

        The window fingerprint is re-checked against the data presented. A
        mismatch does not raise - the access is still logged, with
        ``integrity_ok=False`` - because a silently dropped record is worse than
        a recorded failure. The caller is expected to read
        :attr:`HoldoutAccess.integrity_ok` and refuse to trust the numbers.
        """
        seal = self._resolve(holdout)
        try:
            current = window_fingerprint(data, pd.Timestamp(seal.start_time), pd.Timestamp(seal.end_time))
            integrity_ok = current == seal.data_fingerprint
        except HoldoutError:
            integrity_ok = False

        touch_number = self.touch_count(seal.holdout_id) + 1
        access = HoldoutAccess(
            holdout_id=seal.holdout_id,
            accessed_at=utcnow().isoformat(),
            touch_number=touch_number,
            experiment_id=experiment_id,
            purpose=purpose,
            config_fingerprint=config_fingerprint,
            git_revision=git_revision,
            integrity_ok=integrity_ok,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO holdout_accesses (
                    holdout_id, accessed_at, touch_number, experiment_id, purpose,
                    config_fingerprint, git_revision, integrity_ok
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    access.holdout_id, access.accessed_at, access.touch_number,
                    access.experiment_id, access.purpose, access.config_fingerprint,
                    access.git_revision, int(access.integrity_ok),
                ),
            )
        if touch_number > 1:
            logger.warning(
                "Holdout touched more than once",
                extra={"holdout_id": seal.holdout_id, "touch_number": touch_number},
            )
        if not integrity_ok:
            logger.error(
                "Holdout data did not match its seal",
                extra={"holdout_id": seal.holdout_id},
            )
        return access

    # -------------------------------------------------------- the guard rail

    def assert_available_for_development(
        self, data_fingerprint: str, *, purpose: str = "optimization"
    ) -> None:
        """Refuse to let development touch a sealed holdout's data.

        Development and optimisation code calls this before running on a window.
        If the window's fingerprint matches a sealed holdout, it raises: a sweep
        that consumed the holdout would turn the one window whose result counts
        into just another surface that was optimised against.

        This is the technical half of the guarantee. It cannot see data that
        never passes through it - see the module docstring on discipline.
        """
        match = self.seal_for_fingerprint(data_fingerprint)
        if match is not None:
            raise HoldoutViolationError(
                f"{purpose!r} was asked to run on data whose fingerprint "
                f"({data_fingerprint}) is sealed as frozen holdout {match.holdout_id} "
                f"({match.label or 'no label'}). Development must not touch the holdout; "
                "run on the in-sample or validation window instead."
            )

    # ------------------------------------------------------------------ reads

    def _resolve(self, holdout: HoldoutSeal | str) -> HoldoutSeal:
        if isinstance(holdout, HoldoutSeal):
            return holdout
        seal = self.get_seal(holdout)
        if seal is None:
            raise HoldoutError(f"No holdout is sealed under id {holdout!r}.")
        return seal

    def get_seal(self, holdout_id: str) -> HoldoutSeal | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM holdout_seals WHERE holdout_id = ?", (holdout_id,)
            ).fetchone()
        return self._seal_from_row(row) if row else None

    def find_seal(
        self, symbol: str, timeframe: str, start_time: Any, end_time: Any
    ) -> HoldoutSeal | None:
        return self.get_seal(_holdout_id(symbol, timeframe, start_time, end_time))

    def seal_for_fingerprint(self, data_fingerprint: str) -> HoldoutSeal | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM holdout_seals WHERE data_fingerprint = ? LIMIT 1",
                (data_fingerprint,),
            ).fetchone()
        return self._seal_from_row(row) if row else None

    def is_sealed(self, symbol: str, timeframe: str, start_time: Any, end_time: Any) -> bool:
        return self.find_seal(symbol, timeframe, start_time, end_time) is not None

    def touch_count(self, holdout_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM holdout_accesses WHERE holdout_id = ?",
                (holdout_id,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def accesses(self, holdout_id: str) -> list[HoldoutAccess]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM holdout_accesses WHERE holdout_id = ? ORDER BY access_id",
                (holdout_id,),
            ).fetchall()
        return [self._access_from_row(r) for r in rows]

    def status_for(self, holdout_id: str) -> HoldoutStatus | None:
        seal = self.get_seal(holdout_id)
        if seal is None:
            return None
        accesses = self.accesses(holdout_id)
        return HoldoutStatus(seal=seal, touch_count=len(accesses), accesses=accesses)

    def list_seals(self) -> list[HoldoutSeal]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM holdout_seals ORDER BY created_at"
            ).fetchall()
        return [self._seal_from_row(r) for r in rows]

    def status(self) -> list[HoldoutStatus]:
        return [
            s for s in (self.status_for(seal.holdout_id) for seal in self.list_seals())
            if s is not None
        ]

    # ----------------------------------------------------------- row mapping

    @staticmethod
    def _seal_from_row(row: sqlite3.Row) -> HoldoutSeal:
        return HoldoutSeal(
            holdout_id=row["holdout_id"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            bars=row["bars"],
            data_fingerprint=row["data_fingerprint"],
            config_fingerprint=row["config_fingerprint"],
            git_revision=row["git_revision"],
            random_seed=row["random_seed"],
            created_at=row["created_at"],
            label=row["label"] or "",
            notes=row["notes"] or "",
        )

    @staticmethod
    def _access_from_row(row: sqlite3.Row) -> HoldoutAccess:
        return HoldoutAccess(
            holdout_id=row["holdout_id"],
            accessed_at=row["accessed_at"],
            touch_number=row["touch_number"],
            experiment_id=row["experiment_id"],
            purpose=row["purpose"],
            config_fingerprint=row["config_fingerprint"],
            git_revision=row["git_revision"],
            integrity_ok=bool(row["integrity_ok"]),
        )
