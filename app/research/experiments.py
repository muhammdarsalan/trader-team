"""SQLite experiment tracking.

A research process that cannot say what it has already tried is not a research
process, it is a sequence of anecdotes. Two things are recorded here that the
per-run JSON files in ``experiments/`` cannot express on their own:

**Identity.** An experiment id is derived from what actually determines the
result - the configuration, the data fingerprint, the code revision, the seed
and the study design - rather than from a counter. Run the same study twice and
you get the same id and one row, not two rows that look like independent
confirmations. Change one parameter and you get a different id, automatically.

**Accumulated search.** Every configuration ever evaluated against a given data
checksum is counted. That count is the input the overfitting diagnostics need
and the one nobody remembers honestly: it is always larger than it felt,
because the twelve variants tried last week still happened.

The database is disposable. Everything in it is derived from the configuration
and the data, so deleting it loses history, not results.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.utils.logging import get_logger
from app.utils.paths import ensure_dir, experiments_dir
from app.utils.timeutils import utcnow

logger = get_logger(__name__)

SCHEMA_VERSION = 1

#: Deliberately plain SQL. The whole store is a few hundred rows; an ORM here
#: would be more machinery than the thing it manages.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id     TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    kind              TEXT NOT NULL,
    symbol            TEXT,
    timeframe         TEXT,
    period_start      TEXT,
    period_end        TEXT,
    bars              INTEGER,
    data_provider     TEXT,
    data_checksum     TEXT,
    data_quality      TEXT,
    git_revision      TEXT,
    random_seed       INTEGER,
    config_hash       TEXT,
    config_json       TEXT,
    spec_json         TEXT,
    verdict           TEXT,
    notes             TEXT
);

CREATE INDEX IF NOT EXISTS idx_experiments_checksum ON experiments(data_checksum);

CREATE TABLE IF NOT EXISTS segments (
    experiment_id  TEXT NOT NULL,
    name           TEXT NOT NULL,
    role           TEXT NOT NULL,
    start_time     TEXT,
    end_time       TEXT,
    bars           INTEGER,
    trades         INTEGER,
    data_quality   TEXT,
    variant        TEXT,
    PRIMARY KEY (experiment_id, name),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metrics (
    experiment_id  TEXT NOT NULL,
    segment        TEXT NOT NULL,
    metric         TEXT NOT NULL,
    value          REAL,
    PRIMARY KEY (experiment_id, segment, metric),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS findings (
    experiment_id  TEXT NOT NULL,
    code           TEXT NOT NULL,
    severity       TEXT NOT NULL,
    message        TEXT NOT NULL,
    evidence_json  TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS trials (
    trial_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id  TEXT NOT NULL,
    data_checksum  TEXT NOT NULL,
    variant        TEXT NOT NULL,
    config_hash    TEXT NOT NULL,
    segment        TEXT,
    objective      TEXT,
    value          REAL,
    created_at     TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_trials_checksum ON trials(data_checksum);
CREATE UNIQUE INDEX IF NOT EXISTS idx_trials_unique
    ON trials(data_checksum, config_hash, segment);

CREATE TABLE IF NOT EXISTS artifacts (
    experiment_id  TEXT NOT NULL,
    name           TEXT NOT NULL,
    path           TEXT NOT NULL,
    PRIMARY KEY (experiment_id, name),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);
"""


# ------------------------------------------------------------------ identity


def stable_hash(payload: Any, length: int = 12) -> str:
    """A hash that does not move between processes.

    Python's ``hash()`` is salted per process, so an id built from it would
    change every run and defeat the entire point. This is SHA-256 over
    canonical JSON: same content, same digest, on any machine, forever.
    """
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def config_fingerprint(config_snapshot: dict[str, Any]) -> str:
    """Fingerprint of the settings that can change a result."""
    return stable_hash(config_snapshot)


def reproducible_experiment_id(
    *,
    config_snapshot: dict[str, Any],
    data_checksum: str,
    git_revision: str,
    random_seed: int,
    spec: dict[str, Any] | None = None,
    prefix: str = "RES",
) -> str:
    """A deterministic id for a study.

    Everything that can change the outcome goes into the hash: configuration,
    data, code, seed and the study design itself. Everything that cannot -
    wall-clock time, machine, who ran it - stays out. So the same study run
    twice produces one id, and any real difference produces a different one
    without anybody having to notice and bump a counter.

    The date is carried in the id for human legibility only. It is not part of
    the hash, so re-running a study tomorrow updates the same record rather
    than forking it.
    """
    digest = stable_hash(
        {
            "config": config_snapshot,
            "data": data_checksum,
            "code": git_revision,
            "seed": random_seed,
            "spec": spec or {},
        }
    )
    return f"{prefix}-{digest}"


# --------------------------------------------------------------------- record


@dataclass
class ExperimentRecord:
    """The metadata row for one study."""

    experiment_id: str
    kind: str
    symbol: str | None = None
    timeframe: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    bars: int | None = None
    data_provider: str | None = None
    data_checksum: str | None = None
    data_quality: str | None = None
    git_revision: str | None = None
    random_seed: int | None = None
    config_hash: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    spec: dict[str, Any] = field(default_factory=dict)
    verdict: str | None = None
    notes: str | None = None


# ---------------------------------------------------------------------- store


class ExperimentStore:
    """SQLite-backed record of every study and every configuration tried."""

    def __init__(self, path: Path | None = None) -> None:
        """
        Args:
            path: database file. Defaults to ``experiments/experiments.db``,
                which is gitignored - the database is derived data, and
                committing it would put one machine's search history into
                everyone else's overfitting arithmetic.
        """
        self.path = Path(path) if path is not None else experiments_dir() / "experiments.db"
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
            row = connection.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
                )
            elif row["version"] != SCHEMA_VERSION:
                logger.warning(
                    "Experiment database was written by a different schema version",
                    extra={"found": row["version"], "expected": SCHEMA_VERSION},
                )

    # ------------------------------------------------------------------ writes

    def record_experiment(self, record: ExperimentRecord) -> str:
        """Insert or update one study. Returns its id.

        Re-recording an existing id updates it rather than raising, because a
        deterministic id means a re-run is the *same* experiment. Its
        ``created_at`` is preserved so the first time it was run stays visible.
        """
        now = utcnow().isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    experiment_id, created_at, updated_at, kind, symbol, timeframe,
                    period_start, period_end, bars, data_provider, data_checksum,
                    data_quality, git_revision, random_seed, config_hash, config_json,
                    spec_json, verdict, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    updated_at   = excluded.updated_at,
                    data_quality = excluded.data_quality,
                    verdict      = excluded.verdict,
                    notes        = excluded.notes
                """,
                (
                    record.experiment_id, now, now, record.kind, record.symbol,
                    record.timeframe, record.period_start, record.period_end, record.bars,
                    record.data_provider, record.data_checksum, record.data_quality,
                    record.git_revision, record.random_seed,
                    record.config_hash or config_fingerprint(record.config_snapshot),
                    json.dumps(record.config_snapshot, default=str),
                    json.dumps(record.spec, default=str),
                    record.verdict, record.notes,
                ),
            )
        return record.experiment_id

    def record_segment(
        self,
        experiment_id: str,
        *,
        name: str,
        role: str,
        start_time: str | None,
        end_time: str | None,
        bars: int,
        trades: int,
        data_quality: str,
        variant: str = "baseline",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO segments (
                    experiment_id, name, role, start_time, end_time, bars, trades,
                    data_quality, variant
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(experiment_id, name) DO UPDATE SET
                    role=excluded.role, start_time=excluded.start_time,
                    end_time=excluded.end_time, bars=excluded.bars,
                    trades=excluded.trades, data_quality=excluded.data_quality,
                    variant=excluded.variant
                """,
                (experiment_id, name, role, start_time, end_time, bars, trades,
                 data_quality, variant),
            )

    def record_metrics(
        self, experiment_id: str, segment: str, metrics: dict[str, Any]
    ) -> None:
        """Store the numeric metrics for one segment.

        Non-numeric entries are skipped rather than stringified: a metrics table
        holding the word "inf" next to real numbers is a table nobody can query.
        """
        rows = [
            (experiment_id, segment, key, float(value))
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and float(value) == float(value)          # drop NaN
            and abs(float(value)) != float("inf")
        ]
        if not rows:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO metrics (experiment_id, segment, metric, value)
                VALUES (?,?,?,?)
                ON CONFLICT(experiment_id, segment, metric) DO UPDATE SET
                    value = excluded.value
                """,
                rows,
            )

    def record_findings(self, experiment_id: str, findings: list[dict[str, Any]]) -> None:
        if not findings:
            return
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM findings WHERE experiment_id = ?", (experiment_id,)
            )
            connection.executemany(
                """
                INSERT INTO findings (experiment_id, code, severity, message, evidence_json)
                VALUES (?,?,?,?,?)
                """,
                [
                    (
                        experiment_id, f.get("code", ""), f.get("severity", "INFO"),
                        f.get("message", ""), json.dumps(f.get("evidence", {}), default=str),
                    )
                    for f in findings
                ],
            )

    def record_trial(
        self,
        experiment_id: str,
        *,
        data_checksum: str,
        variant: str,
        config_hash: str,
        segment: str | None = None,
        objective: str | None = None,
        value: float | None = None,
    ) -> None:
        """Record that one configuration was evaluated against this data.

        The unique index on (data_checksum, config_hash, segment) is doing the
        important work: evaluating the same configuration on the same window
        twice is one trial, not two. Overstating the search is as wrong as
        understating it.
        """
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO trials (
                    experiment_id, data_checksum, variant, config_hash, segment,
                    objective, value, created_at
                ) VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(data_checksum, config_hash, segment) DO UPDATE SET
                    value = excluded.value, objective = excluded.objective
                """,
                (
                    experiment_id, data_checksum, variant, config_hash,
                    segment or "", objective, value, utcnow().isoformat(),
                ),
            )

    def record_artifact(self, experiment_id: str, name: str, path: Path | str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (experiment_id, name, path) VALUES (?,?,?)
                ON CONFLICT(experiment_id, name) DO UPDATE SET path = excluded.path
                """,
                (experiment_id, name, str(path)),
            )

    # ------------------------------------------------------------------- reads

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiments ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def count_trials(self, data_checksum: str) -> int:
        """Distinct configurations ever evaluated against this exact data.

        This is the ``trials`` input to the overfitting diagnostics, and taking
        it from the database rather than from the current session is the whole
        point: last week's twelve variants still happened, and the expected best
        result of a search does not reset when the process restarts.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(DISTINCT config_hash) AS n FROM trials WHERE data_checksum = ?",
                (data_checksum,),
            ).fetchone()
        return int(row["n"] or 0)

    def metrics_for(self, experiment_id: str, segment: str | None = None) -> dict[str, float]:
        query = "SELECT segment, metric, value FROM metrics WHERE experiment_id = ?"
        params: list[Any] = [experiment_id]
        if segment is not None:
            query += " AND segment = ?"
            params.append(segment)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        if segment is not None:
            return {row["metric"]: row["value"] for row in rows}
        return {f"{row['segment']}.{row['metric']}": row["value"] for row in rows}

    def segments_for(self, experiment_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM segments WHERE experiment_id = ? ORDER BY start_time",
                (experiment_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def findings_for(self, experiment_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM findings WHERE experiment_id = ?", (experiment_id,)
            ).fetchall()
        return [dict(row) for row in rows]
