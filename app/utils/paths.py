"""Canonical project paths.

Every module resolves filesystem locations through here so that nothing
hard-codes an absolute path and tests can redirect the data root.
"""

from __future__ import annotations

import os
from pathlib import Path

# app/utils/paths.py -> app/utils -> app -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
TESTS_DIR: Path = PROJECT_ROOT / "tests"


def data_root() -> Path:
    """Root of the data tree.

    Overridable with the ``GTP_DATA_ROOT`` environment variable, which keeps
    tests off the real dataset and lets you park large files on another drive.
    """
    override = os.environ.get("GTP_DATA_ROOT")
    return Path(override).resolve() if override else PROJECT_ROOT / "data"


def raw_dir() -> Path:
    """Vendor-shaped files exactly as downloaded or hand-dropped by the user."""
    return data_root() / "raw"


def cache_dir() -> Path:
    """Normalised, canonical-schema parquet/csv cache."""
    return data_root() / "cache"


def external_dir() -> Path:
    """Third-party datasets that are not price series (calendars, etc.)."""
    return data_root() / "external"


def reports_dir() -> Path:
    override = os.environ.get("GTP_REPORTS_DIR")
    return Path(override).resolve() if override else PROJECT_ROOT / "reports"


def experiments_dir() -> Path:
    override = os.environ.get("GTP_EXPERIMENTS_DIR")
    return Path(override).resolve() if override else PROJECT_ROOT / "experiments"


def logs_dir() -> Path:
    override = os.environ.get("GTP_LOGS_DIR")
    return Path(override).resolve() if override else PROJECT_ROOT / "logs"


def paper_state_dir() -> Path:
    """Where paper-trading state files live.

    Overridable with ``GTP_PAPER_STATE_DIR``. It sits under the data root by
    default so a test's redirected root also redirects paper state.
    """
    override = os.environ.get("GTP_PAPER_STATE_DIR")
    return Path(override).resolve() if override else data_root() / "paper"


def paper_state_path(symbol: str, timeframe: str) -> Path:
    """The canonical state file for one paper-trading session.

    The runner and the dashboard both resolve the path through here. If they
    each built their own, the dashboard would report NO_DATA while the runner was
    happily trading into a different file, and the disagreement would look like a
    bug in the engine.
    """
    return paper_state_dir() / f"{symbol.upper()}_{timeframe.upper()}.json"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if absent, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
