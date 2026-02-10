"""
GCA v0 — Persistence layer

Layer 2 of the GCA plan.  Implements append-only JSONL storage and
content-addressed code storage.

Truth model
-----------
* Event logs (``program_evaluated.jsonl``, ``strategy_occurrence.jsonl``)
  are the **source of truth**.
* The strategy catalog (``strategies.jsonl``) is materialised state that
  *can* be rebuilt from logs if needed.
* Every record carries a ``schema_version`` for forward-compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gca.schemas import (
    ProgramEvaluatedEvent,
    Strategy,
    StrategyOccurrence,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory layout (Adjustment 3 — code storage convention)
# ---------------------------------------------------------------------------
#
# <store_path>/
#   programs/
#     <sha256(code)>.py           # content-addressed source files
#   logs/
#     program_evaluated.jsonl     # append-only events
#     strategy_occurrence.jsonl   # append-only join events
#   catalogs/
#     strategies.jsonl            # append-only, deduped on load
#   metadata.json                 # archive metadata + versions + run list
#
# NOTE: GCA keeps its own copy of program source code, independent of
# OpenEvolve's internal storage (db_path/programs/<id>.json).  The
# integration layer (notify_gca) copies code into this store at
# notification time.  The key is sha256(stripped_code), so identical
# programs share a single file regardless of how many times they appear.
# ---------------------------------------------------------------------------


class Persistence:
    """Handles all file I/O for the GCA store.

    Every write method is append-only.  Reads materialise in-memory
    collections from the JSONL files.
    """

    def __init__(self, store_path: str | Path) -> None:
        self.store_path = Path(store_path)
        self._ensure_directories()
        self._ensure_metadata()

    # -- paths -------------------------------------------------------------

    @property
    def programs_dir(self) -> Path:
        return self.store_path / "programs"

    @property
    def logs_dir(self) -> Path:
        return self.store_path / "logs"

    @property
    def catalogs_dir(self) -> Path:
        return self.store_path / "catalogs"

    @property
    def program_events_path(self) -> Path:
        return self.logs_dir / "program_evaluated.jsonl"

    @property
    def occurrence_events_path(self) -> Path:
        return self.logs_dir / "strategy_occurrence.jsonl"

    @property
    def strategies_path(self) -> Path:
        return self.catalogs_dir / "strategies.jsonl"

    @property
    def metadata_path(self) -> Path:
        return self.store_path / "metadata.json"

    # -- bootstrap ---------------------------------------------------------

    def _ensure_directories(self) -> None:
        for d in (self.programs_dir, self.logs_dir, self.catalogs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _ensure_metadata(self) -> None:
        if not self.metadata_path.exists():
            self._write_metadata_initial()

    def _write_metadata_initial(self) -> None:
        meta = {
            "gca_version": "0.1.0",
            "created_at": time.time(),
            "last_updated": time.time(),
            "runs_included": [],
            "log_files": {
                "program_evaluated": "logs/program_evaluated.jsonl",
                "strategy_occurrence": "logs/strategy_occurrence.jsonl",
                "strategies": "catalogs/strategies.jsonl",
            },
        }
        self._atomic_json_write(self.metadata_path, meta)

    # -- code storage (content-addressed) ----------------------------------

    def store_code(self, code_hash: str, code: str, suffix: str = ".py") -> str:
        """Store program source in GCA's content-addressed store.

        The file is written to ``<store_path>/programs/<code_hash><suffix>``.
        If a file with the same hash already exists, the write is skipped
        (idempotent).

        This is GCA's *own* copy of the code — separate from OpenEvolve's
        internal storage (``db_path/programs/<id>.json``).  The integration
        layer (``notify_gca``) calls this to copy code at notification time.

        Returns the **store-relative** path (e.g., ``"programs/abc123.py"``),
        which is stored in ``ProgramRef.code_path``.
        """
        filename = f"{code_hash}{suffix}"
        dest = self.programs_dir / filename
        if not dest.exists():
            dest.write_text(code, encoding="utf-8")
        return str(Path("programs") / filename)

    def load_code(self, code_path: str) -> str:
        """Load program source from a store-relative path.

        Parameters
        ----------
        code_path : str
            Relative path within the GCA store, e.g. ``"programs/abc123.py"``.
            This is the value stored in ``ProgramRef.code_path``.

        Raises
        ------
        FileNotFoundError
            If the code file does not exist (e.g. the store was corrupted
            or the program was never stored).
        """
        full = self.store_path / code_path
        return full.read_text(encoding="utf-8")

    # -- append-only event writers -----------------------------------------

    def append_program_event(self, event: ProgramEvaluatedEvent) -> None:
        self._append_jsonl(self.program_events_path, _dataclass_to_dict(event))
        self._touch_metadata()

    def append_occurrence_event(self, occ: StrategyOccurrence) -> None:
        self._append_jsonl(self.occurrence_events_path, _dataclass_to_dict(occ))
        self._touch_metadata()

    def append_strategy(self, strat: Strategy) -> None:
        self._append_jsonl(self.strategies_path, _dataclass_to_dict(strat))
        self._touch_metadata()

    # -- loaders (materialise from JSONL) ----------------------------------

    def load_program_events(self) -> list[dict[str, Any]]:
        return self._load_jsonl(self.program_events_path)

    def load_occurrence_events(self) -> list[dict[str, Any]]:
        return self._load_jsonl(self.occurrence_events_path)

    def load_strategies(self) -> list[dict[str, Any]]:
        return self._load_jsonl(self.strategies_path)

    # -- metadata helpers --------------------------------------------------

    def load_metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def add_run(self, run_id: str) -> None:
        meta = self.load_metadata()
        if run_id not in meta["runs_included"]:
            meta["runs_included"].append(run_id)
            meta["last_updated"] = time.time()
            self._atomic_json_write(self.metadata_path, meta)

    def _touch_metadata(self) -> None:
        """Lightweight timestamp bump (avoids rewriting full metadata)."""
        try:
            meta = self.load_metadata()
            meta["last_updated"] = time.time()
            self._atomic_json_write(self.metadata_path, meta)
        except Exception:
            pass  # best-effort

    # -- low-level I/O -----------------------------------------------------

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL at %s:%d", path, lineno)
        return records

    @staticmethod
    def _atomic_json_write(path: Path, data: dict[str, Any]) -> None:
        """Write JSON atomically via temp-file + rename."""
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Recursively convert a (possibly nested) frozen dataclass to a dict."""
    return asdict(obj)
