"""
GCA v0 — Standalone query helpers

Thin convenience module that re-exports ``ArchiveQuery`` from archive.py
and provides a ``load_query(store_path)`` factory for quick ad-hoc
inspection of a GCA store from a notebook or REPL.

Usage::

    from gca.query import load_query
    q = load_query("./gca_store")
    for strat, count in q.top_strategies(10):
        print(f"{count:4d}  {strat.description}")
"""

from __future__ import annotations

from gca.archive import ArchiveQuery
from gca.persistence import Persistence
from gca.registry import StrategyRegistry


def load_query(store_path: str) -> ArchiveQuery:
    """One-liner to open a GCA store read-only.

    Returns an ``ArchiveQuery`` instance backed by the on-disk JSONL files.

    Raises
    ------
    FileNotFoundError
        If ``store_path`` does not exist.  This prevents accidentally
        creating an empty store when the caller simply has a typo in the
        path.
    """
    from pathlib import Path

    p = Path(store_path)
    if not p.exists():
        raise FileNotFoundError(
            f"GCA store not found at {store_path!r}.  "
            "Use GCAStack.from_config() to create a new store."
        )

    persistence = Persistence(store_path)
    registry = StrategyRegistry(persistence)
    return ArchiveQuery(persistence, registry)


__all__ = ["ArchiveQuery", "load_query"]
