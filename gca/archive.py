"""
GCA v0 — GlobalCollectiveArchive (archive core)

Layer 6 of the GCA plan.

Orchestrates recording of program events, strategy occurrences, and
provides the ``ArchiveQuery`` read-only interface.
"""

from __future__ import annotations

import logging
from typing import Any

from gca.persistence import Persistence
from gca.registry import StrategyRegistry
from gca.schemas import (
    ProgramEvaluatedEvent,
    Strategy,
    StrategyOccurrence,
)

logger = logging.getLogger(__name__)


class GlobalCollectiveArchive:
    """Central orchestrator for all GCA persistence.

    * Records raw program evaluation events (always).
    * Records strategy occurrences (when extraction runs).
    * Exposes ``query`` for read-only inspection.
    """

    def __init__(
        self,
        store_path: str,
        registry: StrategyRegistry,
        persistence: Persistence,
    ) -> None:
        self._store_path = store_path
        self._registry = registry
        self._persistence = persistence

        # In-memory indices (populated lazily)
        self._occurrences: list[StrategyOccurrence] | None = None

    # -- write path --------------------------------------------------------

    def record_program_event(self, event: ProgramEvaluatedEvent) -> None:
        """Persist a raw program-evaluated event (always called)."""
        self._persistence.append_program_event(event)

    def record_occurrence(self, occ: StrategyOccurrence) -> None:
        """Persist a strategy-occurrence event."""
        self._persistence.append_occurrence_event(occ)
        # Invalidate in-memory cache
        self._occurrences = None

    # -- read / query path -------------------------------------------------

    @property
    def query(self) -> ArchiveQuery:
        """Convenience accessor for the read-only query surface."""
        return ArchiveQuery(self._persistence, self._registry)


class ArchiveQuery:
    """Read-only query helpers over the GCA store.

    Minimal but sufficient to demonstrate "queryable memory":
    * ``list_strategies``
    * ``get_strategy``
    * ``list_occurrences``
    * ``strategies_in_run``
    * ``strategy_usage_count``
    * ``strategy_usage_counts`` (bulk)
    * ``top_strategies``
    * ``timeline`` (strategy counts per iteration for a run)
    """

    def __init__(self, persistence: Persistence, registry: StrategyRegistry) -> None:
        self._persistence = persistence
        self._registry = registry

    # -- strategies --------------------------------------------------------

    def get_strategy(self, strategy_id: str) -> Strategy | None:
        return self._registry.get(strategy_id)

    def list_strategies(self, limit: int = 50) -> list[Strategy]:
        return self._registry.list_all(limit=limit)

    # -- occurrences -------------------------------------------------------

    def list_occurrences(
        self,
        strategy_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return raw occurrence dicts for a given strategy."""
        all_occ = self._persistence.load_occurrence_events()
        filtered = [o for o in all_occ if o.get("strategy_id") == strategy_id]
        return filtered[:limit]

    def strategies_in_run(self, run_id: str) -> list[Strategy]:
        """All distinct strategies observed in a given run."""
        all_occ = self._persistence.load_occurrence_events()
        sids = {o["strategy_id"] for o in all_occ if o.get("run_id") == run_id}
        return [s for s in self._registry.list_all() if s.strategy_id in sids]

    def strategy_usage_count(self, strategy_id: str) -> int:
        all_occ = self._persistence.load_occurrence_events()
        return sum(1 for o in all_occ if o.get("strategy_id") == strategy_id)

    def strategy_usage_counts(self) -> dict[str, int]:
        """Return {strategy_id: count} for all strategies."""
        counts: dict[str, int] = {}
        for occ in self._persistence.load_occurrence_events():
            sid = occ.get("strategy_id", "")
            counts[sid] = counts.get(sid, 0) + 1
        return counts

    def top_strategies(self, limit: int = 20) -> list[tuple[Strategy, int]]:
        """Strategies sorted by usage count (descending)."""
        counts = self.strategy_usage_counts()
        sorted_ids = sorted(counts, key=counts.get, reverse=True)[:limit]
        result: list[tuple[Strategy, int]] = []
        for sid in sorted_ids:
            strat = self._registry.get(sid)
            if strat:
                result.append((strat, counts[sid]))
        return result

    def top_strategies_by_fitness(self, limit: int = 3) -> list[tuple[Strategy, float, int]]:
        """Strategies ranked by best associated program fitness (descending).

        Returns a list of (strategy, best_fitness, occurrence_count) tuples.
        """
        all_occ = self._persistence.load_occurrence_events()

        # Group by strategy_id: track max fitness and count
        best_fitness: dict[str, float] = {}
        counts: dict[str, int] = {}
        for occ in all_occ:
            sid = occ.get("strategy_id", "")
            fitness = occ.get("program_fitness", 0.0)
            if not isinstance(fitness, (int, float)):
                continue
            if sid not in best_fitness or fitness > best_fitness[sid]:
                best_fitness[sid] = fitness
            counts[sid] = counts.get(sid, 0) + 1

        # Sort by best fitness descending
        sorted_ids = sorted(best_fitness, key=best_fitness.get, reverse=True)[:limit]

        result: list[tuple[Strategy, float, int]] = []
        for sid in sorted_ids:
            strat = self._registry.get(sid)
            if strat:
                result.append((strat, best_fitness[sid], counts.get(sid, 0)))
        return result

    def programs_with_strategy(self, strategy_id: str) -> list[str]:
        """Return program IDs that exhibit a given strategy."""
        all_occ = self._persistence.load_occurrence_events()
        return [o["program_id"] for o in all_occ if o.get("strategy_id") == strategy_id]

    def timeline(self, run_id: str) -> dict[int, dict[str, int]]:
        """Strategy counts per iteration for a run.

        Returns ``{iteration: {strategy_id: count}}``.
        """
        all_occ = self._persistence.load_occurrence_events()
        timeline: dict[int, dict[str, int]] = {}
        for occ in all_occ:
            if occ.get("run_id") != run_id:
                continue
            it = occ.get("iteration")
            if it is None:
                continue
            it = int(it)
            sid = occ.get("strategy_id", "")
            timeline.setdefault(it, {})
            timeline[it][sid] = timeline[it].get(sid, 0) + 1
        return dict(sorted(timeline.items()))
