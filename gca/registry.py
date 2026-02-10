"""
GCA v0 — Strategy registry & deduplication

Layer 5 of the GCA plan.

Maintains a de-duplicated catalog of strategies with stable IDs.

v0 dedup = **exact match on canonical text**.
Future: add embedding-based merge behind a flag.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from gca.persistence import Persistence
from gca.schemas import (
    ExtractedStrategy,
    Strategy,
    canonicalize_strategy,
    strategy_id_from_text,
)

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """In-memory strategy catalog backed by append-only JSONL.

    On init the existing catalog is loaded from disk.  New strategies are
    appended; duplicates (by canonical text) are silently de-duped.
    """

    def __init__(self, persistence: Persistence) -> None:
        self._persistence = persistence
        self._strategies: dict[str, Strategy] = {}
        self._load_existing()

    # -- public API --------------------------------------------------------

    def register(
        self,
        extracted: ExtractedStrategy,
        first_observed: dict[str, Any],
        *,
        extraction_method: str = "unknown",
    ) -> Strategy:
        """Register an extracted strategy; returns the canonical ``Strategy``.

        If the strategy already exists (by canonical text hash), the
        existing record is returned without modification.

        Parameters
        ----------
        extraction_method : str
            Identifier for the extractor that produced this strategy,
            e.g. ``"llm_solution_analysis@0.1.0"`` or
            ``"heuristic_structural@0.1.0"``.  Stored in the catalog so
            downstream consumers know *how* the strategy was found.
        """
        sid = strategy_id_from_text(extracted.description)

        if sid in self._strategies:
            logger.debug("Strategy already known: %s", sid[:12])
            return self._strategies[sid]

        strat = Strategy(
            strategy_id=sid,
            description=extracted.description.strip(),
            tags=list(extracted.tags) if extracted.tags else [],
            extraction_method=extraction_method,
            first_observed=dict(first_observed),
            created_at=time.time(),
        )

        self._strategies[sid] = strat
        self._persistence.append_strategy(strat)
        logger.info("New strategy registered: %s — %s", sid[:12], strat.description[:80])
        return strat

    def get(self, strategy_id: str) -> Strategy | None:
        return self._strategies.get(strategy_id)

    def list_all(self, limit: int = 200) -> list[Strategy]:
        return list(self._strategies.values())[:limit]

    @property
    def size(self) -> int:
        return len(self._strategies)

    # -- internal ----------------------------------------------------------

    def _load_existing(self) -> None:
        """Materialise the catalog from the JSONL file (dedup on load)."""
        raw_records = self._persistence.load_strategies()
        for rec in raw_records:
            sid = rec.get("strategy_id", "")
            if sid and sid not in self._strategies:
                self._strategies[sid] = Strategy(
                    strategy_id=sid,
                    description=rec.get("description", ""),
                    tags=rec.get("tags", []),
                    extraction_method=rec.get("extraction_method", "unknown"),
                    first_observed=rec.get("first_observed", {}),
                    created_at=rec.get("created_at", 0.0),
                )
        logger.info("Loaded %d existing strategies from catalog", len(self._strategies))
