"""
GCA v0 — Evolution observer (integration with OpenEvolve)

Layer 7 of the GCA plan.

Single integration touchpoint: ``EvolutionObserver.on_program_evaluated``.

Design constraints
------------------
* If GCA is disabled → no behaviour change in OpenEvolve.
* Extraction does *not* block the evolution loop (async queue + worker).
* Observer is **post-evaluation** only.

Context resolution is NOT done here — it's the caller's responsibility
(``notify_gca`` in ``openevolve_integration.py``) to resolve parent_fitness,
island_id, and code_path before constructing the event.  The observer is a
pure consumer of fully-resolved ``ProgramEvaluatedEvent`` objects.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from gca.archive import GlobalCollectiveArchive
from gca.extractor import StrategyExtractor
from gca.persistence import Persistence
from gca.policy import ExtractionPolicy
from gca.registry import StrategyRegistry
from gca.schemas import (
    ProgramEvaluatedEvent,
    StrategyOccurrence,
)

logger = logging.getLogger(__name__)


class EvolutionObserver:
    """Post-evaluation observer that records events and (optionally)
    extracts strategies.

    Parameters
    ----------
    gca : GlobalCollectiveArchive
        The archive to record events and occurrences into.
    registry : StrategyRegistry
        Strategy catalog for dedup + registration.
    persistence : Persistence
        For content-addressed code storage.
    extractor : StrategyExtractor
        The strategy extractor to invoke.
    policy : ExtractionPolicy
        Decides which programs undergo extraction.
    async_enabled : bool
        If ``True``, extraction is queued and processed by a background
        worker so it never blocks the evolution loop.
    queue_size : int
        Bounded queue size for async mode (backpressure).
    """

    def __init__(
        self,
        gca: GlobalCollectiveArchive,
        registry: StrategyRegistry,
        persistence: Persistence,
        extractor: StrategyExtractor,
        policy: ExtractionPolicy,
        *,
        async_enabled: bool = True,
        queue_size: int = 128,
    ) -> None:
        self._gca = gca
        self._registry = registry
        self._persistence = persistence
        self._extractor = extractor
        self._policy = policy
        self._async_enabled = async_enabled
        self._queue_size = queue_size

        # Async queue + worker — created lazily in start() so that
        # asyncio.Queue is instantiated inside a running event loop.
        self._queue: asyncio.Queue[ProgramEvaluatedEvent | None] | None = None
        self._worker_task: asyncio.Task | None = None

    # -- public: main entry point -----------------------------------------

    async def on_program_evaluated(
        self,
        event: ProgramEvaluatedEvent,
        parent_fitness: float | None = None,
    ) -> None:
        """Called every time OpenEvolve finishes evaluating a program.

        Steps
        -----
        0. Persist raw program event (always).
        1. Store code content-addressed.
        2. Decide whether to extract.
        3. Extract + register + record occurrences (sync or async).
        """
        # 0) Always persist the raw event
        self._gca.record_program_event(event)

        # 1) Check policy
        should, reason = self._policy.should_extract(event)
        if not should:
            return

        logger.debug(
            "Extraction triggered for %s (reason=%s)",
            event.program.program_id,
            reason,
        )

        if self._async_enabled and self._queue is not None:
            # Enqueue for background processing
            try:
                self._queue.put_nowait((event, parent_fitness, reason))
            except asyncio.QueueFull:
                logger.warning("GCA extraction queue full — dropping event %s", event.program.program_id)
        else:
            # Synchronous extraction
            await self._process_extraction(event, parent_fitness, reason)

    # -- async worker infrastructure ---------------------------------------

    async def start(self) -> None:
        """Start the background extraction worker (call once at startup).

        Creates the ``asyncio.Queue`` here (not in ``__init__``) so that
        construction can happen outside a running event loop.
        """
        if self._async_enabled:
            self._queue = asyncio.Queue(maxsize=self._queue_size)
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("GCA extraction worker started")

    async def stop(self) -> None:
        """Gracefully shut down the background worker."""
        if self._queue is not None:
            await self._queue.put(None)  # sentinel
        if self._worker_task is not None:
            await self._worker_task
            logger.info("GCA extraction worker stopped")

    async def _worker_loop(self) -> None:
        """Consume events from the queue and process them."""
        while True:
            item = await self._queue.get()
            if item is None:
                break  # sentinel → shutdown
            event, parent_fitness, reason = item
            try:
                await self._process_extraction(event, parent_fitness, reason)
            except Exception:
                logger.exception(
                    "GCA extraction worker error for program %s",
                    event.program.program_id,
                )

    # -- core extraction logic ---------------------------------------------

    async def _process_extraction(
        self,
        event: ProgramEvaluatedEvent,
        parent_fitness: float | None,
        trigger_reason: str,
    ) -> None:
        """Run the extractor and persist results."""
        # Load code
        try:
            code = self._persistence.load_code(event.program.code_path)
        except FileNotFoundError:
            logger.warning(
                "Code not found for %s at %s — skipping extraction",
                event.program.program_id,
                event.program.code_path,
            )
            return

        # Extract strategies (0..N)
        extracted_list = await self._extractor.extract_from_solution(
            program_code=code,
            program_ref=event.program,
            evaluation=event.evaluation,
            artifacts=event.artifacts,
        )

        if not extracted_list:
            logger.debug("No strategies extracted from %s", event.program.program_id)
            return

        # Register + record occurrences
        extractor_tag = f"{self._extractor.name}@{self._extractor.version}"
        for extracted in extracted_list:
            strat = self._registry.register(
                extracted,
                first_observed={
                    "program_id": event.program.program_id,
                    "run_id": event.program.run_id,
                    "iteration": event.program.iteration,
                    "ts": event.ts,
                },
                extraction_method=extractor_tag,
            )

            # Determine comparison context
            if parent_fitness is not None:
                comparison_context = "parent"
            else:
                comparison_context = "none"

            occ = StrategyOccurrence(
                schema_version="gca.v0.strategy_occurrence.v1",
                ts=time.time(),
                strategy_id=strat.strategy_id,
                program_id=event.program.program_id,
                run_id=event.program.run_id,
                island_id=event.program.island_id,
                iteration=event.program.iteration,
                extractor_version=f"{self._extractor.name}@{self._extractor.version}",
                extraction_confidence=extracted.confidence,
                evidence={
                    **extracted.evidence,
                    "trigger_reason": trigger_reason,
                },
                program_fitness=event.evaluation.fitness,
                parent_fitness=parent_fitness,
                comparison_context=comparison_context,
            )
            self._gca.record_occurrence(occ)

        logger.info(
            "Extracted %d strategies from program %s",
            len(extracted_list),
            event.program.program_id,
        )
