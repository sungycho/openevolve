"""
GCA v0 — OpenEvolve integration helpers

Provides the minimal bridge between OpenEvolve's data model and the GCA
observer pipeline.  The goal is to keep intrusion into OE's codebase to a
bare minimum — ideally a single ``if gca_observer: ...`` block.

Program context resolution (§7.2 of design doc)
-------------------------------------------------
When building a ``ProgramEvaluatedEvent``, ``notify_gca`` resolves optional
fields that require database access:

1. **parent_fitness**: Looked up via ``database.get(parent_id).metrics``.
   Returns ``None`` when the parent doesn't exist or has been evicted.
2. **island_id**: Read from ``child_program.metadata["island"]`` which is
   set by ``database.add()`` *before* this hook fires.  May be ``None``
   for the initial seed or manual runs.
3. **code_path**: The integration layer copies the program's source code
   into GCA's own content-addressed store (``gca_store/programs/<sha256>.py``)
   and records the store-relative path.  GCA never reads from OE's internal
   ``db_path/programs/<id>.json`` files.

Usage in OpenEvolve's ``process_parallel.py`` (after program added to DB)::

    if self.gca_observer is not None:
        from gca.openevolve_integration import notify_gca
        await notify_gca(
            self.gca_observer,
            child_program,
            result,
            self.database,
            run_id=self._gca_run_id,
            iteration=completed_iteration,
        )
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any

from gca.schemas import (
    EvaluationSummary,
    ProgramEvaluatedEvent,
    ProgramRef,
    code_hash,
)

if TYPE_CHECKING:
    from gca.observer import EvolutionObserver

logger = logging.getLogger(__name__)


async def notify_gca(
    observer: EvolutionObserver,
    child_program,               # openevolve.database.Program
    result,                      # SerializableResult
    database,                    # ProgramDatabase
    *,
    run_id: str,
    iteration: int,
    config_hash: str = "",
) -> None:
    """Translate OE objects → GCA event and notify the observer.

    This is the **single adapter** between OpenEvolve's data model and GCA's
    schemas.  It resolves all optional context that requires DB access:

    * **parent_fitness** — looked up from ``database.get(parent_id)``; will
      be ``None`` if the parent was evicted or doesn't exist (Adjustment 1).
    * **island_id** — read from ``child_program.metadata["island"]`` which
      ``database.add()`` has already set by the time this runs (Adjustment 2).
    * **code_path** — code is copied into GCA's content-addressed store
      (``gca_store/programs/<sha256>.py``); GCA never reads OE's internal
      program JSON files (Adjustment 3).

    This function catches **all** exceptions so it never disrupts the
    evolution loop.
    """
    try:
        from openevolve.utils.metrics_utils import get_fitness_score
        fitness = get_fitness_score(child_program.metrics)

        # -- Adjustment 3: Content-addressed code storage --------------------
        # Copy code into GCA's own store.  GCA does NOT read from OE's
        # db_path/programs/<id>.json — it keeps its own copy keyed by
        # sha256(code).  Idempotent: same code → same hash → no rewrite.
        chash = code_hash(child_program.code)
        code_rel_path = observer._persistence.store_code(chash, child_program.code)

        # -- Adjustment 2: Island ID (best-effort) --------------------------
        # database.add() has already set metadata["island"] to the final
        # placement island by the time this hook fires.  May be None for
        # the initial seed program or manual runs.
        island_id = (
            child_program.metadata.get("island")
            if child_program.metadata
            else None
        )

        program_ref = ProgramRef(
            program_id=child_program.id,
            run_id=run_id,
            island_id=island_id,
            iteration=iteration,
            parent_program_id=child_program.parent_id,
            code_hash=chash,
            code_path=code_rel_path,
        )

        eval_summary = EvaluationSummary(
            fitness=fitness,
            metrics=dict(child_program.metrics) if child_program.metrics else {},
            status="ok",
        )

        # Safely convert artifacts to string-serialisable form
        safe_artifacts: dict[str, Any] = {}
        if result.artifacts:
            for k, v in result.artifacts.items():
                if isinstance(v, bytes):
                    safe_artifacts[k] = f"<binary {len(v)} bytes>"
                elif isinstance(v, str):
                    safe_artifacts[k] = v[:10_000]
                else:
                    safe_artifacts[k] = str(v)[:10_000]

        event = ProgramEvaluatedEvent(
            schema_version="gca.v0.program_evaluated.v1",
            ts=time.time(),
            program=program_ref,
            evaluation=eval_summary,
            artifacts=safe_artifacts,
            config_hash=config_hash,
        )

        # -- Adjustment 1: Parent fitness lookup (may be None) ---------------
        # Look up parent program from the database to get its fitness.
        # Returns None when:
        #   - child is the initial seed (no parent_id)
        #   - parent has been evicted from the database
        #   - parent has empty metrics
        parent_fitness: float | None = None
        if child_program.parent_id:
            parent_prog = database.get(child_program.parent_id)
            if parent_prog and parent_prog.metrics:
                parent_fitness = get_fitness_score(parent_prog.metrics)

        await observer.on_program_evaluated(event, parent_fitness=parent_fitness)

    except Exception:
        logger.exception("GCA notify failed (non-fatal) for program %s", child_program.id)


def build_gca_stack_for_oe(
    config_dict: dict[str, Any] | None,
    *,
    llm_ensemble=None,
    output_dir: str = "",
) -> tuple[Any, str] | None:
    """Factory that creates a GCA stack from OpenEvolve's config.

    Parameters
    ----------
    config_dict : dict | None
        The ``gca:`` section from YAML config (or None if absent).
    llm_ensemble : LLMEnsemble, optional
        The OE LLM ensemble, used to create an async callable for the
        LLM-based extractor.
    output_dir : str
        OE output directory (used as default store_path base).

    Returns
    -------
    (GCAStack, run_id) or None if GCA is disabled.
    """
    import os
    from gca import GCAStack
    from gca.config import GCAConfig

    if config_dict is None or not config_dict.get("enabled", False):
        return None

    gca_cfg = GCAConfig.from_dict(config_dict)

    # Default store_path relative to OE output
    if gca_cfg.store_path == "./gca_store" and output_dir:
        gca_cfg = GCAConfig(
            enabled=gca_cfg.enabled,
            store_path=os.path.join(output_dir, "gca_store"),
            run_id=gca_cfg.run_id,
            extractor=gca_cfg.extractor,
            policy=gca_cfg.policy,
            async_config=gca_cfg.async_config,
            autosave=gca_cfg.autosave,
        )

    # Build LLM callable from OE ensemble if available
    llm_callable = None
    if llm_ensemble is not None:

        async def _llm_call(system_msg: str, user_msg: str) -> str:
            messages = [{"role": "user", "content": user_msg}]
            return await llm_ensemble.generate_with_context(system_msg, messages)

        llm_callable = _llm_call

    stack = GCAStack.from_config(gca_cfg, llm_callable=llm_callable)
    return stack, stack.config.run_id
