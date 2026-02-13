"""
GCA v0 — Global Collective Archive

A semantic observer + persistent memory for OpenEvolve evolution runs.

GCA v0 delivers:
  1. A **Program Evaluation Event log** (code pointer + outcome + artifacts)
  2. A **Strategy Catalog** (deduped, stable IDs, first_observed metadata)
  3. A **Strategy Occurrence log** (program<->strategy linking + extraction metadata)
  4. A **read-only query interface** + minimal CLI

GCA v0 explicitly excludes:
  * feature regimes
  * credit assignment / scoring policies
  * strategy selection / prompt injection
  * strategy mutation / optimization

Quick start
-----------
::

    from gca import GCAStack

    stack = GCAStack.from_config(gca_config_dict)
    await stack.start()

    # In the evolution loop:
    await stack.observer.on_program_evaluated(event)

    # Query:
    for strat, count in stack.archive.query.top_strategies(10):
        print(f"{count}  {strat.description}")

    await stack.stop()
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from gca.archive import ArchiveQuery, GlobalCollectiveArchive
from gca.config import GCAConfig
from gca.extractor import (
    HeuristicStrategyExtractor,
    LLMSolutionStrategyExtractor,
    StrategyExtractor,
)
from gca.observer import EvolutionObserver
from gca.persistence import Persistence
from gca.policy import (
    AlwaysExtractPolicy,
    ExtractionPolicy,
    MVPExtractionPolicy,
    NeverExtractPolicy,
)
from gca.registry import StrategyRegistry
from gca.schemas import (
    EvaluationSummary,
    ExtractedStrategy,
    ProgramEvaluatedEvent,
    ProgramRef,
    Strategy,
    StrategyOccurrence,
)

logger = logging.getLogger(__name__)

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# Convenience stack: wires everything together
# ---------------------------------------------------------------------------

@dataclass
class GCAStack:
    """Pre-wired GCA component stack.

    Use ``GCAStack.from_config()`` to build from a ``GCAConfig`` or raw
    dict, or construct manually for full control.
    """

    config: GCAConfig
    persistence: Persistence
    registry: StrategyRegistry
    archive: GlobalCollectiveArchive
    extractor: StrategyExtractor
    policy: ExtractionPolicy
    observer: EvolutionObserver

    async def start(self) -> None:
        """Start background workers (call once at run start)."""
        await self.observer.start()

    async def stop(self) -> None:
        """Graceful shutdown."""
        await self.observer.stop()

    # -- factory -----------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: GCAConfig | dict[str, Any],
        *,
        llm_callable=None,
    ) -> GCAStack:
        """Build a complete GCA stack from configuration.

        Parameters
        ----------
        config : GCAConfig | dict
            GCA configuration (or raw dict from YAML).
        llm_callable : async (system, user) -> str, optional
            Required only when ``extractor.type == "llm_solution_analysis"``.
        """
        if isinstance(config, dict):
            config = GCAConfig.from_dict(config)

        # Assign run_id if not provided
        if not config.run_id:
            config = GCAConfig(
                enabled=config.enabled,
                store_path=config.store_path,
                run_id=f"run_{uuid.uuid4().hex[:8]}",
                extractor=config.extractor,
                policy=config.policy,
                async_config=config.async_config,
                autosave=config.autosave,
                feedback=config.feedback,
            )

        # Persistence
        persistence = Persistence(config.store_path)
        persistence.add_run(config.run_id)

        # Registry
        registry = StrategyRegistry(persistence)

        # Archive
        archive = GlobalCollectiveArchive(
            store_path=config.store_path,
            registry=registry,
            persistence=persistence,
        )

        # Extractor
        extractor: StrategyExtractor
        if config.extractor.type == "heuristic":
            extractor = HeuristicStrategyExtractor()
        elif config.extractor.type == "llm_solution_analysis":
            if llm_callable is None:
                logger.warning(
                    "GCA: LLM extractor requested but no llm_callable provided; "
                    "falling back to heuristic extractor."
                )
                extractor = HeuristicStrategyExtractor()
            else:
                extractor = LLMSolutionStrategyExtractor(
                    llm_callable,
                    max_strategies=config.extractor.max_strategies,
                    max_code_chars=config.extractor.max_code_chars,
                    max_artifact_chars=config.extractor.max_artifact_chars,
                )
        else:
            raise ValueError(f"Unknown extractor type: {config.extractor.type}")

        # Policy
        policy: ExtractionPolicy
        if config.policy.type == "always":
            policy = AlwaysExtractPolicy()
        elif config.policy.type == "never":
            policy = NeverExtractPolicy()
        elif config.policy.type == "mvp":
            policy = MVPExtractionPolicy(
                early_iterations=config.policy.early_iterations,
                improvement_threshold=config.policy.improvement_threshold,
                top_k_window=config.policy.top_k_window,
                recent_window_size=config.policy.recent_window_size,
            )
        else:
            raise ValueError(f"Unknown policy type: {config.policy.type}")

        # Observer
        observer = EvolutionObserver(
            gca=archive,
            registry=registry,
            persistence=persistence,
            extractor=extractor,
            policy=policy,
            async_enabled=config.async_config.enabled,
            queue_size=config.async_config.queue_size,
        )

        return cls(
            config=config,
            persistence=persistence,
            registry=registry,
            archive=archive,
            extractor=extractor,
            policy=policy,
            observer=observer,
        )


__all__ = [
    # Stack
    "GCAStack",
    # Config
    "GCAConfig",
    # Schemas
    "ProgramRef",
    "EvaluationSummary",
    "ProgramEvaluatedEvent",
    "Strategy",
    "StrategyOccurrence",
    "ExtractedStrategy",
    # Components
    "GlobalCollectiveArchive",
    "ArchiveQuery",
    "EvolutionObserver",
    "StrategyRegistry",
    "Persistence",
    # Extractors
    "StrategyExtractor",
    "LLMSolutionStrategyExtractor",
    "HeuristicStrategyExtractor",
    # Policies
    "ExtractionPolicy",
    "MVPExtractionPolicy",
    "AlwaysExtractPolicy",
    "NeverExtractPolicy",
]
