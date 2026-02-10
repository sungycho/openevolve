"""
GCA v0 — Extraction trigger policy

Layer 4 of the GCA plan.

Controls *which* evaluated programs undergo (potentially expensive) LLM
strategy extraction.  The policy is pure and explainable — it logs **why**
it triggered so the decision can be audited later.

MVP policy is an OR of:
  * Top-k in recent iteration window
  * Significant fitness delta vs parent
  * Early exploration window (first N iterations)
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from gca.schemas import ProgramEvaluatedEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol (Layer 4.1)
# ---------------------------------------------------------------------------

@runtime_checkable
class ExtractionPolicy(Protocol):
    """Decides whether a ``ProgramEvaluatedEvent`` warrants extraction."""

    def should_extract(self, event: ProgramEvaluatedEvent) -> tuple[bool, str]:
        """Return (should_extract, trigger_reason)."""
        ...


# ---------------------------------------------------------------------------
# MVP policy (Layer 4.2)
# ---------------------------------------------------------------------------

class MVPExtractionPolicy:
    """Simple, defensible trigger policy for v0.

    Parameters
    ----------
    early_iterations : int
        Extract from every program in the first *N* iterations.
    improvement_threshold : float
        Minimum fitness delta vs parent to trigger extraction.
    top_k_window : int
        Keep the top-*k* fitness values seen in the recent window;
        extract if the new program would enter that top-k.
    recent_window_size : int
        Number of recent events to consider for the top-k test.
    """

    def __init__(
        self,
        *,
        early_iterations: int = 25,
        improvement_threshold: float = 0.05,
        top_k_window: int = 5,
        recent_window_size: int = 50,
    ) -> None:
        self._early_iterations = early_iterations
        self._improvement_threshold = improvement_threshold
        self._top_k_window = top_k_window
        self._recent_window_size = recent_window_size

        # Ring buffer of recent fitness values
        self._recent_fitness: list[float] = []

    def should_extract(self, event: ProgramEvaluatedEvent) -> tuple[bool, str]:
        fitness = event.evaluation.fitness

        # --- criterion 1: early exploration window ---
        iteration = event.program.iteration
        if iteration is not None and iteration <= self._early_iterations:
            # Still record fitness so the ring buffer fills during the
            # early window; top-k will work correctly once it ends.
            self._record_fitness(fitness)
            return True, "early_exploration"

        # --- criterion 2: top-k in recent window ---
        is_top_k = self._check_top_k(fitness)

        # Always record fitness BEFORE returning, so the ring buffer
        # reflects every program we've seen — not just rejected ones.
        self._record_fitness(fitness)

        if is_top_k:
            return True, "top_k_recent"

        # --- criterion 3: improvement delta vs parent ---
        # Not implemented in v0.  The improvement_threshold config is
        # accepted but not evaluated; parent_fitness is not available on
        # the event.  Planned for v1.

        return False, "none"

    # -- helpers -----------------------------------------------------------

    def _check_top_k(self, fitness: float) -> bool:
        """Would *fitness* enter the top-k of the recent window?"""
        if len(self._recent_fitness) < self._top_k_window:
            # Not enough data yet — extract to bootstrap
            return True
        # Sort descending; if fitness >= k-th best, it's top-k
        sorted_recent = sorted(self._recent_fitness, reverse=True)
        threshold = sorted_recent[min(self._top_k_window - 1, len(sorted_recent) - 1)]
        return fitness >= threshold

    def _record_fitness(self, fitness: float) -> None:
        self._recent_fitness.append(fitness)
        if len(self._recent_fitness) > self._recent_window_size:
            self._recent_fitness.pop(0)


class AlwaysExtractPolicy:
    """Trivial policy that extracts from every program.

    Useful for small runs or debugging.
    """

    def should_extract(self, event: ProgramEvaluatedEvent) -> tuple[bool, str]:
        return True, "always"


class NeverExtractPolicy:
    """Trivial policy that never triggers extraction.

    Useful when GCA is enabled only for event logging, not strategy
    extraction.
    """

    def should_extract(self, event: ProgramEvaluatedEvent) -> tuple[bool, str]:
        return False, "disabled"
