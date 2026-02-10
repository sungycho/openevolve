"""
GCA v0 — Data model / schemas

Layer 1 of the GCA plan.  All types are frozen dataclasses so they are
immutable once created, enforcing the append-only truth model.

Design choices locked for v0
-----------------------------
* Extraction source: solution code + eval outcome (post-eval)
* Strategy dedupe: canonical text exact-match
* Strategy provenance: `first_observed` informational only
* Occurrences are *separate* records (not embedded inside Strategy)
* No improvement policy baked in — store raw comparison fields only

Integration assumptions with OpenEvolve (§1.1 of design doc)
--------------------------------------------------------------
1. Programs are stored in ``database.programs[program_id]`` after evaluation.
2. **Parent fitness** is resolved by ``notify_gca()`` which looks up
   ``database.get(parent_id).metrics``.  It may be ``None`` when the parent
   has been evicted, never existed, or is the initial seed program.
3. **Island assignment** is read from ``program.metadata["island"]`` which is
   set by ``database.add()`` *before* the GCA hook fires.  For edge cases
   (manual runs, initial program), ``island_id`` may be ``None``.
4. Artifacts come from ``EvaluationResult.artifacts`` via
   ``evaluator.get_pending_artifacts()``.
5. **Code storage**: GCA keeps its own content-addressed copy under
   ``gca_store/programs/<sha256>.py``.  The integration layer copies code
   into this store at notification time; GCA never reads from OE's internal
   ``db_path/programs/<id>.json`` files.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Layer 1.1 — Program representation (what we observe)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProgramRef:
    """Minimal, stable reference to an evaluated program.

    We intentionally do *not* re-model all of OpenEvolve's ``Program``.
    Only the fields future GCA layers will need are kept here.

    Resolution notes (§7.2 of design doc):

    * ``island_id``: Resolved from ``program.metadata["island"]`` **after**
      ``database.add()`` sets the final placement island.  May be ``None``
      for the initial seed, manual runs, or if island logic hasn't kicked in.
      Best-effort — do not rely on it being always present.
    * ``parent_program_id``: Taken from ``program.parent_id``; ``None`` for
      the initial seed program.
    * ``code_hash`` / ``code_path``: Content-addressed; ``notify_gca()``
      copies the code into GCA's own store and fills both fields.
    """

    program_id: str
    run_id: str
    island_id: int | None = None          # best-effort; None for seed / manual runs
    iteration: int | None = None
    parent_program_id: str | None = None  # None for seed program
    code_hash: str = ""                   # sha256(normalized_code)
    code_path: str = ""                   # GCA store-relative path, e.g. "programs/<hash>.py"


@dataclass(frozen=True)
class EvaluationSummary:
    """Compact summary of how a program performed."""

    fitness: float
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "ok"           # "ok" | "error" | "timeout" | ...
    error_type: str | None = None


@dataclass(frozen=True)
class ProgramEvaluatedEvent:
    """Immutable event emitted every time OpenEvolve finishes evaluating a
    program.  This is the primary input to the GCA observer pipeline.
    """

    schema_version: str = "gca.v0.program_evaluated.v1"
    ts: float = field(default_factory=time.time)
    program: ProgramRef = field(default_factory=ProgramRef)
    evaluation: EvaluationSummary = field(default_factory=EvaluationSummary)
    artifacts: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""        # hash of OE config for reproducibility


# ---------------------------------------------------------------------------
# Layer 1.2 — Strategy as a first-class entity (catalog record)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Strategy:
    """An abstract, reusable strategy with stable identity.

    * No ``occurrences`` list inside Strategy (v0 rule #7).
    * ``first_observed`` is informational-only provenance (v0 rule #6).
    """

    strategy_id: str
    description: str
    tags: list[str] = field(default_factory=list)
    extraction_method: str = "solution_analysis_llm_v1"
    first_observed: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Layer 1.3 — Strategy occurrence (join / event log)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyOccurrence:
    """Records that *Strategy S was observed in Program P*.

    Stores raw comparison data (fitness, parent_fitness) so future layers
    can derive any credit policy without schema surgery (v0 rule #8).

    ``parent_fitness`` resolution (Adjustment 1):
        Resolved by ``notify_gca()`` via ``database.get(parent_id).metrics``.
        Will be ``None`` when:
        - The program is the initial seed (no parent).
        - The parent has been evicted from the database.
        - Parent metrics are empty or unavailable.
        When ``parent_fitness`` is ``None``, ``comparison_context`` is ``"none"``.
        When it is present, ``comparison_context`` is ``"parent"``.

    ``island_id`` (Adjustment 2):
        Inherited from the ``ProgramRef`` on the triggering event.
        May be ``None`` for seed programs or manual runs.
    """

    schema_version: str = "gca.v0.strategy_occurrence.v1"
    ts: float = field(default_factory=time.time)

    strategy_id: str = ""
    program_id: str = ""
    run_id: str = ""
    island_id: int | None = None      # best-effort; None for seed / manual runs
    iteration: int | None = None

    # Extraction metadata
    extractor_version: str = ""
    extraction_confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    # Neutral comparison data (NOT "improvement" yet — v0 rule #8)
    program_fitness: float = 0.0
    parent_fitness: float | None = None   # None if parent unavailable; see docstring
    comparison_context: str = "none"      # "parent" | "cell" | "none"


# ---------------------------------------------------------------------------
# Intermediate type returned by extractors before registry de-duplication
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractedStrategy:
    """Raw extraction result before registry dedup / ID assignment."""

    description: str
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def code_hash(code: str) -> str:
    """Content-addressing hash for program source code."""
    normalized = code.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonicalize_strategy(text: str) -> str:
    """Deterministic canonical form for strategy descriptions.

    Used to generate stable strategy IDs (exact-match dedup in v0).
    """
    text = text.strip().lower()
    # Remove trailing punctuation
    text = text.rstrip(".,;:!?")
    # Collapse whitespace
    import re
    text = re.sub(r"\s+", " ", text)
    return text


def strategy_id_from_text(text: str) -> str:
    """Derive a deterministic strategy ID from its description."""
    canonical = canonicalize_strategy(text)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
