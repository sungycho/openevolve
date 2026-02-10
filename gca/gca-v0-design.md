Below is a **comprehensive, concrete GCA v0 plan** (same granularity as the Copilot draft), but **corrected** with the feedbacks:

* Strategies are extracted **from (solution code + eval outcomes)**, not prompts
* Strategy provenance is **"first_observed"**, not "owned by"
* Occurrences are **not embedded** inside Strategy (use a join/event log)
* No premature `was_improvement`; store raw comparison data instead
* Extract **0–N (up to 3)** strategies only when clearly present
* Observer is **post-evaluation** and can run **async / non-blocking**
* Everything is **append-only + versioned** so future extensions (regimes/credit/selection) don't require schema surgery

---

# GCA v0 (MVP) — Comprehensive Design & Execution Plan

## Layer 0 — MVP boundaries (lock this in README)

**GCA v0 delivers:**

1. A **Program Evaluation Event log** (code pointer + outcome + artifacts)
2. A **Strategy Catalog** (deduped, stable IDs, first_observed metadata)
3. A **Strategy Occurrence log** (program↔strategy linking + extraction metadata)
4. A **read-only query interface** + minimal CLI (optional but recommended)

**GCA v0 explicitly excludes:**

* feature regimes
* credit assignment / scoring policies
* strategy selection / prompt injection
* strategy mutation / optimization

> GCA v0 is a **semantic observer + persistent memory**, not an optimizer.

---

## Layer 1 — Data model (minimal, future-proof)

### 1.1 Program representation (what we observe)

We don't need to re-model all of OpenEvolve. We only need a stable minimal representation that future layers can build on.

```python
@dataclass(frozen=True)
class ProgramRef:
    program_id: str               # stable id from OE (or hash if missing)
    run_id: str
    island_id: int | None
    iteration: int | None
    parent_program_id: str | None # optional (if available)
    code_hash: str                # sha256(normalized_code)
    code_path: str                # content-addressed storage location
```

```python
@dataclass(frozen=True)
class EvaluationSummary:
    fitness: float
    metrics: dict[str, float]     # e.g., runtime, memory, violations
    status: str                   # "ok" | "error" | "timeout" | ...
    error_type: str | None        # optional
```

```python
@dataclass(frozen=True)
class ProgramEvaluatedEvent:
    schema_version: str           # e.g., "gca.v0.program_evaluated.v1"
    ts: float
    program: ProgramRef
    evaluation: EvaluationSummary
    artifacts: dict               # stderr, logs, profiler output (bounded size)
    config_hash: str              # hash of OE config for reproducibility
```

**Key v0 choices**

* **Code is stored separately** using content-addressing; events store only hash/path.
* Include `parent_program_id` if available but don't depend on it.

---

### 1.2 Strategy as a first-class entity (catalog record)

A Strategy is an abstract, reusable description with stable identity.

```python
@dataclass(frozen=True)
class Strategy:
    strategy_id: str              # sha256(canonical_strategy_text)
    description: str              # natural language, meso-granularity
    tags: list[str]               # empty now, future hooks
    extraction_method: str        # "solution_analysis_llm_v1" (or heuristic later)
    first_observed: dict          # provenance, not ownership:
                                 # {program_id, run_id, iteration, ts}
    created_at: float
```

**Notes**

* No `occurrences: List[...]` inside Strategy.
* `first_observed` is informational only.

---

### 1.3 Strategy occurrence (join/event log)

Occurrences are empirical grounding: "Strategy S was observed in Program P".

```python
@dataclass(frozen=True)
class StrategyOccurrence:
    schema_version: str           # "gca.v0.strategy_occurrence.v1"
    ts: float
    strategy_id: str
    program_id: str
    run_id: str
    island_id: int | None
    iteration: int | None

    # extraction metadata
    extractor_version: str        # e.g., "llm-solution-analysis@0.1.0"
    extraction_confidence: float  # 0..1
    evidence: dict                # minimal justification: snippets, rationale, etc.

    # neutral comparison data (NOT "improvement" yet)
    program_fitness: float
    parent_fitness: float | None
    comparison_context: str       # "parent" | "cell" | "none"
```

**Why this matters**

* You can derive any future "credit" definition later without breaking logs.

---

## Layer 2 — Storage & persistence (append-only truth)

### 2.1 Directory layout

Keep everything inspectable and modular:

```
gca/
  __init__.py
  observer.py
  extractor.py
  registry.py
  archive.py
  persistence.py
  query.py
  schemas.py
  cli.py (optional)

gca_store/
  programs/
    <code_hash>.py                # content-addressed storage (one per unique)
  logs/
    program_evaluated.jsonl       # append-only events
    strategy_occurrence.jsonl     # append-only join events
  catalogs/
    strategies.jsonl              # append-only but de-duped on load
  metadata.json                   # archive metadata + versions + run list
```

### 2.2 Truth model

* **Event logs are source-of-truth** (`program_evaluated.jsonl`, `strategy_occurrence.jsonl`)
* Strategy catalog is **materialized state** that can be rebuilt from logs if needed.

### 2.3 Versioning

Every record includes `schema_version`. The extractor includes `extractor_version`.

`metadata.json`:

```json
{
  "gca_version": "0.1.0",
  "created_at": 1739140000.0,
  "last_updated": 1739149999.0,
  "openevolve_commit": "abc123",
  "runs_included": ["run_001", "run_002"],
  "log_files": {
    "program_evaluated": "logs/program_evaluated.jsonl",
    "strategy_occurrence": "logs/strategy_occurrence.jsonl",
    "strategies": "catalogs/strategies.jsonl"
  }
}
```

---

## Layer 3 — Strategy extraction (solution + outcome → strategies)

### 3.1 Extractor interface

Single entrypoint: extract strategies from a evaluated program.

```python
class StrategyExtractor(Protocol):
    name: str
    version: str

    async def extract_from_solution(
        self,
        program_code: str,
        program_ref: ProgramRef,
        evaluation: EvaluationSummary,
        artifacts: dict
    ) -> list[ExtractedStrategy]:
        ...
```

Where:

```python
@dataclass(frozen=True)
class ExtractedStrategy:
    description: str
    confidence: float
    evidence: dict   # e.g., {"rationale": "...", "code_snippets": [...], "signals": {...}}
    tags: list[str]  # optional (empty ok)
```

### 3.2 LLM-based extractor (MVP)

**LLM role:** analysis/understanding, not generation.

**Prompt contract (important):**

* output: **0 to 3** lines
* each line: one strategy
* include optional JSON evidence (or return separate structured JSON)

**Recommended output format for robustness: JSON**
Even in MVP, ask the model for JSON to reduce parsing pain.

Example prompt (conceptual):

* Provide code (or truncated + key functions)
* Provide evaluation summary + artifacts (truncated)
* Ask for *up to 3* strategies if clearly present
* For each: `description`, `confidence`, `evidence.rationale`, `evidence.snippets`

```python
class LLMSolutionStrategyExtractor:
    name = "llm_solution_analysis"
    version = "0.1.0"

    async def extract_from_solution(...)-> list[ExtractedStrategy]:
        # 1) Build analysis input with size limits (see below)
        # 2) Call LLM
        # 3) Parse JSON
        # 4) Return 0..3 strategies
```

### 3.3 Input size management (must-have guardrail)

You *will* hit token limits. Keep this deterministic:

* Always include:

  * top-level function signatures
  * critical code blocks (e.g., main algorithm function)
  * summary of evaluation metrics
* Truncate:

  * long helper code
  * full logs/stderr (keep last N lines)

Add a simple "code summarizer" fallback (non-LLM) that extracts:

* imports
* function defs
* key loops / data structures (regex-ish)

This is still minimal but prevents brittleness.

---

## Layer 4 — Extraction trigger policy (cost control, signal focus)

### 4.1 Trigger interface

```python
class ExtractionPolicy(Protocol):
    def should_extract(self, event: ProgramEvaluatedEvent) -> bool: ...
```

### 4.2 MVP policy (simple, defensible)

Use an OR of a few criteria. Keep it configurable:

* **Top-k per iteration** (within a run) or "best in cell" if available
* **Delta vs parent** if parent fitness exists
* **Early exploration window** (first N iterations)
* **Error cases** optional (only if you want failure strategies later)

Example policy:

```python
return (
    is_top_k_in_recent_window(event) or
    (event.program.parent_program_id is not None and
     event.evaluation.fitness - parent_fitness > threshold) or
    (event.program.iteration is not None and event.program.iteration <= early_N)
)
```

**Important:** The policy should be **pure** and **explainable**. Log *why it triggered*.

Add to occurrence evidence:

```json
"trigger_reason": "improvement_delta"
```

---

## Layer 5 — Registry & deduplication (exact now, upgrade later)

### 5.1 Canonicalization

Define canonical form for strategy text (stable ID generation):

* strip whitespace
* lowercase
* remove trailing punctuation
* collapse repeated spaces

```python
def canonicalize_strategy(text: str) -> str: ...
def strategy_id(text: str) -> str: return sha256(canonicalize_strategy(text))
```

### 5.2 StrategyRegistry

```python
class StrategyRegistry:
    def __init__(self, persistence: ...):
        self._strategies: dict[str, Strategy] = load_existing_catalog()

    def register(self, extracted: ExtractedStrategy, first_observed: dict) -> Strategy:
        sid = strategy_id(extracted.description)
        if sid not in self._strategies:
            self._strategies[sid] = Strategy(
                strategy_id=sid,
                description=extracted.description.strip(),
                tags=extracted.tags or [],
                extraction_method="solution_analysis_llm_v1",
                first_observed=first_observed,
                created_at=time.time(),
            )
            append_to_strategies_jsonl(self._strategies[sid])
        return self._strategies[sid]
```

**MVP dedupe = exact by canonical text.**
Future: add embedding-based merge behind a flag.

---

## Layer 6 — Archive core (record + persist + query)

### 6.1 GlobalCollectiveArchive (GCA)

```python
class GlobalCollectiveArchive:
    def __init__(self, store_path: str, registry: StrategyRegistry, persistence: Persistence):
        ...

    def record_program_event(self, event: ProgramEvaluatedEvent) -> None:
        persistence.append_program_event(event)

    def record_occurrence(self, occ: StrategyOccurrence) -> None:
        persistence.append_occurrence_event(occ)

    def materialize_indices(self) -> None:
        # optional: build in-memory indices for query
        # e.g., occurrences_by_strategy, strategies_by_run
        ...
```

### 6.2 Query surface (read-only)

Minimal but enough to demonstrate "queryable memory":

```python
class ArchiveQuery:
    def get_strategy(self, sid: str) -> Strategy | None: ...
    def list_strategies(self, limit: int = 50) -> list[Strategy]: ...
    def list_occurrences(self, sid: str, limit: int = 200) -> list[StrategyOccurrence]: ...
    def strategies_in_run(self, run_id: str) -> list[Strategy]: ...
    def strategy_usage_count(self, sid: str) -> int: ...
```

No scoring, no regimes.

---

## Layer 7 — Integration with OpenEvolve (minimal, explicit, opt-in)

### 7.1 Integration pattern: Observer / Sink

This should be a single integration touchpoint.

```python
class EvolutionObserver:
    def __init__(self, gca: GlobalCollectiveArchive, extractor: StrategyExtractor, policy: ExtractionPolicy):
        ...

    async def on_program_evaluated(self, event: ProgramEvaluatedEvent):
        # 0) Persist raw program event (always)
        gca.record_program_event(event)

        # 1) Decide whether to extract
        if not policy.should_extract(event):
            return

        # 2) Load code (from program_store via hash/path)
        code = load_code(event.program.code_path)

        # 3) Extract strategies (0..3)
        extracted_list = await extractor.extract_from_solution(
            program_code=code,
            program_ref=event.program,
            evaluation=event.evaluation,
            artifacts=event.artifacts,
        )

        # 4) Register + record occurrences
        for extracted in extracted_list:
            strat = registry.register(
                extracted,
                first_observed={
                    "program_id": event.program.program_id,
                    "run_id": event.program.run_id,
                    "iteration": event.program.iteration,
                    "ts": event.ts
                }
            )
            occ = StrategyOccurrence(
                schema_version="gca.v0.strategy_occurrence.v1",
                ts=time.time(),
                strategy_id=strat.strategy_id,
                program_id=event.program.program_id,
                run_id=event.program.run_id,
                island_id=event.program.island_id,
                iteration=event.program.iteration,
                extractor_version=f"{extractor.name}@{extractor.version}",
                extraction_confidence=extracted.confidence,
                evidence=extracted.evidence,
                program_fitness=event.evaluation.fitness,
                parent_fitness=None,              # fill if available
                comparison_context="none",        # or "parent" if parent_fitness known
            )
            gca.record_occurrence(occ)
```

### 7.2 "Minimal invasiveness" requirement

* One place where OE produces a `ProgramEvaluatedEvent`
* If GCA disabled → no behavior change

### 7.3 Async design (recommended)

Extraction should not block the evolution loop:

* Use an async queue + worker, or
* fire-and-forget tasks with backpressure

MVP approach:

* A bounded `asyncio.Queue`
* One worker consumes events and runs extraction

This keeps v0 clean and prevents evaluation slowdown.

---

## Layer 8 — Config (opt-in, future-proof)

Example YAML:

```yaml
gca:
  enabled: true
  store_path: "./gca_store"
  extractor:
    type: "llm_solution_analysis"
    model: "..."
    max_strategies: 3
    max_code_chars: 12000
    max_artifact_chars: 4000
  policy:
    type: "mvp"
    early_iterations: 25
    improvement_threshold: 0.05
    top_k_window: 5
  async:
    enabled: true
    queue_size: 128
    workers: 1
  autosave:
    interval_events: 50
```

---

# Concrete v0 deliverables checklist

## A) Files/classes you will implement

* `schemas.py`: dataclasses + schema versions
* `persistence.py`: JSONL appenders + loaders + metadata writer
* `registry.py`: canonicalization + dedupe + strategy catalog writer
* `extractor.py`: LLM-based solution analyzer (JSON output)
* `policy.py`: extraction trigger policy
* `observer.py`: post-eval hook + async queue worker
* `archive.py`: GCA orchestrator
* `query.py`: read-only query helpers
* `cli.py` (optional): quick inspection commands

## B) Outputs you will produce to "show" MVP works

* `strategies.jsonl`: a catalog of deduped strategy objects
* `strategy_occurrence.jsonl`: occurrences linked to programs
* `program_evaluated.jsonl`: raw audit trail

## C) Demo queries you should be able to run

* list top strategies (by usage count)
* show all programs that exhibit a strategy
* show timeline of strategies in a run (counts per iteration)

---

# The 8 "locked" design choices for GCA v0

These are the ones you should treat as non-negotiable in implementation:

1. **Extraction source:** solution code + eval outcome (post-eval)
2. **Observer-only integration:** no changes to selection/mutation logic
3. **Append-only event logs** as truth
4. **Content-addressed code storage** referenced by hash/path
5. **Strategy dedupe:** canonical text exact-match in v0
6. **Strategy provenance:** `first_observed` informational only
7. **Occurrences are separate records** (no embedding inside Strategy)
8. **No improvement policy baked in:** store raw comparison fields only

---

# Final sanity: does this match your intention?

* MVP remains minimal and modular
* It produces a **real artifact** (queryable memory) from actual evolution outcomes
* It keeps future extensions clean (regimes/credit/selection won't require rewrites)
