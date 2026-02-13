# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenEvolve is an open-source implementation of Google DeepMind's AlphaEvolve system - an evolutionary coding agent that uses LLMs to optimize code through iterative evolution. The framework can evolve code in multiple languages (Python, R, Rust, etc.) for tasks like scientific computing, optimization, and algorithm discovery.

## Essential Commands

### Development Setup
```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Or use Makefile
make install
```

### Running Tests
```bash
# Run all tests
python -m unittest discover tests

# Or use Makefile
make test
```

### Code Formatting
```bash
# Format with Black
python -m black openevolve examples tests scripts

# Or use Makefile
make lint
```

### Running OpenEvolve
```bash
# Basic evolution run
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py --config path/to/config.yaml --iterations 1000

# Resume from checkpoint
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint_directory \
  --iterations 50
```

### Visualization
```bash
# View evolution tree
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/
```

## High-Level Architecture

### Core Components

1. **Controller (`openevolve/controller.py`)**: Main orchestrator that manages the evolution process using ProcessPoolExecutor for parallel iteration execution.

2. **Database (`openevolve/database.py`)**: Implements MAP-Elites algorithm with island-based evolution:
   - Programs mapped to multi-dimensional feature grid
   - Multiple isolated populations (islands) evolve independently
   - Periodic migration between islands prevents convergence
   - Tracks absolute best program separately

3. **Evaluator (`openevolve/evaluator.py`)**: Cascade evaluation pattern:
   - Stage 1: Quick validation
   - Stage 2: Basic performance testing  
   - Stage 3: Comprehensive evaluation
   - Programs must pass thresholds at each stage

4. **LLM Integration (`openevolve/llm/`)**: Ensemble approach with multiple models, configurable weights, and async generation with retry logic.

5. **Iteration (`openevolve/iteration.py`)**: Worker process that samples from islands, generates mutations via LLM, evaluates programs, and stores artifacts.

### Key Architectural Patterns

- **Island-Based Evolution**: Multiple populations evolve separately with periodic migration
- **MAP-Elites**: Maintains diversity by mapping programs to feature grid cells
- **Artifact System**: Side-channel for programs to return debugging data, stored as JSON or files
- **Process Worker Pattern**: Each iteration runs in fresh process with database snapshot
- **Double-Selection**: Programs for inspiration differ from those shown to LLM
- **Lazy Migration**: Islands migrate based on generation counts, not iterations

### Code Evolution Markers

Mark code sections to evolve using:
```python
# EVOLVE-BLOCK-START
# Code to evolve goes here
# EVOLVE-BLOCK-END
```

### Configuration

YAML-based configuration with hierarchical structure:
- LLM models and parameters
- Evolution strategies (diff-based vs full rewrites)
- Database and island settings
- Evaluation parameters

### Important Patterns

1. **Checkpoint/Resume**: Automatic saving of entire system state with seamless resume capability
2. **Parallel Evaluation**: Multiple programs evaluated concurrently via TaskPool
3. **Error Resilience**: Individual failures don't crash system - extensive retry logic and timeout protection
4. **Prompt Engineering**: Template-based system with context-aware building and evolution history

### Development Notes

- Python >=3.10 required
- Uses OpenAI-compatible APIs for LLM integration
- Tests use unittest framework
- Black for code formatting
- Artifacts threshold: Small (<10KB) stored in DB, large saved to disk
- Process workers load database snapshots for true parallelism

---

## Global Collective Archive (GCA) — `gca/`

### What GCA Is

GCA is a **strategy-level memory system** that sits alongside OpenEvolve's evolution loop as a pure observer. It extracts, deduplicates, and persists reusable algorithmic strategies from evaluated programs — converting implicit, ephemeral knowledge into explicit, queryable, cross-run memory.

**Key principle**: GCA v0 is an **observer + persistent memory**, not an optimizer. It does not modify selection, mutation, or any evolution logic.

### Core Concepts

- **Strategy**: A reusable, meso-level technique (e.g., "uses memoization", "applies binary search for efficient lookup"). Has a stable content-addressed ID derived from canonicalized description text.
- **Strategy Occurrence**: An empirical record linking a strategy to a specific program evaluation. Stores raw fitness data (no premature scoring).
- **Program Evaluated Event**: Raw audit trail of every evaluated program (code hash, fitness, metrics, artifacts).

### Architecture Overview

```
OpenEvolve (unchanged)
    │ evaluation events (post-eval)
    ▼
EvolutionObserver          ← single integration touchpoint
    │
    ├─ ExtractionPolicy    ← decides which programs to analyze
    │
    ├─ StrategyExtractor   ← LLM or heuristic analysis
    │
    ├─ StrategyRegistry    ← dedup + catalog
    │
    └─ GlobalCollectiveArchive + Persistence
         └─ ArchiveQuery   ← read-only query surface
```

GCA sits **outside** the evolution loop. Extraction runs asynchronously via a bounded `asyncio.Queue` + background worker, so it never blocks evolution.

### File Guide

| File | Layer | Role |
|------|-------|------|
| `gca/__init__.py` | — | Public exports + `GCAStack` factory (wires all components) |
| `gca/__main__.py` | — | `python -m gca` CLI entry point |
| `gca/schemas.py` | L1 | Frozen dataclasses: `ProgramRef`, `EvaluationSummary`, `ProgramEvaluatedEvent`, `Strategy`, `StrategyOccurrence`, `ExtractedStrategy` + helpers (`code_hash`, `canonicalize_strategy`, `strategy_id_from_text`) |
| `gca/config.py` | L8 | Typed config schema: `GCAConfig`, `GCAExtractorConfig`, `GCAPolicyConfig`, `GCAAsyncConfig`, `GCAAutosaveConfig` with `from_dict()` YAML parsing |
| `gca/persistence.py` | L2 | Append-only JSONL storage + content-addressed code store (`programs/<sha256>.py`) |
| `gca/extractor.py` | L3 | `StrategyExtractor` protocol + `LLMSolutionStrategyExtractor` (JSON output, truncation guardrails) + `HeuristicStrategyExtractor` (regex/pattern fallback) |
| `gca/policy.py` | L4 | `ExtractionPolicy` protocol + `MVPExtractionPolicy` (early window OR top-k in recent window), `AlwaysExtractPolicy`, `NeverExtractPolicy` |
| `gca/registry.py` | L5 | `StrategyRegistry` — in-memory catalog backed by JSONL, exact-match dedup via canonical text hash |
| `gca/archive.py` | L6 | `GlobalCollectiveArchive` (write path) + `ArchiveQuery` (read-only: `top_strategies`, `timeline`, `programs_with_strategy`, etc.) |
| `gca/query.py` | — | `load_query(store_path)` one-liner for ad-hoc inspection from REPL/notebook |
| `gca/observer.py` | L7 | `EvolutionObserver` — post-eval hook, async queue + worker, calls extractor → registry → archive |
| `gca/openevolve_integration.py` | L7 | `notify_gca()` adapter: translates OE `Program`/`SerializableResult`/`ProgramDatabase` → GCA `ProgramEvaluatedEvent`. Also `build_gca_stack_for_oe()` factory |
| `gca/cli.py` | — | Human-facing CLI: `top-strategies`, `programs-for-strategy`, `timeline`, `info`, `strategies` |

### On-Disk Store Layout

```
gca_store/
  programs/
    <sha256>.py                  # content-addressed source files
  logs/
    program_evaluated.jsonl      # append-only event log (source of truth)
    strategy_occurrence.jsonl    # append-only join events (source of truth)
  catalogs/
    strategies.jsonl             # append-only, deduped on load (materialized state)
  metadata.json                  # archive metadata, version, run list
```

Event logs are the **source of truth**. The strategy catalog is materialized state rebuildable from logs.

### Integration With OpenEvolve

Single touchpoint in the evolution loop (after program added to DB):

```python
if self.gca_observer is not None:
    from gca.openevolve_integration import notify_gca
    await notify_gca(
        self.gca_observer, child_program, result, self.database,
        run_id=self._gca_run_id, iteration=completed_iteration,
    )
```

`notify_gca()` resolves context that requires DB access:
1. **parent_fitness** — looked up via `database.get(parent_id).metrics` (None if evicted/missing)
2. **island_id** — read from `child_program.metadata["island"]` (None for seed/manual runs)
3. **code_path** — copies code into GCA's own content-addressed store (never reads OE's internal files)

All exceptions are caught — GCA never disrupts the evolution loop.

### Configuration

Added as a `gca:` section in the OpenEvolve YAML config:

```yaml
gca:
  enabled: true
  store_path: "./gca_store"
  extractor:
    type: "llm_solution_analysis"   # or "heuristic"
    max_strategies: 3
    max_code_chars: 12000
    max_artifact_chars: 4000
  policy:
    type: "mvp"                     # or "always" / "never"
    early_iterations: 25
    improvement_threshold: 0.05
    top_k_window: 5
    recent_window_size: 50
  async:
    enabled: true
    queue_size: 128
    workers: 1
  autosave:
    interval_events: 50
```

### Quick Start (Programmatic)

```python
from gca import GCAStack

stack = GCAStack.from_config(gca_config_dict, llm_callable=my_async_llm_fn)
await stack.start()

# In the evolution loop:
await stack.observer.on_program_evaluated(event)

# Query:
for strat, count in stack.archive.query.top_strategies(10):
    print(f"{count}  {strat.description}")

await stack.stop()
```

### CLI Usage

```bash
python -m gca --store ./gca_store top-strategies
python -m gca --store ./gca_store programs-for-strategy <strategy_id>
python -m gca --store ./gca_store timeline --run run_001
python -m gca --store ./gca_store info
python -m gca --store ./gca_store strategies
```

### Locked Design Choices (v0)

These are non-negotiable in the current implementation:

1. **Extraction source**: solution code + eval outcome (post-eval only)
2. **Observer-only integration**: no changes to selection/mutation logic
3. **Append-only event logs** as source of truth
4. **Content-addressed code storage** referenced by sha256 hash
5. **Strategy dedup**: canonical text exact-match (future: embedding-based merge)
6. **Strategy provenance**: `first_observed` is informational only
7. **Occurrences are separate records** (not embedded inside Strategy)
8. **No improvement policy baked in**: stores raw comparison fields only

### What GCA v0 Explicitly Excludes

- Feature regimes / regime-conditioned statistics
- Credit assignment / scoring policies
- Strategy selection / prompt injection
- Strategy mutation / optimization
- Closed-loop integration (GCA influencing evolution decisions)

These are planned future extensions built on top of the persistent strategy memory.