# GCA v1 — Strategy Feedback Loop

## Overview

GCA v0 is observer-only: it extracts strategies and stores them, but never feeds them back to the LLM. GCA v1 completes the feedback loop by querying top strategies (ranked by best associated program fitness), serializing them into the database snapshot, and injecting them into the LLM prompt as a "Collective Strategy Insights" section.

## Design Summary

Before each iteration, the main process queries GCA for the top-k strategies (ranked by best associated program fitness), serializes them into the database snapshot, and the worker injects them into the LLM prompt as a new "Collective Strategy Insights" section that serves as meta-reasoning guidance.

## Data Flow

```
Main Process                          Worker Process
─────────────                         ──────────────
gca_stack.archive.query
  .top_strategies_by_fitness(3)
         │
         ▼
snapshot["gca_strategies"] = [
  {"description": "...",
   "best_fitness": 0.85,
   "count": 5}, ...
]
         │
         ▼ (pickle via ProcessPool)
                                      db_snapshot["gca_strategies"]
                                               │
                                               ▼
                                      build_prompt(gca_strategies=...)
                                               │
                                               ▼
                                      "## Collective Strategy Insights
                                       Here are strategies that previously
                                       improved fitness..."
```

## New Components

### `ArchiveQuery.top_strategies_by_fitness(limit)`

- Loads all occurrence events
- Groups by strategy_id, computes max `program_fitness` per strategy
- Returns top N strategies sorted by best fitness (descending)
- Returns `list[tuple[Strategy, float, int]]` — (strategy, best_fitness, occurrence_count)

### `GCAFeedbackConfig`

```python
@dataclass
class GCAFeedbackConfig:
    enabled: bool = False
    top_k: int = 3
```

Added as `feedback` field on `GCAConfig`.

### Prompt Template: `gca_strategies_section.txt`

```
## Collective Strategy Insights

Here are strategies that previously improved fitness. Reflect on whether adapting or combining them could improve the current program.

{gca_strategies}
```

### Strategy Rendering Format

Each strategy is rendered as:
```
- **<description>** (best fitness: X.XXXX, seen in N programs)
```

## Configuration

```yaml
gca:
  enabled: true
  store_path: "./gca_store"
  feedback:
    enabled: true
    top_k: 3
  extractor:
    type: "llm_solution_analysis"
    max_strategies: 3
  policy:
    type: "mvp"
```

## Backward Compatibility

- `feedback` config section defaults to `enabled: false`
- Runs without `feedback:` config work identically to v0
- `gca_strategies` key is only added to snapshot when feedback is enabled and strategies exist
- `gca_strategies_section` placeholder in evolution_history.txt renders as empty string when no strategies are provided

## Locked v0 Design Choices (Preserved)

All v0 design choices remain intact. GCA v1 only adds a read-only feedback path — no changes to extraction, storage, or the observer pipeline.
