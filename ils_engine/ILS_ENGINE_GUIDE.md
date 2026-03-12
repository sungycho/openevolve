# ILS Engine — Complete Guide

> **What is this?** The ILS Engine is a post-processing layer for OpenEvolve that automatically applies Iterated Local Search (ILS) to the best program(s) produced by evolutionary search. It is fully LLM-automated: given only the output directory of an OpenEvolve run and an evaluator script, the engine analyzes the program, designs search strategies, generates runnable optimization scripts, executes them in parallel, and reports the best result found.

---

## Table of Contents

1. [Motivation and Design Philosophy](#1-motivation-and-design-philosophy)
2. [End-to-End Pipeline](#2-end-to-end-pipeline)
3. [Two-Phase Search Architecture](#3-two-phase-search-architecture)
4. [LLM Role at Each Step](#4-llm-role-at-each-step)
5. [Full Configuration Reference](#5-full-configuration-reference)
6. [Quick-Start Examples](#6-quick-start-examples)
7. [Output Directory Structure](#7-output-directory-structure)
8. [Design Decisions and Tradeoffs](#8-design-decisions-and-tradeoffs)

---

## 1. Motivation and Design Philosophy

OpenEvolve uses an LLM to evolve program *structure* — it searches the discrete space of program logic, algorithms, and code organization. This is powerful for discovering novel approaches, but it is not designed to fine-tune the continuous numerical parameters *within* a fixed program structure.

The ILS Engine fills exactly this gap. Once OpenEvolve has converged to a good program structure, the ILS Engine asks: **given this algorithm, what are its tunable degrees of freedom, and can we squeeze more performance by searching over them numerically?**

### What counts as "ILS-able"?

Any of the following makes a program a candidate for ILS:

| Type | Description | Example |
|------|-------------|---------|
| **A — Stochastic initialization** | The program uses random seeds or random placement; different seeds give different solutions | Circle packing: random initial center coordinates |
| **B — Continuous hyperparameters** | Fixed numeric defaults whose optimal values are unknown | Kalman filter: `process_variance=0.01`, `window_size=20` |
| **C — Structural/discrete parameters** | Discrete choices that affect solution structure | Number of clusters, polynomial degree |

Most non-trivial programs qualify as ILS-able. The engine defaults to treating any program with tunable parameters as a candidate.

### Reactive (current) vs. Proactive (future)

The current engine is **reactive**: it runs after OpenEvolve finishes and has no influence on what OpenEvolve evolves. A natural future direction is **proactive** integration, where OpenEvolve's LLM prompt is augmented to reward ILS-friendly program structure (explicit hyperparameters, separable stochastic components), creating a closed two-level optimization loop.

---

## 2. End-to-End Pipeline

The engine runs six sequential steps, driven by `ILSEngine.run()` in `ils_engine/engine.py`.

```
OpenEvolve output
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 0: Program Selection                                          │
│  Select top-K programs from checkpoint DB or best_program.py        │
│  Modes: top_1, vanilla_top_k, novelty_rejection                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  Program(s) with code + metrics
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: ILS Analysis          [LLM: analyzer_model]                │
│  Identify the ILS-able component in the program                     │
│  → ILSComponentSpec: name, type, location, degrees_of_freedom       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  Component spec
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: Strategy Planning     [LLM: planner_model]                 │
│  Design search strategies for this specific problem                 │
│  Always includes: perturb_best, random_restart                      │
│  Optionally adds: domain-specific Tier 2 strategies                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  List of SearchStrategy objects
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: Code Generation       [LLM: generator_model]               │
│  Generate a complete, runnable Python script per strategy           │
│  Each script implements the two-phase search (see §3)               │
│  Scripts are syntax-validated and saved to disk                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  Generated .py scripts
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: Execution                                                  │
│  Run all strategy scripts in parallel (one subprocess per strategy) │
│  Each script runs its own internal parallelism (ProcessPoolExecutor)│
│  Results parsed from JSON stdout lines                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  StrategyResult objects
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 5: Reporting                                                  │
│  Print terminal summary table                                       │
│  Save best_program.py, results_summary.json, per-strategy JSONs     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                      ils_output/<task>_<timestamp>/
```

### Step 0: Program Selection

The engine supports two input formats:

- **Checkpoint directory** (`openevolve_output/checkpoints/checkpoint_N/`) — loads all programs from the MAP-Elites database, enabling diversity-aware top-K selection.
- **Bare directory** (`examples/my_task/`) — looks for `best_program.py` and optionally `best_program_info.json`. If no stored metrics exist, the engine evaluates the program immediately using the evaluator.

Selection modes:
- `top_1`: always take just the single best
- `vanilla_top_k`: take the K highest-scoring programs
- `novelty_rejection` *(default)*: take up to K programs, rejecting candidates whose normalized edit distance to already-selected programs falls below `novelty_threshold`

### Step 1: ILS Analysis

The LLM reads the full program and identifies its most ILS-able component, returning a structured JSON spec:

```json
{
  "is_ils_able": true,
  "component": {
    "name": "Kalman Filter Hyperparameters",
    "type": "continuous",
    "location": "adaptive_filter, lines 83-107",
    "degrees_of_freedom": "process_variance, measurement_variance, window_size"
  },
  "reasoning": "Type B: fixed numeric defaults that control filter behavior"
}
```

If no ILS-able component is found, the program is skipped.

### Step 2: Strategy Planning

The LLM designs concrete search strategies adapted to the specific problem and component type. Two universal strategies are always generated:

- **`perturb_best`** (Tier 1): Gaussian perturbation of the current best solution. Sweeps a range of sigma values as a second axis alongside seeds.
- **`random_restart`** (Tier 1): Fully random initialization, independent of the current best. Pure exploration.

Optional **Tier 2** strategies are domain-specific (e.g., hexagonal grid initialization for circle packing). Disable with `--no-tier2`.

The total seed budget is split equally across all strategies.

### Step 3: Code Generation

For each strategy, the LLM generates a complete, standalone Python script that implements the **two-phase search** (see §3). The script is:

- Syntactically validated with `ast.parse()` before saving
- Retried once on syntax error
- Skipped if both attempts fail

Scripts embed the full original program code as a string literal and accept a standard set of CLI arguments for reproducibility.

### Step 4: Execution

All strategy scripts run in parallel via `ProcessPoolExecutor`, one subprocess per strategy. Each script manages its own internal parallelism (workers). Results are parsed from JSON lines printed to stdout.

If a script crashes or times out, the error is logged and the strategy is marked as FAILED without stopping other strategies.

### Step 5: Reporting

Results are printed as a summary table and saved to the timestamped output directory. If any strategy beats the baseline, the improved program is saved as `best_program.py`.

---

## 3. Two-Phase Search Architecture

Every generated script implements the same two-phase pattern, which is embedded as a reference skeleton in the code generation prompt.

### Phase 1 — Parallel Screening

```
for seed_i in range(phase1_budget):
    sigma = sigma_values[seed_i % len(sigma_values)]  # 2D axis: seed × sigma
    x0 = make_starting_point(seed_i, sigma)            # strategy-specific init
    result = scipy.optimize.minimize(objective, x0, method='SLSQP')
    if check_feasible(result.x):
        candidates.append((seed_i, score, result.x))

top_winners = sorted(candidates, reverse=True)[:phase1_top_k]
```

- All candidates are evaluated in parallel via `ProcessPoolExecutor`.
- `sigma` is swept as a second axis so every sigma value is sampled across many seeds.
- Perturbation strategies start from the *best known solution*, not from a raw grid.
- Only feasible results (passing problem-specific constraints) are kept.

### Phase 2 — Basin-Hopping Chains

```
for winner_seed, winner_x in top_winners:
    chain_seed = winner_seed + phase2_seed_offset
    step = PerturbStep(stepsize, seed=chain_seed)      # seeded RNG
    result = scipy.optimize.basinhopping(
        objective, winner_x,
        niter=niter,
        take_step=step,
        seed=chain_seed,                               # Markov chain seed
    )
```

- One independent Markov chain per Phase 1 winner, run in parallel.
- The `PerturbStep` class perturbs only the stochastic/tunable degrees of freedom.
- Both the step class RNG and the basinhopping sampler are seeded from the same integer → **fully reproducible** given the same seeds.
- Every accepted feasible step is recorded as `(step_idx, score, x)` for traceability.

### Reproducibility

A run is fully reproducible if you know:
- **Phase 1 seed**: the integer index of the Phase 1 candidate that became the winner
- **Phase 2 seed**: `winner_seed + phase2_seed_offset` (where `phase2_seed_offset` = `--phase2-seed`)

The final summary line printed by each script includes both values.

---

## 4. LLM Role at Each Step

The engine routes to three separately configurable LLM slots:

| Step | Config key | Default | Role |
|------|-----------|---------|------|
| 1 — Analysis | `analyzer_model` | `claude-sonnet-4-6` | Identify component type, location, degrees of freedom. Requires good code understanding but short output. |
| 2 — Planning | `planner_model` | `claude-sonnet-4-6` | Design strategy names, descriptions, sigma values, seed budgets. JSON output, moderate reasoning. |
| 3 — Generation | `generator_model` | `gpt-5.3-codex` | Write a complete, runnable ~300–600 line Python script. Most demanding; code-specialized model recommended. |

All three fall back to `llm_model` if not explicitly set. API calls route through OpenRouter (`OPENROUTER_API_KEY`).

---

## 5. Full Configuration Reference

### CLI Arguments (`openevolve-ils.py`)

#### Required

| Argument | Description |
|----------|-------------|
| `openevolve_output_path` | Path to OpenEvolve output directory (checkpoint dir or bare dir with `best_program.py`) |
| `evaluator_path` | Path to `evaluator.py` — must expose `evaluate(program_path) -> dict` |

#### Program Selection

| Argument | Default | Description |
|----------|---------|-------------|
| `--top-k` | `1` | Number of programs to run ILS on |
| `--selection-mode` | `novelty_rejection` | `top_1` \| `vanilla_top_k` \| `novelty_rejection` |
| `--novelty-threshold` | `0.3` | Minimum normalized edit distance between selected programs (novelty_rejection mode only) |

#### Problem Context

| Argument | Default | Description |
|----------|---------|-------------|
| `--problem-description` | `""` | Natural language description of the problem. Passed to all LLM steps. More specific = better strategy design. |
| `--score-metric` | `combined_score` | Key from `evaluator.evaluate()` dict to use as the objective. Set `combined_score` to auto-detect the primary metric (excludes internal OpenEvolve metrics). |

#### LLM Models

| Argument | Default | Description |
|----------|---------|-------------|
| `--llm-model` | `anthropic/claude-sonnet-4-6` | Fallback model for all three steps |
| `--analyzer-model` | *(uses llm-model)* | Model for Step 1 (ILS analysis) |
| `--planner-model` | *(uses llm-model)* | Model for Step 2 (strategy planning) |
| `--generator-model` | `openai/gpt-5.3-codex` | Model for Step 3 (code generation). Recommend a code-specialized model. |
| `--api-key` | `$OPENROUTER_API_KEY` | OpenRouter API key |

#### Strategies

| Argument | Default | Description |
|----------|---------|-------------|
| `--total-seeds` | `150` | Total evaluation budget, split equally across all strategies |
| `--no-tier2` | *(flag)* | Disable domain-specific (Tier 2) strategies; only run universal perturb_best and random_restart |
| `--max-tier2` | `2` | Maximum number of Tier 2 strategies to generate |

#### Execution

| Argument | Default | Description |
|----------|---------|-------------|
| `--workers` | `8` | Parallel workers passed to each strategy script's internal `ProcessPoolExecutor` |
| `--timeout` | `300` | Timeout per strategy script in seconds |

#### Two-Phase Search Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--phase1-budget` | *(script decides, ~total\_seeds/4)* | Number of candidates screened in Phase 1 |
| `--phase1-top-k` | `4` | Number of Phase 1 winners passed to Phase 2 |
| `--phase2-seed` | `0` | Seed offset added to each Phase 1 winner index for the Phase 2 Markov chain |
| `--niter` | `100` | Basin-hopping steps per Phase 2 chain |
| `--stepsize` | `0.04` | Perturbation magnitude for Phase 2 basin-hopping |

#### Output and Logging

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `ils_output` | Base output directory. Actual run saved under `<output-dir>/<task>_<timestamp>/` |
| `--verbose` / `-v` | *(flag)* | Enable DEBUG logging |

### ILSConfig Dataclass (`ils_engine/config.py`)

All CLI arguments map 1:1 to `ILSConfig` fields. When constructing `ILSConfig` directly (e.g., from Python), the same fields apply. Per-step model overrides use the `**{}` pattern so that `None` values do not shadow the dataclass defaults:

```python
config = ILSConfig(
    openevolve_output_path="examples/circle_packing",
    evaluator_path="examples/circle_packing/evaluator.py",
    **({} if args.generator_model is None else {"generator_model": args.generator_model}),
)
```

---

## 6. Quick-Start Examples

### Circle Packing (stochastic, Type A)

```bash
python openevolve-ils.py \
  examples/circle_packing/openevolve_output \
  examples/circle_packing/evaluator.py \
  --problem-description "Pack 26 circles of equal radius in a unit square, maximize sum of radii" \
  --score-metric sum_radii \
  --total-seeds 150 \
  --workers 8 \
  --phase1-budget 40 \
  --phase1-top-k 4 \
  --phase2-seed 42 \
  --niter 100 \
  --stepsize 0.04
```

### Signal Processing (hyperparameter tuning, Type B)

```bash
python openevolve-ils.py \
  examples/signal_processing \
  examples/signal_processing/evaluator.py \
  --problem-description "Adaptive Kalman filter for non-stationary time series denoising. Tunes process_variance, measurement_variance, and window_size to maximize correlation with the clean signal and noise reduction." \
  --total-seeds 1000 \
  --workers 10 \
  --timeout 300 \
  --no-tier2 \
  --phase1-budget 100 \
  --phase1-top-k 4 \
  --phase2-seed 42 \
  --niter 100 \
  --stepsize 0.1
```

### Using a checkpoint with top-K diverse programs

```bash
python openevolve-ils.py \
  openevolve_output/checkpoints/checkpoint_500 \
  examples/my_task/evaluator.py \
  --top-k 3 \
  --selection-mode novelty_rejection \
  --novelty-threshold 0.3 \
  --total-seeds 300 \
  --workers 8
```

---

## 7. Output Directory Structure

Each run creates a timestamped directory:

```
ils_output/
└── <task>_<YYYYMMDD_HHMMSS>/
    ├── best_program.py              # Best program found (or baseline if no improvement)
    ├── best_program_info.json       # Score, strategy, seed, improvement delta
    ├── results_summary.json         # All strategy results in one file
    ├── strategies/
    │   ├── perturb_best_results.json
    │   ├── random_restart_results.json
    │   └── <tier2_strategy>_results.json
    ├── _scripts_prog0/              # Generated .py scripts for program 0
    │   ├── perturb_best_search.py
    │   ├── random_restart_search.py
    │   └── ...
    └── _runner_prog0/               # Best programs found per strategy
        ├── perturb_best_best.py
        └── random_restart_best.py
```

The `best_program_info.json` format:

```json
{
  "score": 2.635983,
  "strategy": "hexagonal_grid",
  "seed": 162,
  "runtime_s": 247.3,
  "baseline_score": 2.601480,
  "improvement": 0.034503
}
```

Each strategy's results JSON contains `all_seed_scores` — a list of `{seed, score, sigma, phase}` records for every evaluated candidate, enabling full post-hoc analysis.

---

## 8. Design Decisions and Tradeoffs

### Why LLM-generated scripts instead of a fixed optimizer?

The search logic is domain-specific: circle packing needs geometric constraint enforcement and center-coordinate perturbation; a Kalman filter needs log-scale perturbation over orders-of-magnitude parameters; a neural architecture search needs different initialization entirely. A fixed optimization loop would require hand-coded adapters for each domain. By having the LLM generate the script from the component spec, the same engine handles any domain without modification.

### Why two phases instead of running basin-hopping directly?

Phase 1 (parallel SLSQP screening) quickly finds the most promising basins of the loss landscape from many random starting points. Phase 2 (basin-hopping) then applies a Markov chain only to the most promising regions. Running basin-hopping from a single or poorly-chosen starting point wastes most of the budget on an unproductive basin.

### Why sigma sweep as a 2D axis in Phase 1?

For perturbation strategies, sigma (perturbation magnitude) has a significant effect on which part of the neighborhood is explored. By sweeping `sigma = sigma_values[seed_i % len(sigma_values)]`, every sigma value gets proportional coverage regardless of budget size. This avoids the common failure mode of a fixed sigma that is too large (random walk) or too small (stuck at local optimum).

### Why per-step LLM model routing?

The three LLM steps have very different requirements:
- Analysis and planning need strong reasoning but produce short, structured output → a mid-tier model is sufficient.
- Code generation produces 300–600 lines of runnable Python with complex numerical optimization logic → a code-specialized model significantly reduces syntax errors and logical bugs.

Separating the models lets you spend budget where it matters.

### Score metric auto-detection

OpenEvolve stores a normalized `combined_score` (always in [0, 1]) alongside raw task metrics. Using `combined_score` as the ILS objective would make all programs appear equally good at ~0.999. The engine excludes known internal metrics (`combined_score`, `target_ratio`, `validity`, `eval_time`, `error`) and selects the highest-value remaining numeric metric as the true objective. This can always be overridden with `--score-metric`.
