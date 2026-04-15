"""
Step 3: LLM generates runnable search scripts for each strategy
"""

import ast
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GeneratedScript:
    strategy_name: str
    code: str
    script_path: str  # path where script is saved on disk
    ils_type: str = "A"
    language: str = "python"


_SYSTEM_PROMPT = (
    "You are an expert Python programmer specialising in combinatorial and numerical optimisation. "
    "Your task is to generate a complete, runnable Python script that implements a search strategy "
    "for a given optimisation program. "
    "The script must be syntactically valid Python and directly executable. "
    "Respond with raw Python code only — no markdown, no explanation, no code fences."
)


# ── Reference skeletons — one per (language × ils_type) combination ─────────

# Python Type A/B: two-phase SLSQP + basin-hopping (continuous search space)
_PHASE2_REFERENCE = """
# ── REFERENCE IMPLEMENTATION (two-phase basin-hopping) ──────────────────────
# The script you generate MUST follow this two-phase pattern exactly.
#
# PHASE 1 — parallel screening
#   • Generate `phase1_budget` candidate starting points using THIS strategy's
#     initialization logic (perturb grid / random restart / hexagonal / etc.).
#   • Run SLSQP on each candidate in parallel via ProcessPoolExecutor.
#   • Keep only feasible results; sort descending by score.
#   • Pass the top `phase1_top_k` winners (their optimized x vectors) to Phase 2.
#
# PHASE 2 — basin-hopping chains (scipy.optimize.basinhopping)
#   • For each Phase 1 winner, run an independent basin-hopping chain.
#   • Use a custom step class (like PerturbStep below) to perturb ONLY the
#     search degrees of freedom (e.g. layout parameters OR hyperparameter values),
#     not the deterministically-optimized values —
#     leave SLSQP free to re-optimise everything else.
#   • Seed BOTH the step class RNG and basinhopping(..., seed=...) from the
#     same integer so every run is fully reproducible given the same seeds.
#   • Record every accepted feasible step as (step_idx, score, x.copy()).
#   • After all chains finish, report the global best x.
#
# KEY SKELETON (adapt to your problem):
#
#   import numpy as np
#   from scipy.optimize import basinhopping, minimize
#   from concurrent.futures import ProcessPoolExecutor, as_completed
#
#   class PerturbStep:
#       def __init__(self, stepsize, seed):
#           self.stepsize = stepsize
#           self.rng = np.random.default_rng(seed)   # MUST be seeded
#       def __call__(self, x):
#           x = x.copy()
#           x[<stochastic_dims>] += self.rng.normal(0, self.stepsize,
#                                                   size=len(<stochastic_dims>))
#           x[<stochastic_dims>] = np.clip(x[<stochastic_dims>], lo, hi)
#           return x
#
#   def phase1_worker(seed, sigma):
#       x0 = make_starting_point(seed, sigma)   # THIS strategy's init logic
#       res = minimize(objective, x0, ...)
#       return seed, float(-res.fun), check_feasible(res.x), res.x
#
#   def phase2_worker(start_seed, phase2_seed, niter, stepsize, x0):
#       accepted = []
#       step_counter = [0]
#       def callback(x, f, accepted_flag):
#           step_counter[0] += 1
#           if accepted_flag and check_feasible(x):
#               accepted.append((step_counter[0], float(-f), x.copy()))
#       step = PerturbStep(stepsize, seed=phase2_seed)
#       result = basinhopping(objective, x0, niter=niter,
#                             minimizer_kwargs={...},
#                             take_step=step, callback=callback,
#                             seed=phase2_seed)
#       return float(-result.fun), check_feasible(result.x), accepted, result.x
#
#   # CLI args: --n-workers, --evaluator-path,
#   #           --phase1-budget (default: n_seeds // 4),
#   #           --phase1-top-k  (default: 4),
#   #           --phase2-seed   (default: 0, offset added to each winner seed),
#   #           --niter         (default: 100),
#   #           --stepsize      (default: 0.04)
#
#   # Sigma sweep in Phase 1 (for perturbation strategies):
#   #   for seed_i in range(phase1_budget):
#   #       sigma = sigma_values[seed_i % len(sigma_values)]
#   #   This ensures every sigma value is sampled with many different seeds.
#
# ── END REFERENCE ────────────────────────────────────────────────────────────
"""

# Python Type C: discrete ILS loop (ordering/combinatorial search, no scipy)
_DISCRETE_REFERENCE = """
# ── REFERENCE IMPLEMENTATION (discrete ILS — ordering/permutation search) ───
# Use this skeleton for Type C problems (piece ordering, job sequencing, etc.)
# DO NOT use scipy, SLSQP, or basin-hopping — those require continuous spaces.
#
# KEY SKELETON:
#
#   import random, json, os, sys, importlib.util, tempfile, argparse
#
#   ORIGINAL_CODE = \"\"\"...original program source...\"\"\"
#   N_ELEMENTS = <number of discrete elements being ordered>
#
#   def evaluate_ordering(ordering, evaluator):
#       \"\"\"Apply ordering to program, evaluate, return score.\"\"\"
#       new_code = inject_ordering(ORIGINAL_CODE, ordering)
#       with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
#           f.write(new_code)
#           tmp = f.name
#       try:
#           metrics = evaluator.evaluate(tmp)
#           return metrics.get(SCORE_METRIC, 0.0)
#       finally:
#           os.unlink(tmp)
#
#   def inject_ordering(code, ordering):
#       \"\"\"Substitute the ordering into the program source.\"\"\"
#       # Strategy-specific: replace the sort key, hardcode piece order, etc.
#       ...
#
#   def perturb_swap(ordering, rng):
#       \"\"\"Randomly swap two elements — small perturbation.\"\"\"
#       o = ordering[:]
#       i, j = rng.sample(range(len(o)), 2)
#       o[i], o[j] = o[j], o[i]
#       return o
#
#   def perturb_segment_reverse(ordering, rng, max_seg=None):
#       \"\"\"Reverse a random contiguous segment — medium perturbation.\"\"\"
#       o = ordering[:]
#       n = len(o)
#       if max_seg is None:
#           max_seg = max(2, n // 4)
#       length = rng.randint(2, max_seg)
#       start = rng.randint(0, n - length)
#       o[start:start+length] = reversed(o[start:start+length])
#       return o
#
#   def search_loop(seed, n_iterations, strategy_name, evaluator):
#       rng = random.Random(seed)
#       # Initialize ordering based on strategy
#       if strategy_name == 'ordering_random_restart':
#           current = list(range(N_ELEMENTS))
#           rng.shuffle(current)
#       elif strategy_name in ('ordering_swap', 'ordering_segment_reverse'):
#           current = BEST_KNOWN_ORDERING[:]  # loaded from previous best
#       else:
#           current = list(range(N_ELEMENTS))
#           rng.shuffle(current)
#
#       best = current[:]
#       best_score = evaluate_ordering(best, evaluator)
#       print(json.dumps({"seed": seed, "score": best_score, "strategy": strategy_name,
#                         "sigma": None, "phase": 1}), flush=True)
#
#       for iteration in range(n_iterations):
#           if strategy_name == 'ordering_swap':
#               candidate = perturb_swap(best, rng)
#           elif strategy_name == 'ordering_segment_reverse':
#               candidate = perturb_segment_reverse(best, rng)
#           else:
#               candidate = best[:]
#               rng.shuffle(candidate)
#
#           score = evaluate_ordering(candidate, evaluator)
#           if score > best_score:
#               best = candidate
#               best_score = score
#           print(json.dumps({"seed": seed, "score": score, "strategy": strategy_name,
#                             "sigma": None, "phase": 2}), flush=True)
#
#       return best_score, best
#
#   # CLI args: --n-workers, --evaluator-path, --n-seeds, --n-iterations (per seed)
#   # Run search_loop in parallel via ProcessPoolExecutor
#   # Output: one JSON line per evaluation + final summary line
#   # If BEST_PROGRAM_OUTPUT env var set, write best program to that path
#
# ── END REFERENCE ────────────────────────────────────────────────────────────
"""

# C++ Type A: compile-once env-var seed sweep (fast — avoids recompile per seed)
_CPP_SEED_REFERENCE = """
# ── REFERENCE IMPLEMENTATION (C++ Type A — compile-once, env-var seed sweep) ─
# PROBLEM: evaluator.evaluate() = full recompile + ALL test cases each call.
#          Calling it N times = N full compiles = hours for large N.
# SOLUTION: Compile ONCE yourself (subprocess g++), run binary N times cheaply
#           with different ILS_SEED env var values, then call evaluator ONCE at
#           the end with the best seed hardcoded into the source.
#
# KEY SKELETON:
#
#   import subprocess, tempfile, os, json, re, argparse, random
#   from concurrent.futures import ProcessPoolExecutor, as_completed
#   import importlib.util
#
#   ORIGINAL_CODE = \"\"\"...original C++ source...\"\"\"
#   SCORE_METRIC = 'combined_score'
#
#   def make_env_var_code(code):
#       \"\"\"Modify source ONCE to read seed from ILS_SEED env var.\"\"\"
#       # Replace bits/stdc++.h (GCC-only, not on macOS Apple Clang) with portable headers
#       if '#include <bits/stdc++.h>' in code:
#           code = code.replace('#include <bits/stdc++.h>',
#               '\\n'.join('#include <' + h + '>' for h in [
#                   'iostream', 'vector', 'algorithm', 'chrono', 'random',
#                   'queue', 'deque', 'set', 'map', 'unordered_map', 'unordered_set',
#                   'numeric', 'functional', 'cstdlib', 'climits', 'cstring',
#                   'string', 'sstream', 'utility', 'tuple', 'cassert',
#               ]))
#       # Replace time-based seed(s) with env var reading:
#       code = re.sub(
#           r'mt19937(_64)?\\s+(\\w+)\\s*\\(\\s*(?:chrono::|(?:unsigned\\s+)?int\\s*\\().*?\\)',
#           lambda m: f'mt19937{m.group(1) or \"\"} {m.group(2)}(atoi(getenv(\"ILS_SEED\") ? getenv(\"ILS_SEED\") : \"0\"))',
#           code
#       )
#       # Add #include <cstdlib> if not present (for getenv/atoi)
#       if '#include <cstdlib>' not in code:
#           code = '#include <cstdlib>\\n' + code
#       return code
#
#   def compile_once(code):
#       \"\"\"Compile modified source once, return path to binary.\"\"\"
#       with tempfile.NamedTemporaryFile(suffix='.cpp', mode='w', delete=False) as f:
#           f.write(code); src = f.name
#       binary = src.replace('.cpp', '_ils_bin')
#       # Try compilers in order: system g++, homebrew GCC variants (macOS)
#       compilers = ['g++', 'g++-14', 'g++-13', 'g++-12', 'g++-11']
#       last_err = ''
#       for compiler in compilers:
#           result = subprocess.run(
#               [compiler, '-O2', '-std=c++17', '-o', binary, src],
#               capture_output=True, text=True
#           )
#           if result.returncode == 0:
#               os.unlink(src)
#               return binary
#           last_err = result.stderr
#       os.unlink(src)
#       raise RuntimeError(f'Compilation failed with all compilers:\\n{last_err}')
#
#   def generate_test_case(rng):
#       \"\"\"Generate one synthetic test case as an input string.
#       PROBLEM-SPECIFIC — read the problem description and match the input format.
#       Example for Max-Cut (graph partition):
#           n = rng.randint(50, 200)
#           edges = set()
#           for _ in range(n * 3):
#               u, v = rng.randint(1, n), rng.randint(1, n)
#               if u != v: edges.add((min(u,v), max(u,v)))
#           return f\"{n} {len(edges)}\\n\" + \"\\n\".join(f\"{u} {v}\" for u,v in edges)
#       \"\"\"
#       ...  # LLM FILLS IN for this specific problem
#
#   def score_output(stdout, test_input):
#       \"\"\"Compute score from binary stdout and the test input string.
#       PROBLEM-SPECIFIC — implement the scoring formula from the problem statement.
#       Example for Max-Cut:
#           lines = test_input.strip().split('\\n')
#           n, m = map(int, lines[0].split())
#           edges = [tuple(map(int, l.split())) for l in lines[1:m+1]]
#           side = list(map(int, stdout.split()))
#           cut = sum(1 for u,v in edges if side[u-1] != side[v-1])
#           return cut / m if m > 0 else 1.0
#       \"\"\"
#       ...  # LLM FILLS IN for this specific problem
#
#   def run_seed(seed, binary, test_cases):
#       \"\"\"Run compiled binary with ILS_SEED=seed on all test cases.\"\"\"
#       env = os.environ.copy(); env['ILS_SEED'] = str(seed)
#       scores = []
#       for test_input in test_cases:
#           try:
#               r = subprocess.run([binary], input=test_input, capture_output=True,
#                                  text=True, env=env, timeout=10)
#               scores.append(score_output(r.stdout.strip(), test_input))
#           except Exception:
#               scores.append(0.0)
#       avg = sum(scores) / len(scores) if scores else 0.0
#       return seed, avg
#
#   # ── MAIN FLOW ────────────────────────────────────────────────────────────
#
#   # Step 1: compile once
#   modified_code = make_env_var_code(ORIGINAL_CODE)
#   binary = compile_once(modified_code)
#
#   # Step 2: generate synthetic test cases
#   rng = random.Random(42)
#   test_cases = [generate_test_case(rng) for _ in range(20)]
#
#   # Step 3: sweep seeds in parallel (cheap — no recompile)
#   best_seed, best_score = 0, -float('inf')
#   with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
#       futures = {ex.submit(run_seed, s, binary, test_cases): s for s in range(N_SEEDS)}
#       for fut in as_completed(futures):
#           seed, score = fut.result()
#           print(json.dumps({'seed': seed, 'score': score, 'strategy': STRATEGY_NAME,
#                             'sigma': None, 'phase': 1}), flush=True)
#           if score > best_score:
#               best_score, best_seed = score, seed
#
#   # Step 4: hardcode best seed into ORIGINAL_CODE (not the env-var version)
#   best_code = re.sub(
#       r'mt19937(_64)?\\s+(\\w+)\\s*\\(\\s*(?:chrono::|(?:unsigned\\s+)?int\\s*\\().*?\\)',
#       lambda m: f'mt19937{m.group(1) or \"\"} {m.group(2)}({best_seed})',
#       ORIGINAL_CODE
#   )
#   # Also handle seed ^= (uintptr_t)&seed; lines — remove them
#   best_code = re.sub(r'.*seed\\s*\\^=.*\\n', '', best_code)
#
#   # Step 5: call official evaluator ONCE with best code
#   spec = importlib.util.spec_from_file_location('evaluator', EVALUATOR_PATH)
#   ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
#   with tempfile.NamedTemporaryFile(suffix='.cpp', mode='w', delete=False) as f:
#       f.write(best_code); tmp = f.name
#   try:
#       metrics = ev.evaluate(tmp)
#       final_score = metrics.get(SCORE_METRIC, 0.0) if isinstance(metrics, dict) else 0.0
#   finally:
#       os.unlink(tmp)
#   os.unlink(binary)
#
#   print(json.dumps({'type': 'summary', 'best_score': final_score,
#                     'best_seed': best_seed, 'best_phase2_seed': None, 'best_step': None}))
#   if os.environ.get('BEST_PROGRAM_OUTPUT'):
#       with open(os.environ['BEST_PROGRAM_OUTPUT'], 'w') as f:
#           f.write(best_code)
#
# NOTES:
#   • N_SEEDS = 300+ is feasible (each run costs only ~1-2s, no recompile)
#   • generate_test_case() and score_output() MUST match the real problem format
#   • Synthetic scores ≈ official scores when test distribution matches
#   • The regex patterns above are examples — adjust to match the actual RNG line(s)
#
# ── END REFERENCE ────────────────────────────────────────────────────────────
"""

# C++ Type C: internal ILS loop embedded in C++ binary (compile-once approach)
_CPP_TYPE_C_REFERENCE = """
# ── REFERENCE IMPLEMENTATION (C++ Type C — internal ILS, compile-once) ───────
# For C++ Type C programs where the ordering/sequence drives the result.
# CRITICAL: Each evaluator.evaluate() call = full recompile + ALL test cases.
#            DO NOT recompile per ordering — that is prohibitively expensive.
#
# STRATEGY: Generate a MODIFIED C++ program that runs ILS internally within its
# time limit, rather than recompiling per ordering from Python.
# The Python wrapper submits only a few variants (different RNG seeds) to the
# evaluator — each variant runs the full internal search on every test case.
#
# INTERNAL ILS PATTERN (embed this into the C++ source):
#
#   #include <chrono>
#   // After reading input, before main output:
#
#   auto t0 = chrono::steady_clock::now();
#   auto elapsed_s = [&]() {
#       return chrono::duration<double>(chrono::steady_clock::now() - t0).count();
#   };
#
#   mt19937 rng(RNG_SEED);  // RNG_SEED is substituted per variant (e.g. 42, 137, 999)
#
#   // Initialize ordering with a good heuristic
#   vector<int> order(n);
#   iota(order.begin(), order.end(), 0);
#   sort(order.begin(), order.end(), [&](int a, int b) {
#       // e.g. decreasing bounding-box area
#       return pieces[a].w * pieces[a].h > pieces[b].w * pieces[b].h;
#   });
#
#   int bestL = evaluate_ordering(order);   // call the pack/evaluate function
#   vector<int> bestOrder = order;
#
#   const double TIME_LIMIT = 1.2;  // conservative — leaves margin for I/O + setup (problem limit is 2s)
#   while (elapsed_s() < TIME_LIMIT) {
#       vector<int> cand = bestOrder;
#       // Perturbation (choose one per strategy):
#       //   swap:    swap two random elements
#       //   reverse: reverse a random contiguous segment
#       //   shuffle: fully re-shuffle
#       int i = rng() % n, j = rng() % n;
#       swap(cand[i], cand[j]);
#       int L = evaluate_ordering(cand);
#       if (L < bestL) { bestL = L; bestOrder = cand; }
#   }
#   // Use bestOrder for final output
#
# PYTHON WRAPPER:
#   1. Embed the ILS loop into ORIGINAL_CODE by string substitution / injection.
#   2. Try N_VARIANTS variants (e.g. 5), each with a different RNG_SEED substituted.
#   3. Call evaluator.evaluate(tmp_cpp_path) once per variant — evaluator compiles.
#   4. Report the best variant score.
#
#   ORIGINAL_CODE = \"\"\"...original C++ source (verbatim)...\"\"\"
#   SCORE_METRIC = 'combined_score'
#   N_VARIANTS = 5   # small — each variant = full compile + all test cases
#
#   def make_variant(rng_seed):
#       \"\"\"Inject ILS loop and RNG seed into C++ source.\"\"\"
#       code = ORIGINAL_CODE
#       # Insert the ILS loop (strategy-specific perturbation) into the source
#       # Replace any existing rng seed or add a new one
#       code = code.replace('RNG_SEED_PLACEHOLDER', str(rng_seed))
#       return code
#
#   def evaluate_variant(rng_seed, evaluator_path):
#       import importlib.util
#       spec = importlib.util.spec_from_file_location('evaluator', evaluator_path)
#       ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
#       code = make_variant(rng_seed)
#       with tempfile.NamedTemporaryFile(suffix='.cpp', mode='w', delete=False) as f:
#           f.write(code); tmp = f.name
#       try:
#           metrics = ev.evaluate(tmp)
#           score = metrics.get(SCORE_METRIC, 0.0) if isinstance(metrics, dict) else 0.0
#           return rng_seed, score, code
#       finally:
#           os.unlink(tmp)
#
#   # CLI: --n-workers, --evaluator-path, --n-variants (default 5)
#   # Run evaluate_variant for seeds [42, 137, 999, 2718, 31415] or similar
#   # Print one JSON line per variant + summary line
#   # If BEST_PROGRAM_OUTPUT set, write best modified C++ source to that path
#
# ── END REFERENCE ────────────────────────────────────────────────────────────
"""

# C++ Type B: Bayesian optimization over source-substituted parameters
_CPP_BAYES_REFERENCE = """
# ── REFERENCE IMPLEMENTATION (C++ parameter search — Bayesian optimization) ──
# For C++ Type B programs with hardcoded numerical parameters.
# Each evaluation: text-substitute the parameter values → write .cpp → evaluator compiles.
# Use Bayesian optimization (skopt.gp_minimize) — budget ~60 evaluations max.
# If skopt unavailable, fall back to grid search or random sampling.
#
# KEY SKELETON:
#
#   import json, os, sys, importlib.util, tempfile, re, argparse
#
#   ORIGINAL_CODE = \"\"\"...original C++ source...\"\"\"
#   SCORE_METRIC = 'combined_score'
#   PARAM_NAMES = ['alpha', 'beta']   # names for logging
#   PARAM_BOUNDS = [(0.1, 5.0), (0.1, 5.0)]  # LLM estimates reasonable ranges
#
#   # Text substitution patterns for each parameter
#   PARAM_PATTERNS = [
#       (r'items\\[a\\]\\.m \\+ items\\[a\\]\\.l',
#        lambda a, b: f'{a}*items[a].m + {b}*items[a].l'),
#   ]
#
#   def inject_params(code, params):
#       \"\"\"Substitute parameter values into C++ source.\"\"\"
#       # Apply text substitutions — be precise to avoid wrong replacements
#       ...
#       return code
#
#   def objective(params, evaluator_path):
#       import importlib.util
#       spec = importlib.util.spec_from_file_location('evaluator', evaluator_path)
#       evaluator = importlib.util.module_from_spec(spec)
#       spec.loader.exec_module(evaluator)
#       new_code = inject_params(ORIGINAL_CODE, params)
#       with tempfile.NamedTemporaryFile(suffix='.cpp', mode='w', delete=False) as f:
#           f.write(new_code)
#           tmp = f.name
#       try:
#           metrics = evaluator.evaluate(tmp)
#           score = metrics.get(SCORE_METRIC, 0.0) if isinstance(metrics, dict) else 0.0
#           print(json.dumps({"params": list(params), "score": score, "strategy": STRATEGY_NAME,
#                             "sigma": None, "phase": 1}), flush=True)
#           return -score   # gp_minimize minimizes
#       finally:
#           os.unlink(tmp)
#
#   try:
#       from skopt import gp_minimize
#       result = gp_minimize(lambda p: objective(p, EVALUATOR_PATH),
#                            PARAM_BOUNDS, n_calls=60, random_state=42)
#       best_params = result.x
#       best_score = -result.fun
#   except ImportError:
#       # Fallback: random search
#       import random; rng = random.Random(42)
#       best_score, best_params = -float('inf'), None
#       for _ in range(60):
#           p = [rng.uniform(*b) for b in PARAM_BOUNDS]
#           s = -objective(p, EVALUATOR_PATH)
#           if s > best_score:
#               best_score, best_params = s, p
#
#   print(json.dumps({"type": "summary", "best_score": best_score,
#                     "best_seed": 0, "best_phase2_seed": None, "best_step": None}))
#   if os.environ.get('BEST_PROGRAM_OUTPUT'):
#       with open(os.environ['BEST_PROGRAM_OUTPUT'], 'w') as f:
#           f.write(inject_params(ORIGINAL_CODE, best_params))
#
# ── END REFERENCE ────────────────────────────────────────────────────────────
"""


def _select_reference(ils_type: str, language: str) -> str:
    """Select the appropriate reference skeleton for this type × language combination."""
    if language == "cpp":
        if ils_type == "B":
            return _CPP_BAYES_REFERENCE
        elif ils_type == "C":
            return _CPP_TYPE_C_REFERENCE  # compile-once internal ILS
        else:  # A, unknown
            return _CPP_SEED_REFERENCE
    else:  # python
        if ils_type == "C":
            return _DISCRETE_REFERENCE
        else:  # A, B, unknown
            return _PHASE2_REFERENCE


def _build_prompt(
    problem_description: str,
    program_code: str,
    component_name: str,
    component_description: str,
    component_type: str,
    component_location: str,
    degrees_of_freedom: str,
    strategy_name: str,
    strategy_description: str,
    strategy_tier: int,
    seeds: int,
    sigma_values: list,
    strategy_justification: str,
    evaluator_path: str,
    score_metric: str = "combined_score",
    ils_type: str = "A",
    language: str = "python",
) -> str:
    sigma_str = json.dumps(sigma_values) if sigma_values else "[]"
    reference = _select_reference(ils_type, language)

    # Build type × language specific structure instructions
    # For C++ scripts: use parse_known_args so Python-only runner args are silently ignored
    parse_args_note = (
        "  IMPORTANT: Use `parser.parse_known_args()` instead of `parser.parse_args()` "
        "so that unrecognized arguments from the runner are silently ignored."
        if language == "cpp"
        else ""
    )

    if language == "cpp" and ils_type == "B":
        structure_instructions = [
            "SEARCH STRUCTURE (C++ Type B — Bayesian optimization):",
            "  • Use Bayesian optimization (skopt.gp_minimize) with budget ~60 evaluations.",
            "  • Each evaluation: text-substitute parameter values into ORIGINAL_CODE,",
            "    write to a .cpp tempfile, call evaluator.evaluate(tmp_path) — evaluator compiles.",
            "  • If skopt unavailable, fall back to random search with 60 samples.",
            "  • NO scipy SLSQP. NO basin-hopping. Budget is expensive (compiles each time).",
            f"  • Estimated reasonable parameter ranges: derive from code context and problem.",
            "",
            "EVALUATOR USAGE (C++ — CRITICAL):",
            "  Load:  importlib.util.spec_from_file_location('evaluator', EVALUATOR_PATH)",
            "  Call:  metrics = evaluator.evaluate(tmp_cpp_path)  # evaluator compiles internally",
            f"  Score: score = metrics.get('{score_metric}', 0.0)",
            "  Tempfile suffix MUST be '.cpp' — NOT '.py'",
            f"  The score metric is '{score_metric}'. Use THIS key ONLY.",
            "",
            "CLI ARGUMENTS:",
            "  --n-workers       INT  (ignored for Bayesian opt — sequential)",
            "  --evaluator-path  STR  override EVALUATOR_PATH",
            "  --n-calls         INT  Bayesian opt evaluations (default 60)",
            parse_args_note,
        ]
    elif language == "cpp" and ils_type == "C":
        n_variants = min(seeds, 10)  # very few — each compile = all test cases
        structure_instructions = [
            "SEARCH STRUCTURE (C++ Type C — internal ILS, compile-once):",
            "  CRITICAL: evaluator.evaluate() = full recompile + ALL test cases each call.",
            "  DO NOT loop Python-side recompiling per ordering — that is too expensive.",
            "",
            "  Instead, generate a MODIFIED C++ source that embeds the ILS loop internally:",
            "  • Inject a time-bounded ILS loop (swap/reverse/shuffle perturbations) directly",
            "    into the C++ source, running within the problem's time limit (e.g. 1.8s).",
            "  • The C++ program reads input, runs internal ILS, outputs the best solution found.",
            "  • Use chrono::steady_clock to stay within the time limit.",
            "  • Set TIME_LIMIT = 1.2 (NOT 1.8) — input reading + orientation setup",
            "    can take 0.3-0.5s on large inputs, leaving only ~1.5s before the 2s wall.",
            "  • TLE = score 0 (judge rejects entire submission). Be conservative.",
            "  • The Python wrapper submits only N_VARIANTS versions with different RNG seeds.",
            f"  • Try {n_variants} variants total — each is one compile + full evaluation.",
            "  • Run variants in parallel via ProcessPoolExecutor.",
            "",
            "  EMBEDDING THE ILS LOOP:",
            "  • Identify where the ordering is determined in ORIGINAL_CODE.",
            "  • Replace/augment it with a hill-climbing loop over orderings.",
            "  • Use a placeholder like RNG_SEED_PLACEHOLDER that Python substitutes per variant.",
            "  • Keep the same pack/evaluate logic — just try many orderings of it.",
            "",
            "EVALUATOR USAGE (C++ — CRITICAL):",
            "  Load:  importlib.util.spec_from_file_location('evaluator', EVALUATOR_PATH)",
            "  Call:  metrics = evaluator.evaluate(tmp_cpp_path)  # evaluator compiles internally",
            f"  Score: score = metrics.get('{score_metric}', 0.0)",
            "  Tempfile suffix MUST be '.cpp' — NOT '.py'",
            f"  The score metric is '{score_metric}'. Use THIS key ONLY.",
            "",
            "CLI ARGUMENTS:",
            "  --n-workers       INT  parallel workers (default 4)",
            "  --evaluator-path  STR  override EVALUATOR_PATH",
            f"  --n-variants      INT  number of RNG seed variants to try (default {n_variants})",
            parse_args_note,
        ]
    elif language == "cpp":  # A or unknown — compile-once env-var seed sweep
        n_seeds = max(seeds * 5, 300)  # many seeds are cheap — no recompile per seed
        structure_instructions = [
            f"SEARCH STRUCTURE (C++ Type {ils_type} — compile-once, env-var seed sweep):",
            "  CRITICAL: Do NOT call evaluator.evaluate() in a loop — that recompiles every time.",
            "  Instead use the compile-once pattern from the reference skeleton above:",
            "",
            f"  1. Call make_env_var_code(ORIGINAL_CODE) to patch the time-based RNG seed to read",
            "     from getenv('ILS_SEED'). Add #include <cstdlib> if missing.",
            "  2. Compile ONCE with subprocess g++ -O2 -std=c++17.",
            f"  3. Generate 20 synthetic test cases using generate_test_case() — read the problem",
            "     description carefully and match the real input format exactly.",
            "  4. Implement score_output() using the scoring formula from the problem description.",
            f"  5. Sweep {n_seeds} seeds in parallel (cheap — each run costs ~1-2s, no recompile).",
            "  6. Hardcode the best seed as a literal integer into ORIGINAL_CODE (not env-var version).",
            "  7. Call evaluator.evaluate() EXACTLY ONCE with the best-seed code for the final score.",
            "",
            "  generate_test_case() and score_output() are the most important parts — implement",
            "  them specifically for this problem. Wrong test distribution = misleading seed ranking.",
            "",
            "EVALUATOR USAGE (called ONCE at the end only):",
            "  Load:  importlib.util.spec_from_file_location('evaluator', EVALUATOR_PATH)",
            "  Call:  metrics = evaluator.evaluate(tmp_cpp_path)  # evaluator compiles internally",
            f"  Score: score = metrics.get('{score_metric}', 0.0)",
            "  Tempfile suffix MUST be '.cpp' — NOT '.py'",
            f"  The score metric is '{score_metric}'. Use THIS key ONLY.",
            "",
            "CLI ARGUMENTS:",
            "  --n-workers       INT  parallel workers for seed sweep (default 4)",
            "  --evaluator-path  STR  override EVALUATOR_PATH",
            f"  --n-seeds         INT  seeds to sweep (default {n_seeds})",
            parse_args_note,
        ]
    elif ils_type == "C":  # Python + C
        structure_instructions = [
            "SEARCH STRUCTURE (Python Type C — discrete ordering ILS):",
            "  • Implement a discrete perturbation loop — NO scipy, NO SLSQP, NO basin-hopping.",
            "  • The search space is orderings/permutations of discrete elements.",
            "  • Each evaluation: inject the ordering into ORIGINAL_CODE (modify sort key,",
            "    hardcode element order, or substitute list literals), write to .py tempfile,",
            "    call evaluator.evaluate(tmp_path).",
            f"  • Run {seeds} total evaluations across --n-seeds parallel workers.",
            "  • For ordering_random_restart: shuffle from scratch each time.",
            "  • For ordering_swap: start from best known ordering, swap 2 elements.",
            "  • For ordering_segment_reverse: reverse a random contiguous segment.",
            "  • Accept new ordering if score improves (greedy hill-climbing).",
            "",
            "EVALUATOR USAGE:",
            "  Load:  importlib.util.spec_from_file_location('evaluator', EVALUATOR_PATH)",
            "  Call:  metrics = evaluator.evaluate(tmp_py_path)  # returns a DICT",
            f"  Score: score = metrics.get('{score_metric}', 0.0)",
            f"  The score metric is '{score_metric}'. Use THIS key ONLY.",
            "  Tempfile suffix must be '.py'",
            "",
            "CLI ARGUMENTS:",
            "  --n-workers        INT   parallel workers (default 1)",
            "  --evaluator-path   STR   override EVALUATOR_PATH",
            f"  --n-seeds          INT   number of independent search runs (default {max(seeds // 10, 5)})",
            "  --n-iterations     INT   perturbation steps per run (default 20)",
        ]
    else:  # Python A or B — two-phase SLSQP + basin-hopping
        phase1_budget = max(seeds // 4, 20)
        structure_instructions = [
            "MANDATORY TWO-PHASE STRUCTURE:",
            f"Phase 1 — parallel screening ({phase1_budget} candidates by default via --phase1-budget):",
            "  • Implement THIS strategy's initialization logic to generate starting points.",
            "  • For perturbation strategies: sweep sigma as a 2D axis alongside seed:",
            f"      sigma = sigma_values[seed_i % len(sigma_values)]  # ALL values in {sigma_str}",
            "    Start perturbation from the best optimized solution in ORIGINAL_CODE, not a raw grid.",
            "  • For random_restart: generate fully random initializations.",
            "  • For domain-specific (tier 2): use the structural layout described above.",
            "  • Run SLSQP on each candidate in parallel; keep feasible results sorted by score.",
            "  • Pass top --phase1-top-k (default 4) winners' optimized x vectors to Phase 2.",
            "",
            f"Phase 2 — basin-hopping chains (--niter steps each, default 100):",
            "  • Run one independent scipy.optimize.basinhopping chain per Phase 1 winner.",
            "  • Use a custom step class that perturbs ONLY the search degrees of freedom,",
            "    seeded with: winner_seed + phase2_seed_offset.",
            "  • Also pass seed=winner_seed+phase2_seed_offset to basinhopping() itself.",
            "  • Record every accepted feasible step as (step_idx, score, x.copy()) in callback.",
            "  • Run Phase 2 chains in parallel via ProcessPoolExecutor.",
            "",
            "EVALUATOR USAGE:",
            "  Load:  importlib.util.spec_from_file_location('evaluator', EVALUATOR_PATH)",
            "  Call:  metrics = evaluator.evaluate(tmp_program_path)  # returns a DICT",
            f"  Score: score = metrics.get('{score_metric}', 0.0)",
            f"  The score metric is '{score_metric}'. Use THIS key ONLY.",
            "  Tempfile suffix must be '.py'",
            "",
            "CLI ARGUMENTS (all scripts must support these):",
            "  --n-workers        INT   parallel workers (default 1)",
            "  --evaluator-path   STR   override EVALUATOR_PATH",
            "  --phase1-budget    INT   Phase 1 candidates (default {})".format(phase1_budget),
            "  --phase1-top-k     INT   winners passed to Phase 2 (default 4)",
            "  --phase2-seed      INT   offset added to each winner seed for Phase 2 (default 0)",
            "  --niter            INT   basin-hopping steps per chain (default 100)",
            "  --stepsize         FLOAT perturbation magnitude (default 0.04)",
        ]

    lines = (
        [
            f"Generate a complete, runnable Python search script for the following ILS strategy.",
            f"ILS Type: {ils_type} | Language: {language} | Strategy: {strategy_name}",
            "",
            reference,
            "",
            "PROBLEM DESCRIPTION:",
            problem_description or "No description provided.",
            "",
            "ORIGINAL PROGRAM CODE (include this verbatim as ORIGINAL_CODE in the script):",
            "--- BEGIN PROGRAM ---",
            program_code,
            "--- END PROGRAM ---",
            "",
            "ILS COMPONENT:",
            f"Name: {component_name}",
            f"Description: {component_description}",
            f"Type: {component_type}",
            f"ILS Type: {ils_type}",
            f"Language: {language}",
            f"Location: {component_location}",
            f"Degrees of freedom: {degrees_of_freedom}",
            "",
            "STRATEGY TO IMPLEMENT:",
            f"Name: {strategy_name}",
            f"Description: {strategy_description}",
            f"Tier: {strategy_tier}",
            f"Total seed/evaluation budget: {seeds}",
            f"Sigma values (for continuous perturbation strategies only): {sigma_str}",
            f"Justification: {strategy_justification}",
            "",
            f"EVALUATOR PATH: {evaluator_path}",
            "",
        ]
        + structure_instructions
        + [
            "",
            "OUTPUT FORMAT (all scripts must emit this):",
            "  Each evaluated candidate must emit a JSON line to stdout:",
            '  {"seed": <int>, "score": <float>, "strategy": "<name>", "sigma": <float_or_null>, "phase": <1_or_2>}',
            "  Final summary line:",
            '  {"type": "summary", "best_score": <float>, "best_seed": <int>, "best_phase2_seed": <int_or_null>, "best_step": <int_or_null>}',
            "  If env var BEST_PROGRAM_OUTPUT is set, write the best program code to that path.",
            "",
            "OTHER REQUIREMENTS:",
            "  • Standalone script — no deps beyond numpy/scipy/scikit-optimize/standard library.",
            "  • Store ORIGINAL_CODE as a module-level string variable.",
            "  • Use concurrent.futures.ProcessPoolExecutor for parallelism where applicable.",
            "",
            "Write the complete, working Python script now. No markdown fences, just raw Python code.",
        ]
    )
    return "\n".join(lines)


def _call_llm_for_code(prompt: str, llm_model: str, client) -> str:
    """Call LLM and return generated code string."""
    response = client.chat.completions.create(
        model=llm_model,
        max_tokens=16384,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if LLM added them anyway
    for fence in ("```python", "```"):
        if raw.startswith(fence):
            raw = raw[len(fence) :]
            break
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def generate_scripts(
    strategies: List,
    program_code: str,
    evaluator_path: str,
    component_spec,
    problem_description: str = "",
    output_dir: str = "/tmp/ils_scripts",
    llm_model: str = "claude-sonnet-4-6",
    llm_api_key: Optional[str] = None,
    score_metric: str = "combined_score",
) -> List[GeneratedScript]:
    """
    Generate runnable search scripts for each strategy.

    Args:
        strategies: List of SearchStrategy objects
        program_code: Original program source code
        evaluator_path: Absolute path to evaluator.py
        component_spec: ILSComponentSpec
        problem_description: Optional domain context
        output_dir: Directory to save generated scripts
        llm_model: Anthropic model name
        llm_api_key: API key (defaults to ANTHROPIC_API_KEY env var)

    Returns:
        List of GeneratedScript objects
    """
    from openai import OpenAI

    api_key = llm_api_key or os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    os.makedirs(output_dir, exist_ok=True)
    evaluator_path = str(Path(evaluator_path).resolve())

    scripts = []
    ils_type = getattr(component_spec, "ils_type", "A") or "A"
    language = getattr(component_spec, "language", "python") or "python"

    for strategy in strategies:
        logger.info(
            f"Generating code for strategy: {strategy.name} (type={ils_type}, lang={language})"
        )

        prompt = _build_prompt(
            problem_description=problem_description or "No description provided.",
            program_code=program_code,
            component_name=component_spec.name,
            component_description=component_spec.description,
            component_type=component_spec.type,
            component_location=component_spec.location,
            degrees_of_freedom=component_spec.degrees_of_freedom,
            strategy_name=strategy.name,
            strategy_description=strategy.description,
            strategy_tier=strategy.tier,
            seeds=strategy.seeds,
            sigma_values=strategy.sigma_values,
            strategy_justification=strategy.justification,
            evaluator_path=evaluator_path,
            score_metric=score_metric,
            ils_type=ils_type,
            language=language,
        )

        code = None
        for attempt in range(2):
            try:
                generated = _call_llm_for_code(prompt, llm_model, client)

                # Validate syntax
                ast.parse(generated)
                code = generated
                logger.info(
                    f"Strategy '{strategy.name}': code generated and validated (attempt {attempt+1})"
                )
                break

            except SyntaxError as e:
                if attempt == 0:
                    logger.warning(
                        f"Strategy '{strategy.name}': syntax error on attempt 1: {e}. Retrying..."
                    )
                    prompt = prompt + f"\n\nPREVIOUS ATTEMPT HAD SYNTAX ERROR:\n{e}\nPlease fix it."
                else:
                    logger.error(
                        f"Strategy '{strategy.name}': syntax error persists after retry: {e}"
                    )
            except Exception as e:
                logger.error(f"Strategy '{strategy.name}': LLM call failed: {e}")
                break

        if code is None:
            logger.warning(f"Skipping strategy '{strategy.name}' due to code generation failure")
            continue

        # Save script to disk
        script_path = os.path.join(output_dir, f"{strategy.name}_search.py")
        with open(script_path, "w") as f:
            f.write(code)

        scripts.append(
            GeneratedScript(
                strategy_name=strategy.name,
                code=code,
                script_path=script_path,
                ils_type=ils_type,
                language=language,
            )
        )

    logger.info(f"Generated {len(scripts)}/{len(strategies)} scripts successfully")
    return scripts
