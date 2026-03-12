"""
Two-phase basin-hopping search for circle packing (n=26).

Phase 1 — Random screening:
  Run SLSQP from many random center layouts in parallel.
  Pick the top `--starts` feasible results as starting points.

Phase 2 — Basin-hopping from winners:
  scipy.optimize.basinhopping repeatedly:
    1. Perturbs the current best solution (escapes local optimum)
    2. Runs SLSQP from the perturbed point (finds new local optimum)
    3. Accepts/rejects based on score (with temperature T)
  Perturbation acts on the *optimized result*, not just the starting guess.

Usage:
    python playground-5-basinhopping.py
    python playground-5-basinhopping.py --phase1-budget 200 --starts 4 --niter 100
    python playground-5-basinhopping.py --phase1-seed 42 --niter 200 --stepsize 0.03

Arguments:
    --phase1-budget  Random seeds to screen in Phase 1 (default: 100)
    --phase1-seed    Base random seed for Phase 1 (default: 0)
    --starts         Top-k Phase 1 winners to pass to Phase 2 (default: 4)
    --niter          Basin-hopping steps per start in Phase 2 (default: 100)
    --stepsize       Perturbation magnitude per step (default: 0.04)
    --T              Acceptance temperature: higher = wider exploration (default: 0.01)
    --workers        Parallel workers (used in both phases, default: 4)
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import basinhopping, minimize

N = 26
SOLVER_OPTS = {"maxiter": 1000, "ftol": 1e-8}


# ---------------------------------------------------------------------------
# Objective and constraints (module-level for pickling in multiprocessing)
# ---------------------------------------------------------------------------

def objective(x):
    return -np.sum(x[2*N:])


def constraint_fn(x):
    n = N
    c = x[:2*n].reshape(n, 2)
    r = x[2*n:]
    cons = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((c[i] - c[j])**2))
            cons.append(d - (r[i] + r[j]))
    for i in range(n):
        cons.append(c[i, 0] - r[i])
        cons.append(1 - c[i, 0] - r[i])
        cons.append(c[i, 1] - r[i])
        cons.append(1 - c[i, 1] - r[i])
    return np.array(cons)


BOUNDS      = [(0, 1)] * (2 * N) + [(0.03, 0.2)] * N
CONSTRAINTS = {"type": "ineq", "fun": constraint_fn}


# ---------------------------------------------------------------------------
# Custom step: perturb only center coordinates, clip to valid range
# ---------------------------------------------------------------------------

class CenterStep:
    """Perturb center positions only; leave radii for SLSQP to optimize."""

    def __init__(self, stepsize, seed=None):
        self.stepsize = stepsize
        self.rng = np.random.default_rng(seed)

    def __call__(self, x):
        x = x.copy()
        # perturb only center coordinates (first 2*N values)
        x[:2*N] += self.rng.normal(0, self.stepsize, size=2*N)
        x[:2*N] = np.clip(x[:2*N], 0.01, 0.99)
        return x


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

def check_feasible(x, tol=1e-9):
    n = N
    c = x[:2*n].reshape(n, 2)
    r = x[2*n:]
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((c[i] - c[j])**2))
            if d < (r[i] + r[j]) - tol:
                return False
    for i in range(n):
        cx, cy = c[i]
        ri = r[i]
        if cx - ri < -tol or cx + ri > 1 + tol or cy - ri < -tol or cy + ri > 1 + tol:
            return False
    return True


# ---------------------------------------------------------------------------
# Starting points
# ---------------------------------------------------------------------------

def make_random_x0(seed):
    """Fully random center layout."""
    np.random.seed(seed)
    centers = np.random.rand(N, 2) * 0.8 + 0.1
    radii = np.full(N, 0.07)
    return np.concatenate([centers.flatten(), radii])


# ---------------------------------------------------------------------------
# Phase 1: random screening (plain SLSQP, no basinhopping)
# ---------------------------------------------------------------------------

def phase1_worker(args):
    seed, = args
    x0 = make_random_x0(seed)
    bounds = BOUNDS
    res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                   constraints=CONSTRAINTS, options=SOLVER_OPTS)
    score = float(-res.fun)
    feasible = check_feasible(res.x)
    return seed, score, feasible, res.x


def run_phase1(budget, base_seed, workers):
    print(f"Phase 1: screening {budget} random seeds "
          f"(seeds {base_seed}–{base_seed+budget-1}) ...")
    tasks = [(base_seed + i,) for i in range(budget)]
    results = []
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(phase1_worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            seed, score, feasible, x = future.result()
            results.append((score, seed, feasible, x))
            done = len(results)
            feasible_so_far = [r for r in results if r[2]]
            best = max((r[0] for r in feasible_so_far), default=float("-inf"))
            tag = "ok" if feasible else "INFEASIBLE"
            print(f"  [{done:4d}/{budget}] seed={seed:6d}  "
                  f"score={score:.6f}  best={best:.6f}  [{tag}]")

    elapsed = time.perf_counter() - t0
    feasible_results = [(s, seed, x) for s, seed, ok, x in results if ok]
    feasible_results.sort(reverse=True)
    print(f"Phase 1 done in {elapsed:.1f}s  |  "
          f"feasible={len(feasible_results)}/{budget}\n")
    return feasible_results


# ---------------------------------------------------------------------------
# Single basin-hopping run
# ---------------------------------------------------------------------------

def run_basinhopping(start_seed, niter, stepsize, T, x0=None):
    if x0 is None:
        x0 = make_random_x0(start_seed)
    step = CenterStep(stepsize, seed=start_seed)

    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": BOUNDS,
        "constraints": CONSTRAINTS,
        "options": SOLVER_OPTS,
    }

    accepted_steps = []  # list of (step_idx, score, x)
    step_counter = [0]

    def callback(x, f, accepted):
        step_counter[0] += 1
        feasible = check_feasible(x)
        if accepted and feasible:
            accepted_steps.append((step_counter[0], float(-f), x.copy()))

    t0 = time.perf_counter()
    result = basinhopping(
        objective,
        x0,
        niter=niter,
        T=T,
        minimizer_kwargs=minimizer_kwargs,
        take_step=step,
        callback=callback,
        seed=start_seed,
    )
    elapsed = time.perf_counter() - t0

    score = float(-result.fun)
    feasible = check_feasible(result.x)
    return score, feasible, accepted_steps, elapsed, result.x


# ---------------------------------------------------------------------------
# Worker (top-level for multiprocessing)
# ---------------------------------------------------------------------------

def worker(args):
    start_seed, niter, stepsize, T, x0 = args
    score, feasible, accepted_steps, elapsed, x = run_basinhopping(
        start_seed, niter, stepsize, T, x0=x0
    )
    return start_seed, score, feasible, accepted_steps, elapsed, x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-budget", type=int,   default=100,
                        help="Random seeds to screen in Phase 1 (default: 100)")
    parser.add_argument("--phase1-seed",   type=int,   default=0,
                        help="Base random seed for Phase 1 (default: 0)")
    parser.add_argument("--starts",        type=int,   default=4,
                        help="Top-k Phase 1 winners passed to Phase 2 (default: 4)")
    parser.add_argument("--niter",         type=int,   default=100,
                        help="Basin-hopping steps per start in Phase 2 (default: 100)")
    parser.add_argument("--stepsize",      type=float, default=0.04,
                        help="Perturbation magnitude (default: 0.04)")
    parser.add_argument("--T",             type=float, default=0.01,
                        help="Acceptance temperature (default: 0.01)")
    parser.add_argument("--workers",       type=int,   default=4,
                        help="Parallel workers for both phases (default: 4)")
    parser.add_argument("--phase2-seed",   type=int,   default=0,
                        help="Offset added to each Phase 2 start seed (default: 0)")
    args = parser.parse_args()

    total_opts = args.phase1_budget + args.starts * args.niter
    print(f"Two-phase basin-hopping")
    print(f"  Phase 1 : {args.phase1_budget} random seeds screened  "
          f"(base seed={args.phase1_seed})")
    print(f"  Phase 2 : top {args.starts} winners x {args.niter} BH steps  "
          f"(stepsize={args.stepsize}, T={args.T}, seed offset={args.phase2_seed})")
    print(f"  Total local optimizations: ~{total_opts}  |  workers={args.workers}\n")

    # Phase 1
    phase1_results = run_phase1(args.phase1_budget, args.phase1_seed, args.workers)
    if not phase1_results:
        print("Phase 1 found no feasible solutions. Exiting.")
        return

    winners = phase1_results[:args.starts]
    print(f"Phase 2 starting points (top {len(winners)} from Phase 1):")
    for rank, (score, seed, _) in enumerate(winners, 1):
        print(f"  #{rank}  score={score:.6f}  seed={seed}")
    print()

    tasks = [(seed + args.phase2_seed, args.niter, args.stepsize, args.T, x0)
             for _, seed, x0 in winners]

    print(f"Phase 2: basin-hopping from {len(winners)} winners ...")
    all_results = []
    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, t): t[1] for t in tasks}  # key by seed
        for future in as_completed(futures):
            seed, score, feasible, accepted_steps, elapsed, x = future.result()
            all_results.append((score, seed, feasible, accepted_steps, x))
            done = len(all_results)
            tag = "ok" if feasible else "INFEASIBLE"
            best_accepted = max(s for _, s, _ in accepted_steps) if accepted_steps else float("nan")
            print(f"  [{done}/{len(winners)}] seed={seed}  "
                  f"final={score:.6f}  best_accepted={best_accepted:.6f}  "
                  f"accepted={len(accepted_steps)}  [{tag}]  ({elapsed:.1f}s)")

    total_time = time.perf_counter() - t_start
    print(f"Phase 2 done in {total_time:.1f}s\n")
    feasible = [(s, seed, steps, x) for s, seed, ok, steps, x in all_results if ok]
    feasible.sort(reverse=True)

    print("\n" + "="*60)
    print(f"Done in {total_time:.1f}s")
    print(f"Baseline (best_program.py) : 2.626891")
    print(f"Best from perturbed search : 2.635983\n")

    if not feasible:
        print("No feasible solutions found.")
        return

    print(f"Best feasible score: {feasible[0][0]:.6f}  (start_seed={feasible[0][1]})")
    for rank, (score, seed, accepted_steps, x) in enumerate(feasible, 1):
        delta = score - 2.626891
        sign = "+" if delta >= 0 else ""
        n_accepted = len(accepted_steps)
        if accepted_steps:
            best_step, best_mid, best_x = max(accepted_steps, key=lambda t: t[1])
        else:
            best_step, best_mid, best_x = None, float("nan"), None
        print(f"  #{rank}  score={score:.6f}  ({sign}{delta:.6f})"
              f"  phase2_seed={seed}  accepted={n_accepted}  best_mid={best_mid:.6f}"
              + (f"  @step={best_step}" if best_step is not None else ""))


if __name__ == "__main__":
    main()
