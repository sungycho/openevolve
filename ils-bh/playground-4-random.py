"""
Multi-start search using fully random center layouts for circle packing (n=26).

Runs SLSQP from many random starting center positions in parallel.
Use --start-seed to continue a previous run from the next seed.

Usage:
    python playground-4-random.py --budget 100
    python playground-4-random.py --budget 500 --start-seed 100
    python playground-4-random.py --budget 500 --workers 8
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import minimize

N = 26
FIXED_RADIUS = 0.07
SOLVER_OPTS = {"maxiter": 1000, "ftol": 1e-8}


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve(centers_flat):
    n = N
    radii = np.full(n, FIXED_RADIUS)
    x0 = np.concatenate([centers_flat, radii])
    bounds = [(0, 1)] * (2 * n) + [(0.03, 0.2)] * n

    def objective(x):
        return -np.sum(x[2*n:])

    def constraint(x):
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

    res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                   constraints={"type": "ineq", "fun": constraint},
                   options=SOLVER_OPTS)

    opt_centers = res.x[:2*n].reshape(n, 2)
    opt_radii = res.x[2*n:]
    return float(-res.fun), opt_centers, opt_radii


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

def check_feasible(centers, radii, tol=1e-9):
    n = len(radii)
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < (radii[i] + radii[j]) - tol:
                return False
    for i in range(n):
        cx, cy = centers[i]
        r = radii[i]
        if cx - r < -tol or cx + r > 1 + tol or cy - r < -tol or cy + r > 1 + tol:
            return False
    return True


# ---------------------------------------------------------------------------
# Random layout
# ---------------------------------------------------------------------------

def make_random_layout(seed):
    np.random.seed(seed)
    centers = np.random.rand(N, 2) * 0.8 + 0.1
    return centers.flatten()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(args):
    seed, centers_flat = args
    t0 = time.perf_counter()
    score, centers, radii = solve(centers_flat)
    feasible = check_feasible(centers, radii)
    elapsed = time.perf_counter() - t0
    return seed, score, feasible, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=100,
                        help="Number of random seeds to try (default: 100)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers (default: 10)")
    parser.add_argument("--start-seed", type=int, default=0,
                        help="Starting seed, to continue a previous run (default: 0)")
    args = parser.parse_args()

    seeds = range(args.start_seed, args.start_seed + args.budget)
    total = args.budget

    print(f"Budget: {total} starts | Workers: {args.workers} | "
          f"Seeds: {args.start_seed}–{args.start_seed + total - 1}\n")

    tasks = [(s, make_random_layout(s)) for s in seeds]
    results = []
    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            seed, score, feasible, elapsed = future.result()
            results.append((score, seed, feasible))
            done = len(results)
            feasible_results = [r for r in results if r[2]]
            best = max((r[0] for r in feasible_results), default=float("-inf"))
            tag = "ok" if feasible else "INFEASIBLE"
            print(f"[{done:4d}/{total}] seed={seed:6d}  score={score:.6f}"
                  f"  best={best:.6f}  [{tag}]  ({elapsed:.1f}s)")

    total_time = time.perf_counter() - t_start
    feasible_results = [(s, seed) for s, seed, ok in results if ok]
    infeasible_count = len(results) - len(feasible_results)
    feasible_results.sort(reverse=True)

    print("\n" + "="*60)
    print(f"Done in {total_time:.1f}s  |  "
          f"Feasible: {len(feasible_results)}/{total}  |  "
          f"Infeasible (discarded): {infeasible_count}")
    print(f"Baseline (best_program.py): 2.626891\n")

    if not feasible_results:
        print("No feasible solutions found.")
        return

    print(f"Best feasible score: {feasible_results[0][0]:.6f}  (seed={feasible_results[0][1]})\n")
    print("Top 10 (feasible only):")
    for rank, (score, seed) in enumerate(feasible_results[:10], 1):
        delta = score - 2.626891
        sign = "+" if delta >= 0 else ""
        print(f"  #{rank:2d}  score={score:.6f}  ({sign}{delta:.6f})  seed={seed}")

    scores = [s for s, _ in feasible_results]
    print(f"\nStats over {len(scores)} feasible runs:")
    print(f"  mean={sum(scores)/len(scores):.6f}  "
          f"max={max(scores):.6f}  min={min(scores):.6f}")


if __name__ == "__main__":
    main()
