"""
Multi-start search over center layouts for circle packing (n=26).

Radii are fixed (proven not to matter). Only center layout varies.
Runs SLSQP from many different starting layouts in parallel.

Usage:
    python playground-3.py --budget 100
    python playground-3.py --budget 500 --workers 8
"""

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import minimize

N = 26
FIXED_RADIUS = 0.07
SOLVER_OPTS = {"maxiter": 1000, "ftol": 1e-8}


# ---------------------------------------------------------------------------
# Solver (identical to best_program.py settings)
# ---------------------------------------------------------------------------

def solve(centers_flat):
    """Run SLSQP from given center layout. Returns (score, centers, radii)."""
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
# Center layout generators
# (all return flat array of shape (2*N,))
# ---------------------------------------------------------------------------

def layout_random(seed):
    np.random.seed(seed)
    centers = np.random.rand(N, 2) * 0.8 + 0.1
    return centers.flatten(), f"random(seed={seed})"


def layout_grid(n_cols, row_offset):
    """Uniform grid with configurable columns and alternating row offset."""
    n = N
    n_rows = math.ceil(n / n_cols)
    xs = np.linspace(0.1, 0.9, n_cols)
    ys = np.linspace(0.1, 0.9, n_rows)
    pts = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            if len(pts) < n:
                offset = row_offset if row % 2 == 1 else 0.0
                pts.append([x + offset, y])
    pts = np.array(pts[:n])
    pts = np.clip(pts, 0.05, 0.95)
    return pts.flatten(), f"grid(cols={n_cols}, offset={row_offset:.2f})"


def layout_hexagonal(n_cols, row_height_scale):
    """Hexagonal close-packing with configurable density."""
    n = N
    row_h = (math.sqrt(3) / 2) * row_height_scale
    pts = []
    row = 0
    while len(pts) < n:
        offset = 0.5 / n_cols if row % 2 == 1 else 0.0
        for col in range(n_cols):
            if len(pts) < n:
                x = (col + 0.5) / n_cols + offset
                y = (row + 0.5) * row_h
                if 0 < x < 1 and 0 < y < 1:
                    pts.append([x, y])
        row += 1
        if row > 30:
            break
    if len(pts) < n:
        # fill remainder randomly
        np.random.seed(42)
        while len(pts) < n:
            pts.append(list(np.random.rand(2) * 0.8 + 0.1))
    pts = np.array(pts[:n])
    pts = np.clip(pts, 0.05, 0.95)
    return pts.flatten(), f"hex(cols={n_cols}, row_h={row_height_scale:.2f})"


def layout_perturbed_grid(noise_sigma, seed):
    """Best grid layout + Gaussian noise to escape the local optimum."""
    # replicate best_program.py grid
    n = N
    grid_x = int(math.sqrt(n))   # 5
    grid_y = int(n / grid_x)     # 5
    xs = np.linspace(0.15, 0.85, grid_x)
    ys = np.linspace(0.15, 0.85, grid_y)
    pts = []
    for i in range(grid_x):
        for j in range(grid_y):
            if len(pts) < n:
                pts.append([xs[i] + 0.05 * (j % 2), ys[j]])
    np.random.seed(seed)
    while len(pts) < n:
        pts.append(list(np.random.rand(2) * 0.7 + 0.15))
    centers = np.array(pts[:n])
    centers += np.random.randn(n, 2) * noise_sigma
    centers = np.clip(centers, 0.05, 0.95)
    return centers.flatten(), f"perturbed_grid(sigma={noise_sigma:.3f}, seed={seed})"


def generate_layouts(budget, mode="full", start_seed=0):
    """
    Distribute `budget` starts across layout types.

    mode="full"      — 40% random, 20% grid, 20% hex, 20% perturbed
    mode="perturbed" — all budget on perturbed grid, sweeping sigma in
                       the effective range [0.01, 0.08]
    """
    layouts = []

    if mode == "perturbed":
        noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        n_sigmas = len(noise_levels)
        for i in range(budget):
            sigma = noise_levels[i % n_sigmas]
            seed  = start_seed + i // n_sigmas
            layouts.append(layout_perturbed_grid(sigma, seed))
        return layouts

    # mode == "full"
    n_random    = int(budget * 0.40)
    n_grid      = int(budget * 0.20)
    n_hex       = int(budget * 0.20)
    n_perturbed = budget - n_random - n_grid - n_hex

    for seed in range(n_random):
        layouts.append(layout_random(seed))

    grid_params = [(c, o) for c in [4, 5, 6, 7]
                          for o in [0.0, 0.05, 0.1, 0.15]]
    for i in range(n_grid):
        c, o = grid_params[i % len(grid_params)]
        layouts.append(layout_grid(c, o))

    hex_params = [(c, s) for c in [4, 5, 6, 7, 8]
                         for s in [0.80, 0.90, 1.00, 1.05]]
    for i in range(n_hex):
        c, s = hex_params[i % len(hex_params)]
        layouts.append(layout_hexagonal(c, s))

    noise_levels = [0.02, 0.05, 0.10, 0.15, 0.20]
    for i in range(n_perturbed):
        sigma = noise_levels[i % len(noise_levels)]
        seed  = i // len(noise_levels)
        layouts.append(layout_perturbed_grid(sigma, seed))

    return layouts


# ---------------------------------------------------------------------------
# Worker (must be top-level for multiprocessing)
# ---------------------------------------------------------------------------

def check_feasible(centers, radii, tol=1e-6):
    """Hard feasibility check independent of solver tolerance."""
    n = len(radii)
    # pairwise no-overlap
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < (radii[i] + radii[j]) - tol:
                return False
    # boundary
    for i in range(n):
        if (centers[i, 0] - radii[i] < -tol or
            centers[i, 0] + radii[i] > 1 + tol or
            centers[i, 1] - radii[i] < -tol or
            centers[i, 1] + radii[i] > 1 + tol):
            return False
    return True


def worker(args):
    idx, centers_flat, label = args
    t0 = time.perf_counter()
    score, centers, radii = solve(centers_flat)
    feasible = check_feasible(centers, radii)
    elapsed = time.perf_counter() - t0
    return idx, score, label, elapsed, feasible


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget",  type=int, default=100,
                        help="Total number of starts (default: 100)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers (default: 10)")
    parser.add_argument("--mode", choices=["full", "perturbed"], default="full",
                        help="Layout mode: full (all types) or perturbed (sigma sweep only)")
    parser.add_argument("--start-seed", type=int, default=0,
                        help="Seed offset for perturbed mode, to continue a previous run (default: 0)")
    args = parser.parse_args()

    layouts = generate_layouts(args.budget, mode=args.mode, start_seed=args.start_seed)
    total = len(layouts)

    print(f"Budget: {total} starts | Workers: {args.workers} | Mode: {args.mode}")
    if args.mode == "full":
        print(f"Layout split: "
              f"{int(total*0.4)} random, "
              f"{int(total*0.2)} grid, "
              f"{int(total*0.2)} hex, "
              f"{total - int(total*0.4) - int(total*0.2) - int(total*0.2)} perturbed")
    else:
        print(f"Sigma sweep: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]"
              f"  x  {total // 8} seeds each")
    print()

    tasks = [(i, centers_flat, label) for i, (centers_flat, label) in enumerate(layouts)]

    results = []
    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            idx, score, label, elapsed, feasible = future.result()
            results.append((score, label, feasible))
            done = len(results)
            best_so_far = max(r[0] for r in results if r[2])  # feasible only
            tag = "ok" if feasible else "INFEASIBLE"
            print(f"[{done:3d}/{total}] score={score:.6f}  best={best_so_far:.6f}"
                  f"  [{tag}]  {label}  ({elapsed:.1f}s)")

    total_time = time.perf_counter() - t_start

    # Summary
    feasible_results = [(s, l) for s, l, ok in results if ok]
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

    print(f"Best feasible score: {feasible_results[0][0]:.6f}\n")
    print("Top 10 (feasible only):")
    for rank, (score, label) in enumerate(feasible_results[:10], 1):
        delta = score - 2.626891
        sign = "+" if delta >= 0 else ""
        print(f"  #{rank:2d}  score={score:.6f}  ({sign}{delta:.6f})  {label}")

    if args.mode == "full":
        print("\nBy layout type (feasible only):")
        for layout_type in ["random", "grid", "hex", "perturbed"]:
            subset = [s for s, l in feasible_results if layout_type in l]
            if subset:
                print(f"  {layout_type:12s}  n={len(subset):3d}"
                      f"  max={max(subset):.6f}"
                      f"  mean={sum(subset)/len(subset):.6f}"
                      f"  min={min(subset):.6f}")
    else:
        print("\nBy sigma (feasible only):")
        for sigma in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
            subset = [s for s, l in feasible_results if f"sigma={sigma:.3f}" in l]
            if subset:
                print(f"  sigma={sigma:.2f}  n={len(subset):3d}"
                      f"  max={max(subset):.6f}"
                      f"  mean={sum(subset)/len(subset):.6f}"
                      f"  min={min(subset):.6f}")


if __name__ == "__main__":
    main()
