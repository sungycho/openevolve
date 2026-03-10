"""
Sensitivity analysis: how does the initial placement (x0) affect the final score?

All solver settings are fixed to match best_program.py:
  method=SLSQP, maxiter=1000, ftol=1e-8

We vary only the initial center layout and initial radius range.
"""

import numpy as np
from scipy.optimize import minimize

N = 26
SOLVER_OPTS = {"maxiter": 1000, "ftol": 1e-8}


# ---------------------------------------------------------------------------
# Solver (fixed)
# ---------------------------------------------------------------------------

def solve(x0):
    n = N

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

    bounds = [(0, 1)] * (2*n) + [(0.03, 0.2)] * n
    res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                   constraints={"type": "ineq", "fun": constraint},
                   options=SOLVER_OPTS)
    return -res.fun, res.success


def is_feasible(x0, score):
    """Re-check constraints on the result (solver success != truly feasible)."""
    return score > 0  # basic sanity; real check would re-run constraint()


# ---------------------------------------------------------------------------
# x0 strategies (vary only the initial placement)
# ---------------------------------------------------------------------------

def x0_best_program():
    """Exact replication of best_program.py initial guess."""
    n = N
    radii = np.linspace(0.12, 0.05, n)
    centers = np.zeros((n, 2))
    grid_x = int(np.sqrt(n))   # 5
    grid_y = int(n / grid_x)   # 5
    xs = np.linspace(0.15, 0.85, grid_x)
    ys = np.linspace(0.15, 0.85, grid_y)
    count = 0
    for i in range(grid_x):
        for j in range(grid_y):
            if count < n:
                centers[count] = [xs[i] + 0.05 * (j % 2), ys[j]]
                count += 1
    np.random.seed(0)
    while count < n:
        centers[count] = np.random.rand(2) * 0.7 + 0.15
        count += 1
    return np.concatenate([centers.flatten(), radii])


def x0_uniform_radii(r0):
    """Same grid layout as best_program.py but all radii = r0."""
    n = N
    radii = np.full(n, r0)
    centers = np.zeros((n, 2))
    grid_x = int(np.sqrt(n))
    grid_y = int(n / grid_x)
    xs = np.linspace(0.15, 0.85, grid_x)
    ys = np.linspace(0.15, 0.85, grid_y)
    count = 0
    for i in range(grid_x):
        for j in range(grid_y):
            if count < n:
                centers[count] = [xs[i] + 0.05 * (j % 2), ys[j]]
                count += 1
    np.random.seed(0)
    while count < n:
        centers[count] = np.random.rand(2) * 0.7 + 0.15
        count += 1
    return np.concatenate([centers.flatten(), radii])


def x0_linspace_radii(r_max, r_min):
    """Same grid layout, but vary the linspace radius range."""
    n = N
    radii = np.linspace(r_max, r_min, n)
    centers = np.zeros((n, 2))
    grid_x = int(np.sqrt(n))
    grid_y = int(n / grid_x)
    xs = np.linspace(0.15, 0.85, grid_x)
    ys = np.linspace(0.15, 0.85, grid_y)
    count = 0
    for i in range(grid_x):
        for j in range(grid_y):
            if count < n:
                centers[count] = [xs[i] + 0.05 * (j % 2), ys[j]]
                count += 1
    np.random.seed(0)
    while count < n:
        centers[count] = np.random.rand(2) * 0.7 + 0.15
        count += 1
    return np.concatenate([centers.flatten(), radii])


def x0_random(seed):
    """Fully random initial placement."""
    np.random.seed(seed)
    n = N
    centers = np.random.rand(n, 2) * 0.7 + 0.15
    radii = np.full(n, 0.07)
    return np.concatenate([centers.flatten(), radii])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(label, x0):
    score, success = solve(x0)
    status = "ok" if success else "no-converge"
    print(f"  {label:45s}  score={score:.6f}  [{status}]")
    return score


if __name__ == "__main__":
    print("=== Baseline: exact best_program.py x0 ===")
    run("linspace(0.12→0.05) + offset grid  [baseline]", x0_best_program())

    print("\n=== Vary initial radius (uniform, same grid layout) ===")
    for r0 in [0.03, 0.05, 0.07, 0.09, 0.11, 0.13]:
        run(f"uniform r0={r0}", x0_uniform_radii(r0))

    print("\n=== Vary linspace range (same grid layout) ===")
    for r_max, r_min in [
        (0.15, 0.08), (0.12, 0.05), (0.10, 0.05), (0.10, 0.03),
        (0.08, 0.03), (0.07, 0.07),  # last one = uniform
    ]:
        run(f"linspace({r_max:.2f}→{r_min:.2f})", x0_linspace_radii(r_max, r_min))

    print("\n=== Vary random seed (fully random placement) ===")
    for seed in range(10):
        run(f"random seed={seed}", x0_random(seed))
