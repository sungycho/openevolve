"""
Reproduces the best circle packing found by perturbed grid search.
Setup: perturbed_grid(sigma=0.010, seed=46) -> score=2.635328
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize

N = 26
SIGMA = 0.010
SEED  = 46


def make_x0(sigma, seed, n=N):
    """Perturbed grid: best_program.py layout + Gaussian noise."""
    grid_x = int(np.sqrt(n))   # 5
    grid_y = int(n / grid_x)   # 5
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
    centers += np.random.randn(n, 2) * sigma
    centers = np.clip(centers, 0.05, 0.95)
    radii = np.full(n, 0.07)
    return np.concatenate([centers.flatten(), radii])


def solve(x0, n=N):
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
                   options={"maxiter": 1000, "ftol": 1e-8})

    centers = res.x[:2*n].reshape(n, 2)
    radii   = res.x[2*n:]
    score   = float(-res.fun)
    return centers, radii, score


def plot(centers, radii, score):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=2,
                                    edgecolor="black", facecolor="none"))

    cmap = plt.get_cmap("tab20", N)
    for i, (c, r) in enumerate(zip(centers, radii)):
        circle = plt.Circle(c, r, color=cmap(i), alpha=0.6, linewidth=0.8,
                             edgecolor="black")
        ax.add_patch(circle)
        ax.text(c[0], c[1], str(i), ha="center", va="center",
                fontsize=7, fontweight="bold")

    ax.set_title(
        f"Circle packing  n={N}  |  sum of radii = {score:.6f}\n"
        f"Setup: perturbed_grid(sigma={SIGMA}, seed={SEED})",
        fontsize=11
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(f"circle_packing_{score:.6f}.png", dpi=150)
    print("Saved: circle_packing_best.png")
    plt.show()


def verify(centers, radii, tol=1e-9):
    """Check all problem constraints and report violations beyond tol."""
    n = len(radii)
    violations = []

    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            gap = d - (radii[i] + radii[j])
            if gap < -tol:
                violations.append(f"  OVERLAP   circles {i} & {j}: gap={gap:.2e}")

    for i in range(n):
        cx, cy = centers[i]
        r = radii[i]
        if cx - r < -tol:
            violations.append(f"  BOUNDARY  circle {i}: left edge  x-r={cx-r:.2e}")
        if cx + r > 1 + tol:
            violations.append(f"  BOUNDARY  circle {i}: right edge x+r={cx+r:.2e}")
        if cy - r < -tol:
            violations.append(f"  BOUNDARY  circle {i}: bottom     y-r={cy-r:.2e}")
        if cy + r > 1 + tol:
            violations.append(f"  BOUNDARY  circle {i}: top        y+r={cy+r:.2e}")

    print("\n--- Constraint verification ---")
    if violations:
        print(f"FAILED  ({len(violations)} violation(s)):")
        for v in violations:
            print(v)
    else:
        min_gap = min(
            np.sqrt(np.sum((centers[i] - centers[j])**2)) - (radii[i] + radii[j])
            for i in range(n) for j in range(i+1, n)
        )
        min_boundary = min(
            min(cx - r, 1 - cx - r, cy - r, 1 - cy - r)
            for (cx, cy), r in zip(centers, radii)
        )
        print(f"PASSED  all {n*(n-1)//2} pair checks + {4*n} boundary checks")
        print(f"  Tightest pairwise gap : {min_gap:.2e}")
        print(f"  Tightest boundary gap : {min_boundary:.2e}")


if __name__ == "__main__":
    print(f"Building x0: perturbed_grid(sigma={SIGMA}, seed={SEED})")
    x0 = make_x0(SIGMA, SEED)

    print("Running SLSQP...")
    centers, radii, score = solve(x0)

    print(f"Score: {score:.6f}  (baseline: 2.626891, AlphaEvolve: 2.635)")
    verify(centers, radii)
    plot(centers, radii, score)

