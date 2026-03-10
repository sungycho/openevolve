#!/usr/bin/env python3
"""
Compare score vs iteration between two OpenEvolve runs (e.g., with GCA vs without GCA).

The plot includes, for each run:
1) Best score in that iteration
2) Running global best score up to that iteration

Edit the CONFIG section below, then run:
    python visualization_scores.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

# ----------------------------- CONFIG ---------------------------------------
# Base directory of this script (used for default run paths)
BASE_DIR = Path(__file__).resolve().parent

# Run folder with GCA enabled (contains checkpoints/checkpoint_*/)
WITH_GCA_RUN_DIR = BASE_DIR / "runs/with_gca"

# Run folder with GCA disabled (contains checkpoints/checkpoint_*/)
WITHOUT_GCA_RUN_DIR = BASE_DIR / "runs/without_gca"

# Metric key from program metrics to plot
METRIC = "combined_score"

# Plot title
PLOT_TITLE = "GCA vs No-GCA: Score by Iteration"

# If None, show interactive plot. If path, save image there.
OUTPUT_PATH: Path | None = None
# ---------------------------------------------------------------------------


def latest_checkpoint_dir(run_dir: Path) -> Path:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints = [p for p in checkpoints_dir.glob("checkpoint_*") if p.is_dir()]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under: {checkpoints_dir}")
    return sorted(checkpoints, key=lambda p: int(p.name.split("_")[-1]))[-1]


def load_best_per_iteration(checkpoint_dir: Path, metric: str) -> dict[int, float]:
    programs_dir = checkpoint_dir / "programs"
    if not programs_dir.exists():
        raise FileNotFoundError(f"Missing programs directory: {programs_dir}")

    best_by_iter: dict[int, float] = {}
    for program_file in programs_dir.glob("*.json"):
        try:
            data = json.loads(program_file.read_text())
        except Exception:
            continue

        iteration = data.get("iteration_found")
        metrics = data.get("metrics") or {}
        score = metrics.get(metric)
        if iteration is None or not isinstance(score, (int, float)):
            continue

        i = int(iteration)
        s = float(score)
        best_by_iter[i] = max(s, best_by_iter.get(i, float("-inf")))

    return best_by_iter


def build_series(best_by_iter: dict[int, float]) -> tuple[list[int], list[float], list[float]]:
    if not best_by_iter:
        return [], [], []

    xs = sorted(best_by_iter)
    per_iter_best: list[float] = []
    running_best: list[float] = []
    current = float("-inf")

    for i in xs:
        val = best_by_iter[i]
        per_iter_best.append(val)
        current = max(current, val)
        running_best.append(current)

    return xs, per_iter_best, running_best


def add_condition_lines(
    label: str,
    run_dir: Path,
    color: str,
) -> None:
    checkpoint_dir = latest_checkpoint_dir(run_dir)
    best_by_iter = load_best_per_iteration(checkpoint_dir, METRIC)
    if not best_by_iter:
        raise RuntimeError(
            f"No numeric metric '{METRIC}' found in programs under {checkpoint_dir / 'programs'}"
        )

    xs, per_iter_best, running_best = build_series(best_by_iter)

    plt.plot(
        xs,
        per_iter_best,
        color=color,
        linestyle="--",
        linewidth=1.7,
        alpha=0.75,
        label=f"{label} - best in iteration",
    )
    plt.plot(
        xs,
        running_best,
        color=color,
        linestyle="-",
        linewidth=2.3,
        label=f"{label} - global best",
    )

    print(
        f"{label}: last_iter={xs[-1]}, "
        f"best_in_iteration(max)={max(per_iter_best):.6f}, "
        f"global_best(final)={running_best[-1]:.6f}"
    )


def main() -> None:
    plt.figure(figsize=(11, 5.5))
    add_condition_lines("With GCA", WITH_GCA_RUN_DIR, "tab:red")
    add_condition_lines("Without GCA", WITHOUT_GCA_RUN_DIR, "tab:blue")

    plt.xlabel("Iteration")
    plt.ylabel(METRIC)
    plt.title(f"{PLOT_TITLE} ({METRIC})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if OUTPUT_PATH:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(OUTPUT_PATH, dpi=150)
        print(f"Saved plot to: {OUTPUT_PATH}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
