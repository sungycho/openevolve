"""
Step 5: Format and save ILS results
"""

import json
import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _task_name_from_evaluator(evaluator_path: str) -> str:
    """Derive task name from evaluator path (e.g. examples/circle_packing/evaluator.py → circle_packing)."""
    path = Path(evaluator_path).resolve()
    return path.parent.name


def print_results(
    results: List,
    baseline_score: float,
    task_name: str = "task",
) -> None:
    """Print clean terminal summary of ILS results."""
    print()
    print("ILS Engine Results")
    print("==================")
    print(f"Baseline (OpenEvolve best): {baseline_score:.6f}")
    print()

    header = f"{'Strategy':<22} | {'Seeds':>5} | {'Best Score':>10} | {'Improvement':>11} | {'Runtime':>8}"
    print(header)
    print("-" * len(header))

    overall_best_score = baseline_score
    overall_best_strategy = None
    overall_best_seed = None

    for r in results:
        if r.best_score == -float("inf"):
            score_str = "   FAILED"
            improvement = "         -"
        else:
            score_str = f"{r.best_score:10.6f}"
            delta = r.best_score - baseline_score
            improvement = f"{delta:+11.6f}"
            if r.best_score > overall_best_score:
                overall_best_score = r.best_score
                overall_best_strategy = r.strategy_name
                overall_best_seed = r.best_seed

        seeds_count = len(r.all_seed_scores)
        runtime_str = f"{r.runtime_s:7.1f}s"
        print(
            f"{r.strategy_name:<22} | {seeds_count:>5} | {score_str} | {improvement} | {runtime_str}"
        )

    print()
    if overall_best_strategy:
        print(
            f"Overall best: {overall_best_score:.6f} "
            f"({overall_best_strategy}, seed={overall_best_seed})"
        )
    else:
        print(f"No improvement found over baseline {baseline_score:.6f}")
    print()


def save_results(
    results: List,
    baseline_score: float,
    baseline_program_code: str,
    run_dir: str,
) -> str:
    """
    Save all ILS results to disk.

    Returns:
        Path to the task output directory
    """
    task_dir = run_dir
    task_name = Path(run_dir).name
    strategies_dir = os.path.join(task_dir, "strategies")
    os.makedirs(strategies_dir, exist_ok=True)

    # Find overall best
    overall_best = None
    for r in results:
        if r.best_score > -float("inf") and r.best_program_code:
            if overall_best is None or r.best_score > overall_best.best_score:
                overall_best = r

    # Save best program
    if overall_best and overall_best.best_score > baseline_score:
        best_code = overall_best.best_program_code
        best_info = {
            "score": overall_best.best_score,
            "strategy": overall_best.strategy_name,
            "seed": overall_best.best_seed,
            "runtime_s": overall_best.runtime_s,
            "baseline_score": baseline_score,
            "improvement": overall_best.best_score - baseline_score,
        }
    else:
        # No improvement — save baseline
        best_code = baseline_program_code
        best_info = {
            "score": baseline_score,
            "strategy": "baseline",
            "seed": None,
            "runtime_s": 0.0,
            "baseline_score": baseline_score,
            "improvement": 0.0,
        }

    with open(os.path.join(task_dir, "best_program.py"), "w") as f:
        f.write(best_code)
    with open(os.path.join(task_dir, "best_program_info.json"), "w") as f:
        json.dump(best_info, f, indent=2)

    # Save results summary
    summary = {
        "task": task_name,
        "baseline_score": baseline_score,
        "strategies": [
            {
                "name": r.strategy_name,
                "best_score": r.best_score if r.best_score > -float("inf") else None,
                "best_seed": r.best_seed,
                "improvement": (r.best_score - baseline_score) if r.best_score > -float("inf") else None,
                "seeds_run": len(r.all_seed_scores),
                "runtime_s": r.runtime_s,
                "error": r.error,
            }
            for r in results
        ],
        "overall_best": best_info,
    }

    with open(os.path.join(task_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-strategy results
    for r in results:
        strategy_data = {
            "strategy_name": r.strategy_name,
            "best_score": r.best_score if r.best_score > -float("inf") else None,
            "best_seed": r.best_seed,
            "runtime_s": r.runtime_s,
            "error": r.error,
            "all_seed_scores": r.all_seed_scores,
        }
        strategy_file = os.path.join(strategies_dir, f"{r.strategy_name}_results.json")
        with open(strategy_file, "w") as f:
            json.dump(strategy_data, f, indent=2)

    logger.info(f"Results saved to {task_dir}")
    return task_dir
