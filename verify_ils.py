"""
Quick verification script: run evaluator on baseline vs ILS best program.
Usage: python verify_ils.py <ils_output_dir> <evaluator_path>

Example:
  python verify_ils.py ils_output/signal_processing_20260312_104314 examples/signal_processing/evaluator.py
"""

import importlib.util
import json
import sys
from pathlib import Path


def load_evaluator(evaluator_path):
    spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def evaluate(evaluator, program_path):
    return evaluator.evaluate(str(program_path))


def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_ils.py <ils_output_dir> <evaluator_path>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    evaluator_path = sys.argv[2]

    evaluator = load_evaluator(evaluator_path)

    # Load summary to see what ILS claimed
    summary_file = run_dir / "results_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        print("=== ILS Claimed Results ===")
        print(f"Baseline score (as reported): {summary['baseline_score']}")
        for s in summary["strategies"]:
            print(f"  {s['name']}: best={s['best_score']}, improvement={s['improvement']}")
        print()

    # Find programs to evaluate
    candidates = {}

    # Baseline: look for best_program.py in the openevolve output path
    # (the original program before ILS)
    baseline_path = run_dir / "strategies"
    ils_best = run_dir / "best_program.py"

    # Also check per-strategy best programs in _runner dirs
    runner_dirs = sorted(run_dir.glob("_runner_prog*"))

    print("=== Re-evaluating Programs ===\n")

    # Evaluate ILS best
    if ils_best.exists():
        print(f"Evaluating ILS best: {ils_best}")
        metrics = evaluate(evaluator, ils_best)
        print(f"  Metrics: {json.dumps(metrics, indent=4)}")
        print()

    # Evaluate per-strategy bests
    for runner_dir in runner_dirs:
        for strategy_best in sorted(runner_dir.glob("*_best.py")):
            print(f"Evaluating {strategy_best.relative_to(run_dir)}:")
            try:
                metrics = evaluate(evaluator, strategy_best)
                print(f"  Metrics: {json.dumps(metrics, indent=4)}")
            except Exception as e:
                print(f"  ERROR: {e}")
            print()


if __name__ == "__main__":
    main()
