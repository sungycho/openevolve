#!/usr/bin/env python
"""
CLI entry point for the ILS Engine — post-processing for OpenEvolve output.

Usage:
  python openevolve-ils.py <openevolve_output_path> <evaluator_path> [options]

Example:
  python openevolve-ils.py \\
    examples/circle_packing/openevolve_output \\
    examples/circle_packing/evaluator.py \\
    --top-k 3 \\
    --selection-mode novelty_rejection \\
    --total-seeds 150 \\
    --workers 8 \\
    --problem-description "Pack 26 circles of equal radius in a unit square, maximize sum of radii"
"""

import argparse
import logging
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="ILS Engine: Iterated Local Search post-processing for OpenEvolve output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required positional args
    parser.add_argument(
        "openevolve_output_path",
        help="Path to OpenEvolve output (checkpoint directory or best/ folder)",
    )
    parser.add_argument(
        "evaluator_path",
        help="Path to the evaluator.py script",
    )

    # Program selection
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of programs to run ILS on (default: 1)",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["top_1", "vanilla_top_k", "novelty_rejection"],
        default="novelty_rejection",
        help="Program selection mode (default: novelty_rejection)",
    )
    parser.add_argument(
        "--novelty-threshold",
        type=float,
        default=0.3,
        help="Minimum normalized edit distance for novelty rejection (default: 0.3)",
    )

    # Problem context
    parser.add_argument(
        "--problem-description",
        default="",
        help="Natural language description of the problem (helps LLM understand domain)",
    )
    parser.add_argument(
        "--score-metric",
        default="combined_score",
        help="Metric key from evaluator output to use as the objective score "
             "(default: combined_score). Use the raw metric for absolute scores "
             "e.g. --score-metric sum_radii for circle packing",
    )

    # LLM
    parser.add_argument(
        "--llm-model",
        default="anthropic/claude-sonnet-4-6",
        help="Fallback OpenRouter model for all steps (default: anthropic/claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--analyzer-model",
        default=None,
        help="Model for Step 1 (ILS analysis). Defaults to --llm-model",
    )
    parser.add_argument(
        "--planner-model",
        default=None,
        help="Model for Step 2 (strategy planning). Defaults to --llm-model",
    )
    parser.add_argument(
        "--generator-model",
        default=None,
        help="Model for Step 3 (code generation). Defaults to --llm-model. "
             "Recommend: anthropic/claude-opus-4-6 or google/gemini-2.5-pro",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API key (defaults to OPENROUTER_API_KEY environment variable)",
    )

    # Strategies
    parser.add_argument(
        "--total-seeds",
        type=int,
        default=150,
        help="Total seed budget split across strategies (default: 150)",
    )
    parser.add_argument(
        "--no-tier2",
        action="store_true",
        help="Disable domain-specific (Tier 2) strategies",
    )
    parser.add_argument(
        "--max-tier2",
        type=int,
        default=2,
        help="Maximum number of Tier 2 strategies (default: 2)",
    )

    # Execution
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers per strategy script (default: 8)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per strategy in seconds (default: 300)",
    )

    # Two-phase search
    parser.add_argument(
        "--phase1-budget",
        type=int,
        default=None,
        help="Phase 1 candidates to screen (default: let script decide, ~total_seeds/4)",
    )
    parser.add_argument(
        "--phase1-top-k",
        type=int,
        default=4,
        help="Phase 1 winners passed to Phase 2 basin-hopping (default: 4)",
    )
    parser.add_argument(
        "--phase2-seed",
        type=int,
        default=0,
        help="Seed offset for Phase 2 chains (default: 0)",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=100,
        help="Basin-hopping steps per Phase 2 chain (default: 100)",
    )
    parser.add_argument(
        "--stepsize",
        type=float,
        default=0.04,
        help="Perturbation magnitude for Phase 2 (default: 0.04)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="ils_output",
        help="Base output directory (default: ils_output)",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build config
    from ils_engine.config import ILSConfig
    from ils_engine.engine import ILSEngine

    config = ILSConfig(
        openevolve_output_path=args.openevolve_output_path,
        evaluator_path=args.evaluator_path,
        problem_description=args.problem_description,
        score_metric=args.score_metric,
        top_k=args.top_k,
        selection_mode=args.selection_mode,
        novelty_threshold=args.novelty_threshold,
        llm_model=args.llm_model,
        analyzer_model=args.analyzer_model,
        planner_model=args.planner_model,
        generator_model=args.generator_model,
        llm_api_key=args.api_key,
        enable_tier2_strategies=not args.no_tier2,
        max_tier2_strategies=args.max_tier2,
        total_seeds=args.total_seeds,
        num_workers=args.workers,
        phase1_budget=args.phase1_budget,
        phase1_top_k=args.phase1_top_k,
        phase2_seed=args.phase2_seed,
        niter=args.niter,
        stepsize=args.stepsize,
        timeout_per_strategy=args.timeout,
        output_dir=args.output_dir,
    )

    # Validate paths
    if not os.path.exists(config.openevolve_output_path):
        print(f"ERROR: OpenEvolve output path does not exist: {config.openevolve_output_path}")
        sys.exit(1)

    if not os.path.exists(config.evaluator_path):
        print(f"ERROR: Evaluator path does not exist: {config.evaluator_path}")
        sys.exit(1)

    # Check API key
    if not config.llm_api_key and not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: No OpenRouter API key found. Set OPENROUTER_API_KEY or use --api-key.")
        sys.exit(1)

    print(f"ILS Engine starting...")
    print(f"  Output path: {config.openevolve_output_path}")
    print(f"  Evaluator:   {config.evaluator_path}")
    print(f"  Mode:        {config.selection_mode} (top-{config.top_k})")
    print(f"  Seeds:       {config.total_seeds}")
    print(f"  Workers:     {config.num_workers}")
    print(f"  Analyzer:    {config.analyzer_model or config.llm_model}")
    print(f"  Planner:     {config.planner_model or config.llm_model}")
    print(f"  Generator:   {config.generator_model or config.llm_model}")
    print()

    engine = ILSEngine(config)
    summary = engine.run()

    print(f"Output saved to: {summary.get('output_dir', config.output_dir)}")

    if summary.get("improvement", 0) > 0:
        print(
            f"Improvement: +{summary['improvement']:.6f} "
            f"({summary['baseline_score']:.6f} → {summary['overall_best_score']:.6f})"
        )
    else:
        print(f"No improvement found over baseline {summary.get('baseline_score', 0):.6f}")


if __name__ == "__main__":
    main()
