"""
ILSEngine — orchestrates the full ILS pipeline
"""

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ils_engine.config import ILSConfig

logger = logging.getLogger(__name__)


def _evaluate_program(code: str, evaluator_path: str) -> dict:
    """Write code to a temp file and evaluate it using the evaluator script."""
    import importlib.util

    evaluator_path = str(Path(evaluator_path).resolve())
    spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = evaluator.evaluate(tmp_path)
        return result if isinstance(result, dict) else {}
    except Exception as e:
        logger.warning(f"Baseline evaluation failed: {e}")
        return {}
    finally:
        os.unlink(tmp_path)


class ILSEngine:
    """
    Orchestrates the full Iterated Local Search post-processing pipeline:

    Step 0: Select top-K programs from OpenEvolve output
    Step 1: Identify ILS-able components via LLM
    Step 2: Generate search strategies via LLM
    Step 3: Generate runnable search scripts via LLM
    Step 4: Execute scripts and collect results
    Step 5: Report and save results
    """

    def __init__(self, config: ILSConfig):
        self.config = config
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    def run(self) -> dict:
        """
        Run the full ILS pipeline.

        Returns:
            dict with summary of results
        """
        config = self.config
        from ils_engine.reporter import _task_name_from_evaluator
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Known OpenEvolve-internal normalized metrics — never use these as the objective
        _INTERNAL_METRICS = {"combined_score", "target_ratio", "validity", "eval_time", "error"}
        task_name = _task_name_from_evaluator(config.evaluator_path)
        run_dir = os.path.join(config.output_dir, f"{task_name}_{run_timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # ── Step 0: Program selection ────────────────────────────────────────
        logger.info("Step 0: Selecting programs...")
        from ils_engine.program_selector import select_programs

        programs = select_programs(
            output_path=config.openevolve_output_path,
            top_k=config.top_k,
            selection_mode=config.selection_mode,
            novelty_threshold=config.novelty_threshold,
        )

        if not programs:
            raise RuntimeError("No programs found in the specified output path.")

        logger.info(f"Selected {len(programs)} program(s) for ILS")

        # Use the first (best) program as baseline
        best_program = programs[0]

        # If no metrics were stored (e.g. loading from a bare best_program.py without
        # best_program_info.json), evaluate the program now to get real metrics.
        if not best_program.metrics:
            logger.info("No stored metrics found — evaluating baseline program...")
            best_program.metrics = _evaluate_program(
                best_program.code, config.evaluator_path
            )
            logger.info(f"Baseline metrics: {best_program.metrics}")

        # Auto-detect primary objective metric if not explicitly set
        score_metric = config.score_metric
        if score_metric == "combined_score":
            raw_metrics = {k: v for k, v in best_program.metrics.items()
                           if k not in _INTERNAL_METRICS and isinstance(v, (int, float))}
            if raw_metrics:
                # Prefer normalized score metrics (0–1 range, "score" in name) over raw
                # counts/penalties like slope_changes or false_reversals which can be
                # large integers and would be wrongly selected as "highest value".
                score_candidates = {k: v for k, v in raw_metrics.items()
                                    if "score" in k.lower() and 0.0 <= v <= 1.0}
                if score_candidates:
                    score_metric = max(score_candidates, key=score_candidates.get)
                else:
                    score_metric = max(raw_metrics, key=raw_metrics.get)
                logger.info(f"Auto-detected score metric: '{score_metric}' "
                            f"(override with --score-metric)")
        baseline_score = best_program.metrics.get(score_metric, best_program.metrics.get("combined_score", 0.0))
        logger.info(f"Baseline score: {baseline_score:.6f}")

        all_results = []

        for prog_idx, program in enumerate(programs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing program {prog_idx + 1}/{len(programs)}: {program.id}")
            logger.info(f"Score: {program.metrics.get('combined_score', 0.0):.6f}")

            # ── Step 1: ILS analysis ─────────────────────────────────────────
            logger.info("Step 1: Analyzing for ILS-able components...")
            from ils_engine.ils_analyzer import analyze_program

            component_spec = analyze_program(
                program=program,
                problem_description=config.problem_description,
                llm_model=config.analyzer_model or config.llm_model,
                llm_api_key=config.llm_api_key,
            )

            if component_spec is None:
                logger.warning(f"Program {program.id} is not ILS-able, skipping.")
                continue

            # ── Step 2: Strategy planning ────────────────────────────────────
            logger.info("Step 2: Planning search strategies...")
            from ils_engine.strategy_planner import plan_strategies

            strategies = plan_strategies(
                component_spec=component_spec,
                program_code=program.code,
                problem_description=config.problem_description,
                total_seeds=config.total_seeds,
                enable_tier2=config.enable_tier2_strategies,
                max_tier2=config.max_tier2_strategies,
                llm_model=config.planner_model or config.llm_model,
                llm_api_key=config.llm_api_key,
            )

            logger.info(
                f"Strategies: {[s.name for s in strategies]} "
                f"(total seeds: {sum(s.seeds for s in strategies)})"
            )

            # ── Step 3: Code generation ──────────────────────────────────────
            logger.info("Step 3: Generating search scripts...")

            scripts_dir = os.path.join(run_dir, f"_scripts_prog{prog_idx}")

            from ils_engine.code_generator import generate_scripts

            scripts = generate_scripts(
                strategies=strategies,
                program_code=program.code,
                evaluator_path=config.evaluator_path,
                component_spec=component_spec,
                problem_description=config.problem_description,
                output_dir=scripts_dir,
                llm_model=config.generator_model or config.llm_model,
                llm_api_key=config.llm_api_key,
                score_metric=score_metric,
            )

            if not scripts:
                logger.warning(f"No scripts generated for program {program.id}, skipping.")
                continue

            # ── Step 4: Execution ────────────────────────────────────────────
            logger.info(f"Step 4: Running {len(scripts)} search scripts...")

            runner_output_dir = os.path.join(run_dir, f"_runner_prog{prog_idx}")

            from ils_engine.runner import run_scripts

            results = run_scripts(
                scripts=scripts,
                num_workers=config.num_workers,
                timeout_per_strategy=config.timeout_per_strategy,
                output_dir=runner_output_dir,
                phase1_budget=config.phase1_budget,
                phase1_top_k=config.phase1_top_k,
                phase2_seed=config.phase2_seed,
                niter=config.niter,
                stepsize=config.stepsize,
            )

            all_results.extend(results)

        if not all_results:
            logger.warning("No results collected from any program.")
            # Still report baseline
            from ils_engine.reporter import print_results, save_results
            print_results([], baseline_score, task_name="unknown")
            task_dir = save_results(
                results=[],
                baseline_score=baseline_score,
                baseline_program_code=best_program.code,
                run_dir=run_dir,
            )
            return {"baseline_score": baseline_score, "output_dir": task_dir}

        # ── Step 5: Reporting ────────────────────────────────────────────────
        logger.info("Step 5: Reporting results...")
        from ils_engine.reporter import print_results, save_results

        print_results(all_results, baseline_score, task_name=task_name)

        task_dir = save_results(
            results=all_results,
            baseline_score=baseline_score,
            baseline_program_code=best_program.code,
            run_dir=run_dir,
        )

        # Compute overall best score
        valid_scores = [r.best_score for r in all_results if r.best_score > -float("inf")]
        overall_best = max(valid_scores) if valid_scores else baseline_score

        return {
            "baseline_score": baseline_score,
            "overall_best_score": overall_best,
            "improvement": overall_best - baseline_score,
            "output_dir": task_dir,
            "num_strategies": len(all_results),
        }
