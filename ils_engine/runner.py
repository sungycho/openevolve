"""
Step 4: Execute generated search scripts and collect results
"""

import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    strategy_name: str
    best_score: float
    best_seed: int
    best_program_code: Optional[str]
    all_seed_scores: List[dict]       # list of {seed, score, sigma} dicts
    runtime_s: float
    error: Optional[str] = None


def _run_single_script(
    script: object,
    num_workers: int,
    timeout_s: int,
    best_program_output: Optional[str] = None,
    phase1_budget: Optional[int] = None,
    phase1_top_k: int = 4,
    phase2_seed: int = 0,
    niter: int = 100,
    stepsize: float = 0.04,
) -> StrategyResult:
    """Execute one generated script and parse its stdout."""
    start = time.time()

    env = os.environ.copy()
    if best_program_output:
        env["BEST_PROGRAM_OUTPUT"] = best_program_output

    cmd = [
        sys.executable,
        script.script_path,
        "--n-workers",    str(max(1, num_workers)),
        "--phase1-top-k", str(phase1_top_k),
        "--phase2-seed",  str(phase2_seed),
        "--niter",        str(niter),
        "--stepsize",     str(stepsize),
    ]
    if phase1_budget is not None:
        cmd += ["--phase1-budget", str(phase1_budget)]

    logger.info(f"Running strategy '{script.strategy_name}': {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        runtime = time.time() - start
        logger.warning(f"Strategy '{script.strategy_name}' timed out after {timeout_s}s")
        return StrategyResult(
            strategy_name=script.strategy_name,
            best_score=-float("inf"),
            best_seed=-1,
            best_program_code=None,
            all_seed_scores=[],
            runtime_s=runtime,
            error=f"Timeout after {timeout_s}s",
        )
    except Exception as e:
        runtime = time.time() - start
        logger.error(f"Strategy '{script.strategy_name}' failed to run: {e}")
        return StrategyResult(
            strategy_name=script.strategy_name,
            best_score=-float("inf"),
            best_seed=-1,
            best_program_code=None,
            all_seed_scores=[],
            runtime_s=runtime,
            error=str(e),
        )

    runtime = time.time() - start

    if proc.returncode != 0:
        logger.warning(
            f"Strategy '{script.strategy_name}' exited with code {proc.returncode}\n"
            f"stderr:\n{proc.stderr[:2000]}"
        )

    # Parse stdout JSON lines
    seed_scores = []
    best_score = -float("inf")
    best_seed = -1

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip non-JSON lines (e.g. debug prints)

        if record.get("type") == "summary":
            best_score = record.get("best_score", best_score)
            best_seed = record.get("best_seed", best_seed)
        elif "seed" in record and "score" in record:
            seed_scores.append(record)

    # Load best program code if saved
    best_code = None
    if best_program_output and os.path.exists(best_program_output):
        try:
            with open(best_program_output, "r") as f:
                best_code = f.read()
        except Exception:
            pass

    # Fallback: derive best from seed_scores if summary missing
    if best_score == -float("inf") and seed_scores:
        best_record = max(seed_scores, key=lambda r: r.get("score", -float("inf")))
        best_score = best_record.get("score", -float("inf"))
        best_seed = best_record.get("seed", -1)

    # If no output at all, log stderr to help debug silent crashes
    if not seed_scores and best_score == -float("inf") and proc.stderr.strip():
        logger.error(
            f"Strategy '{script.strategy_name}' produced no output. "
            f"stderr:\n{proc.stderr[:3000]}"
        )

    logger.info(
        f"Strategy '{script.strategy_name}' done in {runtime:.1f}s. "
        f"Best score: {best_score:.6f} (seed {best_seed})"
    )

    return StrategyResult(
        strategy_name=script.strategy_name,
        best_score=best_score,
        best_seed=best_seed,
        best_program_code=best_code,
        all_seed_scores=seed_scores,
        runtime_s=runtime,
    )


def _run_script_worker(args):
    """Module-level wrapper so ProcessPoolExecutor can pickle it."""
    script, num_workers, timeout_s, best_program_output, phase_kwargs = args
    return _run_single_script(script, num_workers, timeout_s, best_program_output, **phase_kwargs)


def run_scripts(
    scripts: List,
    num_workers: int = 8,
    timeout_per_strategy: int = 300,
    output_dir: str = "/tmp/ils_output",
    phase1_budget: Optional[int] = None,
    phase1_top_k: int = 4,
    phase2_seed: int = 0,
    niter: int = 100,
    stepsize: float = 0.04,
) -> List[StrategyResult]:
    """
    Execute all generated scripts in parallel (one subprocess per strategy).

    Args:
        scripts: List of GeneratedScript objects
        num_workers: Workers passed to each script's internal parallelism
        timeout_per_strategy: Max seconds per strategy script
        output_dir: Directory to save best program files per strategy

    Returns:
        List of StrategyResult objects
    """
    os.makedirs(output_dir, exist_ok=True)

    phase_kwargs = {
        "phase1_budget": phase1_budget,
        "phase1_top_k": phase1_top_k,
        "phase2_seed": phase2_seed,
        "niter": niter,
        "stepsize": stepsize,
    }

    # Run strategies in parallel (one process per strategy)
    results = []
    run_args = [
        (s, num_workers, timeout_per_strategy, os.path.join(output_dir, f"{s.strategy_name}_best.py"), phase_kwargs)
        for s in scripts
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(scripts)) as executor:
        future_to_script = {executor.submit(_run_script_worker, a): a[0] for a in run_args}
        for future in concurrent.futures.as_completed(future_to_script):
            script = future_to_script[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Strategy '{script.strategy_name}' executor error: {e}")
                results.append(StrategyResult(
                    strategy_name=script.strategy_name,
                    best_score=-float("inf"),
                    best_seed=-1,
                    best_program_code=None,
                    all_seed_scores=[],
                    runtime_s=0.0,
                    error=str(e),
                ))

    # Sort by strategy name for deterministic ordering
    results.sort(key=lambda r: r.strategy_name)
    return results
