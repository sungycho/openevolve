"""
Step 3: LLM generates runnable search scripts for each strategy
"""

import ast
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GeneratedScript:
    strategy_name: str
    code: str
    script_path: str  # path where script is saved on disk


_SYSTEM_PROMPT = (
    "You are an expert Python programmer specialising in numerical optimisation. "
    "Your task is to generate a complete, runnable Python script that implements a "
    "two-phase search strategy (Phase 1: parallel screening, Phase 2: basin-hopping) "
    "for a given optimisation program. "
    "The script must be syntactically valid Python and directly executable. "
    "Respond with raw Python code only — no markdown, no explanation, no code fences."
)


# Condensed reference implementation embedded in every prompt so the LLM has a
# concrete, proven skeleton to follow rather than inventing its own structure.
_PHASE2_REFERENCE = '''
# ── REFERENCE IMPLEMENTATION (two-phase basin-hopping) ──────────────────────
# The script you generate MUST follow this two-phase pattern exactly.
#
# PHASE 1 — parallel screening
#   • Generate `phase1_budget` candidate starting points using THIS strategy's
#     initialization logic (perturb grid / random restart / hexagonal / etc.).
#   • Run SLSQP on each candidate in parallel via ProcessPoolExecutor.
#   • Keep only feasible results; sort descending by score.
#   • Pass the top `phase1_top_k` winners (their optimized x vectors) to Phase 2.
#
# PHASE 2 — basin-hopping chains (scipy.optimize.basinhopping)
#   • For each Phase 1 winner, run an independent basin-hopping chain.
#   • Use a custom step class (like PerturbStep below) to perturb ONLY the
#     search degrees of freedom (e.g. layout parameters OR hyperparameter values),
#     not the deterministically-optimized values —
#     leave SLSQP free to re-optimise everything else.
#   • Seed BOTH the step class RNG and basinhopping(..., seed=...) from the
#     same integer so every run is fully reproducible given the same seeds.
#   • Record every accepted feasible step as (step_idx, score, x.copy()).
#   • After all chains finish, report the global best x.
#
# KEY SKELETON (adapt to your problem):
#
#   import numpy as np
#   from scipy.optimize import basinhopping, minimize
#   from concurrent.futures import ProcessPoolExecutor, as_completed
#
#   class PerturbStep:
#       def __init__(self, stepsize, seed):
#           self.stepsize = stepsize
#           self.rng = np.random.default_rng(seed)   # MUST be seeded
#       def __call__(self, x):
#           x = x.copy()
#           x[<stochastic_dims>] += self.rng.normal(0, self.stepsize,
#                                                   size=len(<stochastic_dims>))
#           x[<stochastic_dims>] = np.clip(x[<stochastic_dims>], lo, hi)
#           return x
#
#   def phase1_worker(seed, sigma):
#       x0 = make_starting_point(seed, sigma)   # THIS strategy's init logic
#       res = minimize(objective, x0, ...)
#       return seed, float(-res.fun), check_feasible(res.x), res.x
#
#   def phase2_worker(start_seed, phase2_seed, niter, stepsize, x0):
#       accepted = []
#       step_counter = [0]
#       def callback(x, f, accepted_flag):
#           step_counter[0] += 1
#           if accepted_flag and check_feasible(x):
#               accepted.append((step_counter[0], float(-f), x.copy()))
#       step = PerturbStep(stepsize, seed=phase2_seed)
#       result = basinhopping(objective, x0, niter=niter,
#                             minimizer_kwargs={...},
#                             take_step=step, callback=callback,
#                             seed=phase2_seed)
#       return float(-result.fun), check_feasible(result.x), accepted, result.x
#
#   # CLI args: --n-workers, --evaluator-path,
#   #           --phase1-budget (default: n_seeds // 4),
#   #           --phase1-top-k  (default: 4),
#   #           --phase2-seed   (default: 0, offset added to each winner seed),
#   #           --niter         (default: 100),
#   #           --stepsize      (default: 0.04)
#
#   # Sigma sweep in Phase 1 (for perturbation strategies):
#   #   for seed_i in range(phase1_budget):
#   #       sigma = sigma_values[seed_i % len(sigma_values)]
#   #   This ensures every sigma value is sampled with many different seeds.
#
# ── END REFERENCE ────────────────────────────────────────────────────────────
'''


def _build_prompt(
    problem_description: str,
    program_code: str,
    component_name: str,
    component_description: str,
    component_type: str,
    component_location: str,
    degrees_of_freedom: str,
    strategy_name: str,
    strategy_description: str,
    strategy_tier: int,
    seeds: int,
    sigma_values: list,
    strategy_justification: str,
    evaluator_path: str,
    score_metric: str = "combined_score",
) -> str:
    sigma_str = json.dumps(sigma_values) if sigma_values else "[]"
    phase1_budget = max(seeds // 4, 20)
    has_sigma = bool(sigma_values)

    lines = [
        "Generate a complete two-phase search script for the following ILS strategy.",
        "",
        _PHASE2_REFERENCE,
        "",
        "PROBLEM DESCRIPTION:",
        problem_description or "No description provided.",
        "",
        "ORIGINAL PROGRAM CODE (include this verbatim as ORIGINAL_CODE in the script):",
        "--- BEGIN PROGRAM ---",
        program_code,
        "--- END PROGRAM ---",
        "",
        "ILS COMPONENT:",
        f"Name: {component_name}",
        f"Description: {component_description}",
        f"Type: {component_type}",
        f"Location: {component_location}",
        f"Degrees of freedom: {degrees_of_freedom}",
        "",
        "STRATEGY TO IMPLEMENT:",
        f"Name: {strategy_name}",
        f"Description: {strategy_description}",
        f"Tier: {strategy_tier}",
        f"Total seed budget: {seeds}",
        f"Sigma values (perturbation strategies only): {sigma_str}",
        f"Justification: {strategy_justification}",
        "",
        f"EVALUATOR PATH: {evaluator_path}",
        "",
        "MANDATORY TWO-PHASE STRUCTURE:",
        f"Phase 1 — parallel screening ({phase1_budget} candidates by default via --phase1-budget):",
        "  • Implement THIS strategy's initialization logic to generate starting points.",
        "  • For perturbation strategies: sweep sigma as a 2D axis alongside seed:",
        f"      sigma = sigma_values[seed_i % len(sigma_values)]  # ALL values in {sigma_str}",
        "    Start perturbation from the best optimized solution in ORIGINAL_CODE, not a raw grid.",
        "  • For random_restart: generate fully random initializations.",
        "  • For domain-specific (tier 2): use the structural layout described above.",
        "  • Run SLSQP on each candidate in parallel; keep feasible results sorted by score.",
        "  • Pass top --phase1-top-k (default 4) winners' optimized x vectors to Phase 2.",
        "",
        f"Phase 2 — basin-hopping chains (--niter steps each, default 100):",
        "  • Run one independent scipy.optimize.basinhopping chain per Phase 1 winner.",
        "  • Use a custom step class that perturbs ONLY the search degrees of freedom",
        "    (e.g. center coordinates for layout problems, or hyperparameter values for",
        "    tuning problems), seeded with: winner_seed + phase2_seed_offset.",
        "  • Also pass seed=winner_seed+phase2_seed_offset to basinhopping() itself.",
        "  • Record every accepted feasible step as (step_idx, score, x.copy()) in callback.",
        "  • Run Phase 2 chains in parallel via ProcessPoolExecutor.",
        "  • Skip Phase 2 for strategies where the search space is fully deterministic.",
        "",
        "EVALUATOR USAGE (CRITICAL):",
        "  Load:  importlib.util.spec_from_file_location('evaluator', EVALUATOR_PATH)",
        "  Call:  metrics = evaluator.evaluate(tmp_program_path)  # returns a DICT",
        f"  Score: score = metrics.get('{score_metric}', 0.0)",
        f"  The score metric is '{score_metric}'. Use THIS key ONLY — do NOT use 'combined_score',",
        "  'target_ratio', or any other key. 'combined_score' is a normalized internal ratio (≤1)",
        "  and will make all scores look identical. Always extract a float — never compare dicts.",
        "",
        "CLI ARGUMENTS (all scripts must support these):",
        "  --n-workers        INT   parallel workers (default 1)",
        "  --evaluator-path   STR   override EVALUATOR_PATH",
        "  --phase1-budget    INT   Phase 1 candidates (default {})".format(phase1_budget),
        "  --phase1-top-k     INT   winners passed to Phase 2 (default 4)",
        "  --phase2-seed      INT   offset added to each winner seed for Phase 2 (default 0)",
        "  --niter            INT   basin-hopping steps per chain (default 100)",
        "  --stepsize         FLOAT perturbation magnitude (default 0.04)",
        "",
        "OUTPUT FORMAT:",
        "  Each evaluated candidate (both phases) must emit a JSON line to stdout:",
        '  {"seed": <int>, "score": <float>, "strategy": "<name>", "sigma": <float_or_null>, "phase": <1_or_2>}',
        "  Final summary line:",
        '  {"type": "summary", "best_score": <float>, "best_seed": <int>, "best_phase2_seed": <int_or_null>, "best_step": <int_or_null>}',
        "  If env var BEST_PROGRAM_OUTPUT is set, write the best program code to that path.",
        "",
        "OTHER REQUIREMENTS:",
        "  • Standalone script — no deps beyond numpy/scipy/standard library.",
        "  • Store ORIGINAL_CODE as a module-level string variable.",
        "  • All generated program code strings must be valid Python.",
        "  • Use concurrent.futures.ProcessPoolExecutor for all parallelism.",
        "",
        "Write the complete, working Python script now. No markdown fences, just raw Python code.",
    ]
    return "\n".join(lines)


def _call_llm_for_code(prompt: str, llm_model: str, client) -> str:
    """Call LLM and return generated code string."""
    response = client.chat.completions.create(
        model=llm_model,
        max_tokens=16384,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if LLM added them anyway
    for fence in ("```python", "```"):
        if raw.startswith(fence):
            raw = raw[len(fence):]
            break
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def generate_scripts(
    strategies: List,
    program_code: str,
    evaluator_path: str,
    component_spec,
    problem_description: str = "",
    output_dir: str = "/tmp/ils_scripts",
    llm_model: str = "claude-sonnet-4-6",
    llm_api_key: Optional[str] = None,
    score_metric: str = "combined_score",
) -> List[GeneratedScript]:
    """
    Generate runnable search scripts for each strategy.

    Args:
        strategies: List of SearchStrategy objects
        program_code: Original program source code
        evaluator_path: Absolute path to evaluator.py
        component_spec: ILSComponentSpec
        problem_description: Optional domain context
        output_dir: Directory to save generated scripts
        llm_model: Anthropic model name
        llm_api_key: API key (defaults to ANTHROPIC_API_KEY env var)

    Returns:
        List of GeneratedScript objects
    """
    from openai import OpenAI

    api_key = llm_api_key or os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    os.makedirs(output_dir, exist_ok=True)
    evaluator_path = str(Path(evaluator_path).resolve())

    scripts = []

    for strategy in strategies:
        logger.info(f"Generating code for strategy: {strategy.name}")

        prompt = _build_prompt(
            problem_description=problem_description or "No description provided.",
            program_code=program_code,
            component_name=component_spec.name,
            component_description=component_spec.description,
            component_type=component_spec.type,
            component_location=component_spec.location,
            degrees_of_freedom=component_spec.degrees_of_freedom,
            strategy_name=strategy.name,
            strategy_description=strategy.description,
            strategy_tier=strategy.tier,
            seeds=strategy.seeds,
            sigma_values=strategy.sigma_values,
            strategy_justification=strategy.justification,
            evaluator_path=evaluator_path,
            score_metric=score_metric,
        )

        code = None
        for attempt in range(2):
            try:
                generated = _call_llm_for_code(prompt, llm_model, client)

                # Validate syntax
                ast.parse(generated)
                code = generated
                logger.info(f"Strategy '{strategy.name}': code generated and validated (attempt {attempt+1})")
                break

            except SyntaxError as e:
                if attempt == 0:
                    logger.warning(
                        f"Strategy '{strategy.name}': syntax error on attempt 1: {e}. Retrying..."
                    )
                    prompt = prompt + f"\n\nPREVIOUS ATTEMPT HAD SYNTAX ERROR:\n{e}\nPlease fix it."
                else:
                    logger.error(
                        f"Strategy '{strategy.name}': syntax error persists after retry: {e}"
                    )
            except Exception as e:
                logger.error(f"Strategy '{strategy.name}': LLM call failed: {e}")
                break

        if code is None:
            logger.warning(f"Skipping strategy '{strategy.name}' due to code generation failure")
            continue

        # Save script to disk
        script_path = os.path.join(output_dir, f"{strategy.name}_search.py")
        with open(script_path, "w") as f:
            f.write(code)

        scripts.append(GeneratedScript(
            strategy_name=strategy.name,
            code=code,
            script_path=script_path,
        ))

    logger.info(f"Generated {len(scripts)}/{len(strategies)} scripts successfully")
    return scripts
