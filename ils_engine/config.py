"""
Configuration for the ILS Engine
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ILSConfig:
    # Required
    openevolve_output_path: str = ""   # path to checkpoint or best/ folder
    evaluator_path: str = ""           # path to evaluator.py

    # Optional
    problem_description: str = ""      # helps LLM understand domain
    score_metric: str = "combined_score"  # metric key to use as the objective score

    # Program selection
    top_k: int = 1
    selection_mode: str = "novelty_rejection"  # "top_1"|"vanilla_top_k"|"novelty_rejection"
    novelty_threshold: float = 0.3

    # LLM (OpenRouter)
    llm_model: str = "anthropic/claude-sonnet-4-6"  # fallback for all steps
    analyzer_model: Optional[str] = "anthropic/claude-sonnet-4-6"   # Step 1: identify ILS-able components
    planner_model: Optional[str] = "anthropic/claude-sonnet-4-6"    # Step 2: plan search strategies
    generator_model: Optional[str] = "openai/gpt-5.3-codex"  # Step 3: generate search scripts (most demanding)
    llm_api_key: Optional[str] = None      # defaults to OPENROUTER_API_KEY env var
    llm_api_base: str = "https://openrouter.ai/api/v1"

    # Strategies
    enable_tier2_strategies: bool = True
    max_tier2_strategies: int = 2
    total_seeds: int = 150             # split across strategies

    # Execution
    num_workers: int = 8
    timeout_per_strategy: int = 300

    # Two-phase search (forwarded to generated scripts)
    phase1_budget: Optional[int] = None   # Phase 1 candidates (default: let script decide)
    phase1_top_k: int = 4                 # Phase 1 winners passed to Phase 2
    phase2_seed: int = 0                  # Seed offset for Phase 2 chains
    niter: int = 100                      # Basin-hopping steps per Phase 2 chain
    stepsize: float = 0.04                # Perturbation magnitude

    # Output
    output_dir: str = "ils_output"

    # Optional stubs (not implemented in v1)
    enable_budget_allocation: bool = False
