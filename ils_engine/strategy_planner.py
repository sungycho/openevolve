"""
Step 2: LLM generates search strategies for ILS
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchStrategy:
    name: str
    tier: int  # 1 = universal, 2 = domain-specific
    description: str
    seeds: int
    justification: str
    sigma_values: List[float] = field(default_factory=list)  # for perturbation strategies


_SYSTEM_PROMPT = """\
You are an expert in Iterated Local Search (ILS) and combinatorial/numerical optimization. \
Your task is to design concrete search strategies for improving a given program through ILS. \
Strategies must be precisely named and concretely defined for the specific problem type. \
You must respond with valid JSON only, no markdown code blocks."""


def _build_strategy_prompt(
    problem_description: str,
    code: str,
    language: str,
    component_name: str,
    component_description: str,
    component_type: str,
    component_location: str,
    degrees_of_freedom: str,
    ils_type: str,
    max_tier2: int,
    total_seeds: int,
) -> str:
    """Build a type-specific strategy planning prompt."""

    lang_note = ""
    if language == "cpp":
        lang_note = (
            "\nNOTE: This is a C++ program. Strategies run by varying the program via env vars "
            "or source substitution — the search mechanism is the same conceptually, just "
            "executed differently. Name and describe strategies the same way as for Python."
        )

    if ils_type == "A":
        mandatory = """\
You MUST generate these two strategies (use these exact names):
1. "seed_sweep" — Try N different integer seeds. Each seed generates a different random \
initialization from scratch. Pure exploration of the initialization space. No perturbation \
from any prior result. Sigma values: [].
2. "init_perturbation" — Take the best initialization found so far (the actual configuration \
values: positions, assignments, etc.). Add Gaussian noise directly to those values. Re-run \
optimization from there. This exploits the current best neighborhood. Provide sigma_values \
as a sweep: [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100].

Additionally, generate up to {max_tier2} domain-specific Tier 2 strategies. These are \
structured initializations that bypass randomness entirely — the LLM proposes a specific \
point in configuration space based on geometric or domain insight (e.g. hexagonal grid \
for circle packing, sorted-by-constraint for scheduling). Only include if you can give \
a concrete structural justification."""

        schema_strategies = """\
    {{
      "name": "seed_sweep",
      "tier": 1,
      "description": "<specific: what each seed initializes for this problem>",
      "sigma_values": [],
      "seeds": <integer>,
      "justification": "Explores diverse initializations via different random seeds"
    }},
    {{
      "name": "init_perturbation",
      "tier": 1,
      "description": "<specific: what configuration values are perturbed and how>",
      "sigma_values": [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100],
      "seeds": <integer>,
      "justification": "Exploits neighborhood of best known initialization"
    }}"""

    elif ils_type == "B":
        mandatory = """\
You MUST generate these two strategies (use these exact names):
1. "param_local_search" — Start from the current best parameter values. Add small Gaussian \
noise to each parameter. Evaluate. This is hill-climbing in parameter space. Provide \
sigma_values as a sweep over perturbation magnitudes relative to parameter scale: \
[0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200, 0.300].
2. "param_global_sample" — Sample parameter values uniformly from a range you estimate \
as reasonable given the code context and problem constraints. Full exploration, no \
dependence on prior results. Sigma values: [].

Consider whether parameters span orders of magnitude. If so, also generate:
3. "param_log_scale_search" (Tier 1) — Same as param_local_search but perturbation in \
log space. Required if any parameter is a variance, rate, or scaling factor.

Additionally, generate up to {max_tier2} domain-specific Tier 2 strategies where you \
propose a specific parameter initialization based on problem structure (e.g. for knapsack \
with MAX_MASS=20M and MAX_VOLUME=25M: initialize mass_weight = MAX_VOLUME/MAX_MASS = 1.25 \
since the tighter mass constraint should be weighted more). Only include with concrete \
reasoning."""

        schema_strategies = """\
    {{
      "name": "param_local_search",
      "tier": 1,
      "description": "<specific: which parameters are perturbed and what scale>",
      "sigma_values": [0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200, 0.300],
      "seeds": <integer>,
      "justification": "Hill-climbing in parameter space from current best"
    }},
    {{
      "name": "param_global_sample",
      "tier": 1,
      "description": "<specific: what ranges are sampled and why>",
      "sigma_values": [],
      "seeds": <integer>,
      "justification": "Global exploration of parameter space"
    }}"""

    elif ils_type == "C":
        mandatory = """\
You MUST generate these two strategies (use these exact names). Note: discrete strategies \
NEVER use sigma_values — perturbation is via element swaps, not Gaussian noise.
1. "ordering_random_restart" — Randomly shuffle the full element ordering from scratch. \
Run the greedy algorithm on the shuffled order. Full exploration, no dependence on any \
prior result. Sigma values: [].
2. "ordering_swap" — Take the best known ordering. Randomly pick two elements and swap \
their positions. Re-run the greedy algorithm. Small perturbation — hill-climbing in \
ordering space. Sigma values: [].

Additionally, generate:
3. "ordering_segment_reverse" (Tier 1 or 2) — Take the best ordering. Pick a random \
contiguous segment of length 2 to n//4 elements. Reverse it. Medium perturbation that \
disrupts a local region while preserving global structure (like 2-opt in TSP).

And up to {max_tier2} domain-specific Tier 2 strategies where the ordering is NOT random \
but structured — e.g. sort by a different criterion (decreasing area first, then by \
aspect ratio; or sort by constraint tightness). Only include with concrete justification \
for why this ordering is likely to produce better greedy outcomes."""

        schema_strategies = """\
    {{
      "name": "ordering_random_restart",
      "tier": 1,
      "description": "<specific: what elements are reordered and what greedy algorithm runs>",
      "sigma_values": [],
      "seeds": <integer>,
      "justification": "Explores diverse orderings via full random shuffle"
    }},
    {{
      "name": "ordering_swap",
      "tier": 1,
      "description": "<specific: what two elements are swapped and how acceptance works>",
      "sigma_values": [],
      "seeds": <integer>,
      "justification": "Hill-climbing via small ordering perturbations"
    }},
    {{
      "name": "ordering_segment_reverse",
      "tier": 1,
      "description": "<specific: what segment length range and how reversed>",
      "sigma_values": [],
      "seeds": <integer>,
      "justification": "Medium perturbation preserving global ordering structure"
    }}"""

    else:  # unknown
        mandatory = """\
The component type is 'unknown' — no clear Type A/B/C structure was identified. You have \
maximum discretion. Generate strategies that you think might find a better solution, even \
if you are uncertain. Use these exact names:
1. "llm_free_perturbation" — Your best guess at what to vary. Describe concretely what \
you would change and how. Sigma values: [] (use whatever perturbation makes sense).
2. "llm_random_restart" — A completely different starting point or configuration. Sigma values: [].

Additionally, up to {max_tier2} domain-specific strategies."""

        schema_strategies = """\
    {{
      "name": "llm_free_perturbation",
      "tier": 1,
      "description": "<specific: what you propose to vary and how>",
      "sigma_values": [],
      "seeds": <integer>,
      "justification": "<your reasoning for why this might improve the solution>"
    }},
    {{
      "name": "llm_random_restart",
      "tier": 1,
      "description": "<specific: what completely different starting configuration you propose>",
      "sigma_values": [],
      "seeds": <integer>,
      "justification": "Alternative configuration to avoid local optima"
    }}"""

    mandatory_filled = mandatory.format(max_tier2=max_tier2)

    return f"""\
I have a program that solves the following problem:

PROBLEM DESCRIPTION:
{problem_description}

PROGRAM CODE ({language}):
```{language}
{code}
```

ILS-ABLE COMPONENT:
Name: {component_name}
Description: {component_description}
Type: {component_type}
ILS Type: {ils_type}
Location: {component_location}
Degrees of freedom: {degrees_of_freedom}
{lang_note}

=== STRATEGY REQUIREMENTS (ILS Type {ils_type}) ===

{mandatory_filled}

Total seed budget: {total_seeds} — split across all strategies proportionally \
(Tier 1 strategies get roughly equal shares; Tier 2 strategies get smaller shares).

=== RESPONSE FORMAT ===

Respond with exactly this JSON (no markdown):
{{
  "strategies": [
    {schema_strategies},
    // 0-{max_tier2} domain-specific strategies with tier=2
    {{
      "name": "<strategy_name>",
      "tier": 2,
      "description": "<what this strategy does, concretely>",
      "sigma_values": [],
      "seeds": <integer>,
      "justification": "<concrete structural justification from problem/code analysis>"
    }}
  ]
}}

=== WORKED EXAMPLE (for format reference only — your strategies must be specific to THIS program) ===

Example: Type A program (circle packing), total_seeds=150, max_tier2=1

INPUT:
  ILS Type: A | Component: "circle_positions" | DoF: "x,y coordinates of N=100 circles"
  Problem: Pack N circles of radius r into a unit square, maximize minimum pairwise distance.

CORRECT OUTPUT:
{{
  "strategies": [
    {{
      "name": "seed_sweep",
      "tier": 1,
      "description": "Try 75 different integer seeds. Each seed randomly scatters all 100 circle centers uniformly in [r, 1-r]^2, then runs L-BFGS-B to push circles apart. Pure exploration — no dependence on prior results.",
      "sigma_values": [],
      "seeds": 75,
      "justification": "Explores diverse initial configurations; random scatter covers the initialization space broadly"
    }},
    {{
      "name": "init_perturbation",
      "tier": 1,
      "description": "Take the best circle positions found so far. Add Gaussian noise (sigma from sweep) independently to each (x, y) coordinate. Clip to [r, 1-r]. Re-run L-BFGS-B from the perturbed positions. Exploits best known configuration neighborhood.",
      "sigma_values": [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100],
      "seeds": 60,
      "justification": "Small perturbations from best known config find nearby local optima that random restart would miss"
    }},
    {{
      "name": "hexagonal_grid_init",
      "tier": 2,
      "description": "Initialize circles on a hexagonal close-packing grid scaled to fit the unit square. Hexagonal arrangement maximizes packing density for equal circles. Add small jitter (sigma=0.01) to break symmetry before optimization.",
      "sigma_values": [],
      "seeds": 15,
      "justification": "Hexagonal close-packing is the theoretically optimal arrangement for equal circles in 2D. Starting from this structured geometry is more likely to converge to globally optimal configurations than random scatter, particularly for the dense-packing regime."
    }}
  ]
}}

NOTE: The example above is for illustration only. Your response must describe strategies \
specific to the actual program and component described above — different variable names, \
different perturbation logic, different domain justifications."""


def plan_strategies(
    component_spec,
    program_code: str,
    problem_description: str = "",
    total_seeds: int = 150,
    enable_tier2: bool = True,
    max_tier2: int = 2,
    llm_model: str = "claude-sonnet-4-6",
    llm_api_key: Optional[str] = None,
) -> List[SearchStrategy]:
    """
    Generate search strategies for the given ILS component.

    Args:
        component_spec: ILSComponentSpec from ils_analyzer
        program_code: Original program source code
        problem_description: Optional domain context
        total_seeds: Total seed budget to split across strategies
        enable_tier2: Whether to generate domain-specific strategies
        max_tier2: Maximum number of Tier 2 strategies
        llm_model: Anthropic model name
        llm_api_key: API key (defaults to ANTHROPIC_API_KEY env var)

    Returns:
        List of SearchStrategy objects
    """
    from openai import OpenAI

    api_key = llm_api_key or os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    max_t2 = max_tier2 if enable_tier2 else 0
    ils_type = getattr(component_spec, "ils_type", "unknown") or "unknown"
    language = getattr(component_spec, "language", "python") or "python"

    prompt = _build_strategy_prompt(
        problem_description=problem_description or "No description provided.",
        code=program_code,
        language=language,
        component_name=component_spec.name,
        component_description=component_spec.description,
        component_type=component_spec.type,
        component_location=component_spec.location,
        degrees_of_freedom=component_spec.degrees_of_freedom,
        ils_type=ils_type,
        max_tier2=max_t2,
        total_seeds=total_seeds,
    )

    logger.info("Generating search strategies via LLM...")

    try:
        response = client.chat.completions.create(
            model=llm_model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code blocks if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        result = json.loads(raw)

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse strategy JSON: {e}. Falling back to default strategies.")
        return _default_strategies(total_seeds, ils_type)
    except Exception as e:
        logger.error(f"LLM call failed during strategy planning: {e}")
        raise

    raw_strategies = result.get("strategies", [])

    strategies = []
    tier2_count = 0

    for s in raw_strategies:
        tier = s.get("tier", 1)
        if tier == 2:
            if not enable_tier2 or tier2_count >= max_tier2:
                continue
            tier2_count += 1

        strategy = SearchStrategy(
            name=s.get("name", "unknown"),
            tier=tier,
            description=s.get("description", ""),
            seeds=int(s.get("seeds", total_seeds // max(1, len(raw_strategies)))),
            justification=s.get("justification", ""),
            sigma_values=s.get("sigma_values", []),
        )
        strategies.append(strategy)

    # Ensure at least one strategy was generated (fallback if LLM returned empty)
    if not strategies:
        logger.warning("LLM returned no strategies. Falling back to defaults.")
        return _default_strategies(total_seeds, ils_type)

    # Redistribute seeds evenly if not already set
    total_assigned = sum(s.seeds for s in strategies)
    if total_assigned == 0:
        for s in strategies:
            s.seeds = total_seeds // len(strategies)
    elif total_assigned != total_seeds:
        # Scale proportionally
        scale = total_seeds / total_assigned
        for s in strategies:
            s.seeds = max(1, int(s.seeds * scale))

    logger.info(f"Generated {len(strategies)} strategies: {[s.name for s in strategies]}")
    return strategies


def _default_strategies(total_seeds: int, ils_type: str = "unknown") -> List[SearchStrategy]:
    """Fallback strategies when LLM fails, adapted to ILS type."""
    per = total_seeds // 2
    remainder = total_seeds - per

    if ils_type == "A":
        return [
            SearchStrategy(
                name="seed_sweep",
                tier=1,
                description="Try different random seeds for initialization",
                seeds=per,
                justification="Universal exploration via different seeds",
                sigma_values=[],
            ),
            SearchStrategy(
                name="init_perturbation",
                tier=1,
                description="Perturb the best initialization with Gaussian noise",
                seeds=remainder,
                justification="Exploit best known initialization neighborhood",
                sigma_values=[0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100],
            ),
        ]
    elif ils_type == "B":
        return [
            SearchStrategy(
                name="param_local_search",
                tier=1,
                description="Perturb parameter values with Gaussian noise",
                seeds=per,
                justification="Hill-climbing in parameter space",
                sigma_values=[0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200, 0.300],
            ),
            SearchStrategy(
                name="param_global_sample",
                tier=1,
                description="Sample parameters uniformly from reasonable range",
                seeds=remainder,
                justification="Global exploration of parameter space",
                sigma_values=[],
            ),
        ]
    elif ils_type == "C":
        third = total_seeds // 3
        return [
            SearchStrategy(
                name="ordering_random_restart",
                tier=1,
                description="Randomly shuffle element ordering from scratch",
                seeds=third,
                justification="Explores diverse orderings",
                sigma_values=[],
            ),
            SearchStrategy(
                name="ordering_swap",
                tier=1,
                description="Swap two elements in the best known ordering",
                seeds=third,
                justification="Hill-climbing via small ordering perturbations",
                sigma_values=[],
            ),
            SearchStrategy(
                name="ordering_segment_reverse",
                tier=1,
                description="Reverse a random contiguous segment of the best ordering",
                seeds=total_seeds - 2 * third,
                justification="Medium perturbation preserving global ordering structure",
                sigma_values=[],
            ),
        ]
    else:  # unknown
        return [
            SearchStrategy(
                name="llm_free_perturbation",
                tier=1,
                description="Perturb the program's key decision variable",
                seeds=per,
                justification="Best-guess perturbation for unknown type",
                sigma_values=[],
            ),
            SearchStrategy(
                name="llm_random_restart",
                tier=1,
                description="Completely different starting configuration",
                seeds=remainder,
                justification="Alternative configuration to avoid local optima",
                sigma_values=[],
            ),
        ]
