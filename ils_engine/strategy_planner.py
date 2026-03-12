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
    tier: int                          # 1 = universal, 2 = domain-specific
    description: str
    seeds: int
    justification: str
    sigma_values: List[float] = field(default_factory=list)  # for perturbation strategies


_SYSTEM_PROMPT = """\
You are an expert in Iterated Local Search (ILS) and stochastic optimization. \
Your task is to design concrete search strategies for improving a given program \
through ILS. You must respond with valid JSON only, no markdown code blocks."""

_USER_PROMPT_TEMPLATE = """\
I have a program that solves the following problem:

PROBLEM DESCRIPTION:
{problem_description}

PROGRAM CODE:
```python
{code}
```

ILS-ABLE COMPONENT:
Name: {component_name}
Description: {component_description}
Type: {component_type}
Location: {component_location}
Degrees of freedom: {degrees_of_freedom}

I need to generate search strategies for ILS. You must always generate:
1. "perturb_best" - Perturb the ILS component from the current best solution with noise
2. "random_restart" - Completely randomize the ILS component

Additionally, generate up to {max_tier2} domain-specific structured initialization strategies \
that exploit knowledge of the problem structure. Only include a strategy if you can \
provide a concrete structural justification. Skip if you cannot justify it.

Total seed budget: {total_seeds} (split equally across all strategies)

Respond with exactly this JSON structure (no markdown):
{{
  "strategies": [
    {{
      "name": "perturb_best",
      "tier": 1,
      "description": "<specific description for this problem>",
      "sigma_values": [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100],
      "seeds": <integer>,
      "justification": "Universal exploitation; exploits known-good solution neighborhood"
    }},
    {{
      "name": "random_restart",
      "tier": 1,
      "description": "<specific description for this problem>",
      "sigma_values": [],
      "seeds": <integer>,
      "justification": "Universal exploration; avoids local optima"
    }},
    // 0-{max_tier2} domain-specific strategies with tier=2
    {{
      "name": "<strategy_name>",
      "tier": 2,
      "description": "<what this strategy does>",
      "sigma_values": [],
      "seeds": <integer>,
      "justification": "<concrete structural justification>"
    }}
  ]
}}"""


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

    prompt = _USER_PROMPT_TEMPLATE.format(
        problem_description=problem_description or "No description provided.",
        code=program_code,
        component_name=component_spec.name,
        component_description=component_spec.description,
        component_type=component_spec.type,
        component_location=component_spec.location,
        degrees_of_freedom=component_spec.degrees_of_freedom,
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
        return _default_strategies(total_seeds)
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

    # Ensure at least the two universal strategies are present
    names = {s.name for s in strategies}
    if "perturb_best" not in names:
        strategies.insert(0, SearchStrategy(
            name="perturb_best",
            tier=1,
            description="Perturb the best solution with Gaussian noise",
            seeds=total_seeds // 2,
            justification="Universal exploitation strategy",
            sigma_values=[0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100],
        ))
    if "random_restart" not in names:
        strategies.append(SearchStrategy(
            name="random_restart",
            tier=1,
            description="Completely randomize the initialization",
            seeds=total_seeds // 2,
            justification="Universal exploration strategy",
            sigma_values=[],
        ))

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


def _default_strategies(total_seeds: int) -> List[SearchStrategy]:
    """Fallback strategies when LLM fails."""
    per = total_seeds // 2
    return [
        SearchStrategy(
            name="perturb_best",
            tier=1,
            description="Perturb the best solution with Gaussian noise",
            seeds=per,
            justification="Universal exploitation strategy",
            sigma_values=[0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100],
        ),
        SearchStrategy(
            name="random_restart",
            tier=1,
            description="Completely randomize the initialization",
            seeds=total_seeds - per,
            justification="Universal exploration strategy",
            sigma_values=[],
        ),
    ]
