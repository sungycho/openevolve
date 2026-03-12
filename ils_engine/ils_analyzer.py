"""
Step 1: LLM identifies ILS-able stochastic component in a program
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ILSComponentSpec:
    name: str
    description: str
    type: str  # "continuous" | "discrete" | "constrained"
    location: str
    degrees_of_freedom: str
    reasoning: str


_ANALYSIS_SCHEMA = {
    "is_ils_able": True,
    "component": {
        "name": "<name>",
        "description": "<description>",
        "type": "<continuous|discrete|constrained>",
        "location": "<function/lines>",
        "degrees_of_freedom": "<description>",
    },
    "reasoning": "<reasoning>",
}

_NOT_ILS_ABLE_SCHEMA = {
    "is_ils_able": False,
    "reasoning": "<reasoning>",
}

_SYSTEM_PROMPT = """\
You are an expert algorithm analyst specializing in Iterated Local Search (ILS) and \
numerical optimization. Your task is to analyze a given program and identify whether \
it contains a component that could benefit from ILS — either a stochastic initialization \
component (random seeds, random layouts) OR tunable continuous hyperparameters \
(fixed default values that control algorithm behavior and whose optimal values are unknown).

You must respond with valid JSON only, no markdown code blocks."""

_USER_PROMPT_TEMPLATE = """\
Analyze the following program and determine if it has an ILS-able component.

PROBLEM DESCRIPTION:
{problem_description}

PROGRAM CODE:
```python
{code}
```

An ILS-able component is ANY of the following:

TYPE A — Stochastic initialization (random seed / random restart):
  • There is explicit randomness: np.random, random.seed, torch.manual_seed, etc.
  • Different random seeds or initial placements lead to different final solutions.
  • Example: circle packing with random center initialization.

TYPE B — Tunable continuous hyperparameters (fixed defaults that can be optimized):
  • The algorithm has numeric parameters with hardcoded default values.
  • Perturbing these parameters changes algorithm behavior and output quality.
  • The optimal values are unknown and problem-dependent.
  • Example: Kalman filter with process_variance=0.01, measurement_variance=0.1, window_size=20.
  • Example: neural network with learning_rate=0.001, hidden_size=128, dropout=0.2.

TYPE C — Structural/layout parameters (discrete or constrained search spaces):
  • Discrete choices that affect solution structure.
  • Example: number of clusters, polynomial degree, filter type.

A program is ILS-able if it has ANY of the above. Most non-trivial algorithms have
at least tunable hyperparameters (Type B), so default to ILS-able unless the program
is purely deterministic with no configurable parameters whatsoever.

Prefer Type A if present (stochastic search is most powerful). Fall back to Type B
(hyperparameter tuning) if no stochastic component exists.

Respond with exactly this JSON structure (no markdown):

If ILS-able:
{{
  "is_ils_able": true,
  "component": {{
    "name": "<short name for the component>",
    "description": "<1-2 sentence description of what it does>",
    "type": "<continuous|discrete|constrained>",
    "location": "<function name and approximate line range>",
    "degrees_of_freedom": "<what can be perturbed, e.g. 'process_variance, measurement_variance, window_size' or 'x,y coordinates of N centers'>"
  }},
  "reasoning": "<why this component is ILS-able and which type (A/B/C) applies>"
}}

If not ILS-able:
{{
  "is_ils_able": false,
  "reasoning": "<why the algorithm has no stochastic components AND no tunable hyperparameters>"
}}"""


def analyze_program(
    program,
    problem_description: str = "",
    llm_model: str = "claude-sonnet-4-6",
    llm_api_key: Optional[str] = None,
) -> Optional[ILSComponentSpec]:
    """
    Use LLM to analyze a program and identify its ILS-able component.

    Args:
        program: Program object with .code attribute
        problem_description: Optional domain context
        llm_model: Anthropic model name
        llm_api_key: API key (defaults to ANTHROPIC_API_KEY env var)

    Returns:
        ILSComponentSpec if ILS-able, None otherwise
    """
    from openai import OpenAI

    api_key = llm_api_key or os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    prompt = _USER_PROMPT_TEMPLATE.format(
        problem_description=problem_description or "No description provided.",
        code=program.code,
    )

    logger.info(f"Analyzing program {program.id} for ILS-able components...")

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
        logger.warning(f"Failed to parse LLM JSON response: {e}. Marking as not ILS-able.")
        return None
    except Exception as e:
        logger.error(f"LLM call failed during ILS analysis: {e}")
        raise

    if not result.get("is_ils_able", False):
        logger.info(f"Program {program.id} is not ILS-able: {result.get('reasoning', '')}")
        return None

    comp = result.get("component", {})
    spec = ILSComponentSpec(
        name=comp.get("name", "unknown"),
        description=comp.get("description", ""),
        type=comp.get("type", "continuous"),
        location=comp.get("location", ""),
        degrees_of_freedom=comp.get("degrees_of_freedom", ""),
        reasoning=result.get("reasoning", ""),
    )

    logger.info(
        f"Program {program.id} is ILS-able. Component: '{spec.name}' ({spec.type}) "
        f"at {spec.location}"
    )
    return spec
