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
stochastic optimization. Your task is to analyze a given program and identify whether \
it contains a stochastic initialization component that could benefit from ILS — i.e., \
a component where different random seeds or initial configurations lead to different \
final solutions.

You must respond with valid JSON only, no markdown code blocks."""

_USER_PROMPT_TEMPLATE = """\
Analyze the following program and determine if it has an ILS-able stochastic component.

PROBLEM DESCRIPTION:
{problem_description}

PROGRAM CODE:
```python
{code}
```

An ILS-able component is one where:
1. There is a stochastic initialization (random seed, random placement, random restart)
2. Different initializations can lead to meaningfully different final solutions
3. The component can be perturbed or replaced to explore the solution space

Respond with exactly this JSON structure (no markdown):

If ILS-able:
{{
  "is_ils_able": true,
  "component": {{
    "name": "<short name for the component>",
    "description": "<1-2 sentence description of what it does>",
    "type": "<continuous|discrete|constrained>",
    "location": "<function name and approximate line range>",
    "degrees_of_freedom": "<what can be perturbed, e.g. x,y coordinates of N centers>"
  }},
  "reasoning": "<why this component is ILS-able>"
}}

If not ILS-able:
{{
  "is_ils_able": false,
  "reasoning": "<why the algorithm is not ILS-able>"
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
