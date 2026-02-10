"""
GCA v0 — Strategy extraction from evaluated programs

Layer 3 of the GCA plan.

The extractor analyses *solution code + evaluation outcomes* (not prompts)
and returns 0-N (up to ``max_strategies``, default 3) strategies when
clearly present.

Two extractors are provided:
  1. ``LLMSolutionStrategyExtractor`` — calls an LLM for rich analysis.
  2. ``HeuristicStrategyExtractor``   — regex/AST fallback (no LLM cost).
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from typing import Any, Protocol, runtime_checkable

from gca.schemas import (
    EvaluationSummary,
    ExtractedStrategy,
    ProgramRef,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol (Layer 3.1)
# ---------------------------------------------------------------------------

@runtime_checkable
class StrategyExtractor(Protocol):
    """Minimal interface that every extractor must satisfy."""

    name: str
    version: str

    async def extract_from_solution(
        self,
        program_code: str,
        program_ref: ProgramRef,
        evaluation: EvaluationSummary,
        artifacts: dict[str, Any],
    ) -> list[ExtractedStrategy]:
        ...


# ---------------------------------------------------------------------------
# Input-size management helpers (Layer 3.3)
# ---------------------------------------------------------------------------

def _truncate_code(code: str, max_chars: int = 12_000) -> str:
    """Keep imports + function signatures + critical blocks within budget."""
    if len(code) <= max_chars:
        return code

    lines = code.splitlines(keepends=True)
    head: list[str] = []
    budget = max_chars

    # Phase 1: keep imports and function/class signatures
    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("def ")
            or stripped.startswith("class ")
            or stripped.startswith("# EVOLVE-BLOCK")
        ):
            head.append(line)
            budget -= len(line)
            if budget <= 0:
                break

    # Phase 2: fill remaining budget with top-of-file content
    remaining = max_chars - sum(len(l) for l in head)
    if remaining > 200:
        body = "".join(lines)[:remaining]
        return body + "\n# ... [truncated] ...\n"

    return "".join(head) + "\n# ... [truncated] ...\n"


def _truncate_artifacts(artifacts: dict[str, Any], max_chars: int = 4_000) -> dict[str, str]:
    """Return a string-only copy of artifacts, truncated to budget."""
    result: dict[str, str] = {}
    budget = max_chars
    for key, value in artifacts.items():
        s = str(value)
        if len(s) > budget:
            s = s[:budget] + " ... [truncated]"
        result[key] = s
        budget -= len(s)
        if budget <= 0:
            break
    return result


# ---------------------------------------------------------------------------
# LLM-based extractor (Layer 3.2)
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a code strategy analyst.  Given a program and its evaluation
    results, identify **up to {max_strategies}** high-level strategies
    that are clearly present in the solution.

    A "strategy" is a *reusable, meso-level technique* — more abstract than
    a single line of code, but more concrete than "write good code".

    Output valid JSON — an array of objects (may be empty):
    [
      {{
        "description": "short natural-language description of the strategy",
        "confidence": 0.0-1.0,
        "rationale": "why you believe this strategy is present",
        "snippets": ["relevant code fragment (short)"]
      }}
    ]

    Rules:
    * Return an **empty array** if no clear strategies are present.
    * Do NOT invent strategies that are not evidenced by the code.
    * Keep descriptions concise (one sentence).
""")

_EXTRACTION_USER_TEMPLATE = textwrap.dedent("""\
    ## Program code
    ```
    {code}
    ```

    ## Evaluation summary
    - Fitness: {fitness}
    - Status: {status}
    - Metrics: {metrics}

    ## Artifacts (truncated)
    {artifacts}

    Identify up to {max_strategies} strategies present in this code.
""")


class LLMSolutionStrategyExtractor:
    """Calls an OpenAI-compatible LLM to extract strategies from evaluated
    programs (Layer 3.2).

    Parameters
    ----------
    llm_callable : async (system: str, user: str) -> str
        Any async function that takes a system prompt and user message and
        returns the LLM text response.  This keeps us decoupled from
        OpenEvolve's ``LLMInterface``.
    max_strategies : int
        Cap on strategies per extraction (default 3).
    max_code_chars / max_artifact_chars : int
        Token-budget guardrails (Layer 3.3).
    """

    name: str = "llm_solution_analysis"
    version: str = "0.1.0"

    def __init__(
        self,
        llm_callable,  # async (system: str, user: str) -> str
        *,
        max_strategies: int = 3,
        max_code_chars: int = 12_000,
        max_artifact_chars: int = 4_000,
    ) -> None:
        self._llm = llm_callable
        self._max_strategies = max_strategies
        self._max_code_chars = max_code_chars
        self._max_artifact_chars = max_artifact_chars

    async def extract_from_solution(
        self,
        program_code: str,
        program_ref: ProgramRef,
        evaluation: EvaluationSummary,
        artifacts: dict[str, Any],
    ) -> list[ExtractedStrategy]:
        truncated_code = _truncate_code(program_code, self._max_code_chars)
        truncated_arts = _truncate_artifacts(artifacts, self._max_artifact_chars)

        system_msg = _EXTRACTION_SYSTEM_PROMPT.format(max_strategies=self._max_strategies)
        user_msg = _EXTRACTION_USER_TEMPLATE.format(
            code=truncated_code,
            fitness=evaluation.fitness,
            status=evaluation.status,
            metrics=json.dumps(evaluation.metrics),
            artifacts=json.dumps(truncated_arts, indent=2),
            max_strategies=self._max_strategies,
        )

        try:
            raw_response = await self._llm(system_msg, user_msg)
            return self._parse_response(raw_response)
        except Exception:
            logger.exception("LLM strategy extraction failed")
            return []

    def _parse_response(self, raw: str) -> list[ExtractedStrategy]:
        """Parse JSON array from LLM output (tolerant of markdown fences)."""
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM extraction response as JSON")
            return []

        if not isinstance(items, list):
            items = [items]

        results: list[ExtractedStrategy] = []
        for item in items[: self._max_strategies]:
            if not isinstance(item, dict):
                continue
            desc = item.get("description", "").strip()
            if not desc:
                continue
            results.append(
                ExtractedStrategy(
                    description=desc,
                    confidence=float(item.get("confidence", 0.5)),
                    evidence={
                        "rationale": item.get("rationale", ""),
                        "snippets": item.get("snippets", []),
                    },
                    tags=item.get("tags", []),
                )
            )
        return results


# ---------------------------------------------------------------------------
# Heuristic (non-LLM) extractor — zero-cost fallback
# ---------------------------------------------------------------------------

class HeuristicStrategyExtractor:
    """Pattern-based extractor that identifies common algorithmic strategies
    from code structure alone.  Useful as a fallback when LLM calls are
    too expensive or unavailable.
    """

    name: str = "heuristic_structural"
    version: str = "0.1.0"

    _PATTERNS: list[tuple[str, str]] = [
        (r"\bmemoiz", "Uses memoization or caching to avoid redundant computation"),
        (r"\b(dp|dynamic.?programming)\b", "Applies dynamic programming"),
        (r"\bvectoriz", "Uses vectorized operations instead of loops"),
        (r"\bnumba|@jit\b", "Applies JIT compilation for performance"),
        (r"\bparallel|multiprocess|ThreadPool|ProcessPool", "Uses parallelism or concurrency"),
        (r"\bbinary.?search\b", "Uses binary search for efficient lookup"),
        (r"\bgreedy\b", "Uses a greedy algorithm approach"),
        (r"\brecursi", "Uses recursion"),
        (r"\bnp\.einsum|einsum", "Uses Einstein summation for tensor operations"),
        (r"\bsort\(|sorted\(", "Applies sorting as a preprocessing step"),
    ]

    async def extract_from_solution(
        self,
        program_code: str,
        program_ref: ProgramRef,
        evaluation: EvaluationSummary,
        artifacts: dict[str, Any],
    ) -> list[ExtractedStrategy]:
        results: list[ExtractedStrategy] = []
        for pattern, description in self._PATTERNS:
            matches = re.findall(pattern, program_code, re.IGNORECASE)
            if matches:
                results.append(
                    ExtractedStrategy(
                        description=description,
                        confidence=0.6,
                        evidence={
                            "rationale": f"Pattern '{pattern}' matched {len(matches)} time(s)",
                            "snippets": [m if isinstance(m, str) else m[0] for m in matches[:3]],
                        },
                    )
                )
            if len(results) >= 3:
                break
        return results
