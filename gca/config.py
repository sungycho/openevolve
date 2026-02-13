"""
GCA v0 — Configuration

Layer 8 of the GCA plan.  YAML-driven, opt-in, future-proof.

Example YAML block (to be added under an OpenEvolve config)::

    gca:
      enabled: true
      store_path: "./gca_store"
      extractor:
        type: "llm_solution_analysis"
        max_strategies: 3
        max_code_chars: 12000
        max_artifact_chars: 4000
      policy:
        type: "mvp"
        early_iterations: 25
        improvement_threshold: 0.05
        top_k_window: 5
      async:
        enabled: true
        queue_size: 128
        workers: 1
      autosave:
        interval_events: 50
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GCAExtractorConfig:
    """Configuration for the strategy extractor."""

    type: str = "llm_solution_analysis"   # "llm_solution_analysis" | "heuristic"
    model: Optional[str] = None           # LLM model override (None = use OE default)
    max_strategies: int = 3
    max_code_chars: int = 12_000
    max_artifact_chars: int = 4_000


@dataclass
class GCAPolicyConfig:
    """Configuration for the extraction trigger policy."""

    type: str = "mvp"                     # "mvp" | "always" | "never"
    early_iterations: int = 25
    improvement_threshold: float = 0.05
    top_k_window: int = 5
    recent_window_size: int = 50


@dataclass
class GCAAsyncConfig:
    """Configuration for async extraction worker."""

    enabled: bool = True
    queue_size: int = 128
    workers: int = 1                      # v0 always uses 1 worker


@dataclass
class GCAAutosaveConfig:
    """Configuration for periodic metadata saves."""

    interval_events: int = 50


@dataclass
class GCAFeedbackConfig:
    """Configuration for strategy feedback into the LLM prompt."""

    enabled: bool = False
    top_k: int = 3


@dataclass
class GCAConfig:
    """Top-level GCA configuration.

    Can be nested under OpenEvolve's ``Config`` or loaded standalone.
    """

    enabled: bool = True
    store_path: str = "./gca_store"
    run_id: Optional[str] = None          # auto-generated if None

    extractor: GCAExtractorConfig = field(default_factory=GCAExtractorConfig)
    policy: GCAPolicyConfig = field(default_factory=GCAPolicyConfig)
    async_config: GCAAsyncConfig = field(default_factory=GCAAsyncConfig)
    autosave: GCAAutosaveConfig = field(default_factory=GCAAutosaveConfig)
    feedback: GCAFeedbackConfig = field(default_factory=GCAFeedbackConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GCAConfig:
        """Build a GCAConfig from a plain dict (e.g., parsed YAML)."""
        if not data:
            return cls()

        extractor_data = data.get("extractor", {})
        policy_data = data.get("policy", {})
        async_data = data.get("async", data.get("async_config", {}))
        autosave_data = data.get("autosave", {})
        feedback_data = data.get("feedback", {})

        return cls(
            enabled=data.get("enabled", False),
            store_path=data.get("store_path", "./gca_store"),
            run_id=data.get("run_id"),
            extractor=GCAExtractorConfig(**{
                k: v for k, v in extractor_data.items()
                if k in GCAExtractorConfig.__dataclass_fields__
            }),
            policy=GCAPolicyConfig(**{
                k: v for k, v in policy_data.items()
                if k in GCAPolicyConfig.__dataclass_fields__
            }),
            async_config=GCAAsyncConfig(**{
                k: v for k, v in async_data.items()
                if k in GCAAsyncConfig.__dataclass_fields__
            }),
            autosave=GCAAutosaveConfig(**{
                k: v for k, v in autosave_data.items()
                if k in GCAAutosaveConfig.__dataclass_fields__
            }),
            feedback=GCAFeedbackConfig(**{
                k: v for k, v in feedback_data.items()
                if k in GCAFeedbackConfig.__dataclass_fields__
            }),
        )

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
