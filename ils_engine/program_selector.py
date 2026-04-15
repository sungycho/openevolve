"""
Step 0: Top-K program selection from OpenEvolve output
"""

import json
import logging
import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _normalized_edit_distance(a: str, b: str) -> float:
    """Compute normalized edit distance (0=identical, 1=completely different)."""
    if not a and not b:
        return 0.0
    matcher = SequenceMatcher(None, a, b)
    similarity = matcher.ratio()
    return 1.0 - similarity


def _get_combined_score(program) -> float:
    """Extract combined_score from a Program's metrics, falling back to avg."""
    if "combined_score" in program.metrics:
        return program.metrics["combined_score"]
    numeric = [v for v in program.metrics.values() if isinstance(v, (int, float))]
    return sum(numeric) / max(1, len(numeric)) if numeric else 0.0


def _load_programs_from_db(path: str):
    """Load all programs from a checkpoint directory using ProgramDatabase."""
    from openevolve.database import ProgramDatabase
    from openevolve.config import DatabaseConfig

    programs_dir = os.path.join(path, "programs")
    if not os.path.isdir(programs_dir):
        raise FileNotFoundError(f"No programs/ directory found at {path}")

    db_config = DatabaseConfig(db_path=path, in_memory=False)
    db = ProgramDatabase(db_config)
    return list(db.programs.values())


def _load_best_program_from_dir(path: str):
    """Load just the best program from a best/ or checkpoint directory."""
    from openevolve.database import Program

    # Support both Python and C++ best programs
    program_file = None
    for candidate in ("best_program.py", "best_program.cpp"):
        candidate_path = os.path.join(path, candidate)
        if os.path.exists(candidate_path):
            program_file = candidate_path
            break

    info_file = os.path.join(path, "best_program_info.json")

    if program_file is None:
        raise FileNotFoundError(f"No best_program.py or best_program.cpp found at {path}")

    with open(program_file, "r") as f:
        code = f.read()

    metrics = {}
    program_id = "best"
    if os.path.exists(info_file):
        with open(info_file, "r") as f:
            info = json.load(f)
        metrics = info.get("metrics", {})
        program_id = info.get("id", "best")

    return Program(id=program_id, code=code, metrics=metrics)


def select_programs(
    output_path: str,
    top_k: int = 1,
    selection_mode: str = "novelty_rejection",
    novelty_threshold: float = 0.3,
) -> List:
    """
    Select top-K programs from OpenEvolve output.

    Args:
        output_path: Path to checkpoint directory or best/ folder
        top_k: Maximum number of programs to return
        selection_mode: "top_1" | "vanilla_top_k" | "novelty_rejection"
        novelty_threshold: Min edit distance for novelty_rejection mode

    Returns:
        List of Program objects (max top_k, ordered by score)
    """
    output_path = str(Path(output_path).resolve())

    # Detect path type
    has_programs_dir = os.path.isdir(os.path.join(output_path, "programs"))
    has_best_program = os.path.exists(os.path.join(output_path, "best_program.py"))

    # top_1 mode or best/ folder without programs/
    if selection_mode == "top_1" or (not has_programs_dir and has_best_program):
        logger.info("Loading single best program from directory")
        program = _load_best_program_from_dir(output_path)
        return [program]

    if not has_programs_dir:
        # Check for checkpoints/ subdirectory → use latest checkpoint
        checkpoints_dir = os.path.join(output_path, "checkpoints")
        if os.path.isdir(checkpoints_dir):
            checkpoints = sorted(
                [d for d in os.listdir(checkpoints_dir) if d.startswith("checkpoint_")],
                key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0,
            )
            if checkpoints:
                output_path = os.path.join(checkpoints_dir, checkpoints[-1])
                logger.info(f"Using latest checkpoint: {output_path}")
                has_programs_dir = os.path.isdir(os.path.join(output_path, "programs"))

    if not has_programs_dir:
        raise FileNotFoundError(
            f"Cannot find programs at {output_path}. "
            "Expected either a checkpoint dir with programs/ or a best_program.py file."
        )

    logger.info(f"Loading programs from database at {output_path}")
    all_programs = _load_programs_from_db(output_path)

    if not all_programs:
        raise ValueError(f"No programs found in database at {output_path}")

    # Sort all programs by score descending
    all_programs.sort(key=_get_combined_score, reverse=True)
    logger.info(
        f"Loaded {len(all_programs)} programs, top score: {_get_combined_score(all_programs[0]):.6f}"
    )

    if selection_mode == "vanilla_top_k":
        return all_programs[:top_k]

    # novelty_rejection mode
    selected = []
    for candidate in all_programs:
        if len(selected) >= top_k:
            break
        if not selected:
            selected.append(candidate)
            continue
        # Check novelty against already selected
        min_dist = min(_normalized_edit_distance(candidate.code, s.code) for s in selected)
        if min_dist >= novelty_threshold:
            selected.append(candidate)
            logger.debug(
                f"Selected program {candidate.id} (score={_get_combined_score(candidate):.4f}, "
                f"min_dist={min_dist:.3f})"
            )
        else:
            logger.debug(f"Rejected program {candidate.id} (too similar, min_dist={min_dist:.3f})")

    logger.info(f"Selected {len(selected)} programs via {selection_mode}")
    return selected
