import json
import pathlib

import matplotlib.pyplot as plt

# --- Configure paths here ---------------------------------------------------
RESULT_ROOT = pathlib.Path("examples/circle_packing/gca-v1-feedback-four-mix-results")
NO_GCA_PHASE1_DIR = RESULT_ROOT / "phase1_no_gca"
WITH_GCA_PHASE1_DIR = RESULT_ROOT / "phase1_with_gca"
MIX_NO_P1_TO_NO_P2_DIR = RESULT_ROOT / "mix_no_p1_to_no_p2"
MIX_NO_P1_TO_WITH_P2_DIR = RESULT_ROOT / "mix_no_p1_to_with_p2"
MIX_WITH_P1_TO_NO_P2_DIR = RESULT_ROOT / "mix_with_p1_to_no_p2"
MIX_WITH_P1_TO_WITH_P2_DIR = RESULT_ROOT / "mix_with_p1_to_with_p2"

METRIC = "sum_radii"
# -----------------------------------------------------------------------------


def latest_checkpoint_dir(run_dir: pathlib.Path) -> pathlib.Path:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints = [p for p in checkpoints_dir.glob("checkpoint_*") if p.is_dir()]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return sorted(checkpoints, key=lambda p: int(p.name.split("_")[-1]))[-1]


def load_best_per_iteration(checkpoint_dir: pathlib.Path, metric: str) -> dict[int, float]:
    programs_dir = checkpoint_dir / "programs"
    if not programs_dir.exists():
        raise FileNotFoundError(f"Missing programs directory: {programs_dir}")

    best: dict[int, float] = {}
    for fp in programs_dir.glob("*.json"):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        it = data.get("iteration_found")
        metrics = data.get("metrics") or {}
        score = metrics.get(metric)
        if it is None or not isinstance(score, (int, float)):
            continue
        i = int(it)
        s = float(score)
        best[i] = max(s, best.get(i, float("-inf")))
    return best


def build_running_best(
    by_iter: dict[int, float],
    start_iter: int,
    end_iter: int,
    initial_best: float | None = None,
) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    best = initial_best if initial_best is not None else float("-inf")

    for i in range(start_iter, end_iter + 1):
        if i in by_iter:
            best = max(best, by_iter[i])
        if best != float("-inf"):
            xs.append(i)
            ys.append(best)
    return xs, ys


def shift_phase2_iterations(
    phase2_by_iter: dict[int, float],
    phase1_end: int,
) -> dict[int, float]:
    if not phase2_by_iter:
        return {}

    p2_min = min(phase2_by_iter)
    p2_max = max(phase2_by_iter)

    # If phase2 run restarted at 0/1 (common when starting from best_program.py),
    # shift its x-axis to continue after phase1.
    if p2_max <= phase1_end or p2_min <= phase1_end:
        return {i + phase1_end: v for i, v in phase2_by_iter.items()}
    return phase2_by_iter


def build_condition_curve(
    phase1_dir: pathlib.Path,
    phase2_dir: pathlib.Path,
) -> tuple[list[int], list[float], int]:
    p1_ckpt = latest_checkpoint_dir(phase1_dir)
    p2_ckpt = latest_checkpoint_dir(phase2_dir)

    p1_by_iter = load_best_per_iteration(p1_ckpt, METRIC)
    p2_by_iter = load_best_per_iteration(p2_ckpt, METRIC)

    if not p1_by_iter:
        return [], [], 0

    phase1_start = min(p1_by_iter)
    phase1_end = max(p1_by_iter)
    x1, y1 = build_running_best(p1_by_iter, phase1_start, phase1_end)
    initial_best = y1[-1] if y1 else None

    p2_display = shift_phase2_iterations(p2_by_iter, phase1_end)
    if p2_display:
        phase2_start = min(p2_display)
        phase2_end = max(p2_display)
        x2, y2 = build_running_best(p2_display, phase2_start, phase2_end, initial_best=initial_best)
    else:
        x2, y2 = [], []

    return x1 + x2, y1 + y2, phase1_end


def main() -> None:
    x_no, y_no, p1_end_no = build_condition_curve(
        NO_GCA_PHASE1_DIR,
        MIX_NO_P1_TO_NO_P2_DIR,
    )
    x_no_to_yes, y_no_to_yes, _ = build_condition_curve(
        NO_GCA_PHASE1_DIR,
        MIX_NO_P1_TO_WITH_P2_DIR,
    )
    x_yes_to_no, y_yes_to_no, p1_end_yes = build_condition_curve(
        WITH_GCA_PHASE1_DIR,
        MIX_WITH_P1_TO_NO_P2_DIR,
    )
    x_yes, y_yes, _ = build_condition_curve(
        WITH_GCA_PHASE1_DIR,
        MIX_WITH_P1_TO_WITH_P2_DIR,
    )

    plt.figure(figsize=(11, 5.5))

    # Four red-family lines: two baseline runs + two crossover runs.
    plt.plot(x_no, y_no, color="darkred", linewidth=2.2, label="No GCA (P1) -> No GCA (P2)")
    plt.plot(
        x_yes,
        y_yes,
        color="red",
        linewidth=2.2,
        linestyle="--",
        label="With GCA (P1) -> With GCA (P2)",
    )
    plt.plot(
        x_no_to_yes,
        y_no_to_yes,
        color="firebrick",
        linewidth=2.0,
        linestyle="-.",
        label="No GCA (P1) -> With GCA (P2)",
    )
    plt.plot(
        x_yes_to_no,
        y_yes_to_no,
        color="salmon",
        linewidth=2.0,
        linestyle=":",
        label="With GCA (P1) -> No GCA (P2)",
    )

    # Visual phase split on x-axis.
    phase1_end = max(p1_end_no, p1_end_yes)
    x_min = min(
        min(x_no, default=0),
        min(x_yes, default=0),
        min(x_no_to_yes, default=0),
        min(x_yes_to_no, default=0),
    )
    x_max = max(
        max(x_no, default=phase1_end),
        max(x_yes, default=phase1_end),
        max(x_no_to_yes, default=phase1_end),
        max(x_yes_to_no, default=phase1_end),
    )
    phase2_start = phase1_end + 1
    plt.axvspan(x_min, phase1_end, color="gray", alpha=0.08, label="Phase 1")
    plt.axvspan(phase2_start, x_max, color="gray", alpha=0.03, label="Phase 2")
    plt.axvline(phase1_end + 0.5, color="black", linestyle=":", linewidth=1)

    plt.xlabel("Iteration")
    plt.ylabel(METRIC)
    plt.title("Best-So-Far Score by Iteration (GCA v1 Four Mixes)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
