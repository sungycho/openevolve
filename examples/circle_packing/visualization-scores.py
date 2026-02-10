import json
import pathlib

import matplotlib.pyplot as plt

# --- Configure paths here ---------------------------------------------------
NO_GCA_PHASE1_DIR = pathlib.Path("examples/circle_packing/gca-specific-results/no_gca_phase1")
NO_GCA_PHASE2_DIR = pathlib.Path("examples/circle_packing/gca-specific-results/no_gca_phase2")
WITH_GCA_PHASE1_DIR = pathlib.Path("examples/circle_packing/gca-specific-results/with_gca_phase1")
WITH_GCA_PHASE2_DIR = pathlib.Path("examples/circle_packing/gca-specific-results/with_gca_phase2")
WITH_GCA_PHASE2_FROM_NO_GCA_PHASE1_DIR = pathlib.Path(
    "examples/circle_packing/gca-specific-results/with_gca_phase2_from_no_gca_phase1"
)
NO_GCA_PHASE2_FROM_WITH_GCA_PHASE1_DIR = pathlib.Path(
    "examples/circle_packing/gca-specific-results/no_gca_phase2_from_with_gca_phase1"
)

METRIC = "sum_radii"
PHASE1_END = 100
PHASE2_START = 101
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


def build_condition_curve(phase1_dir: pathlib.Path, phase2_dir: pathlib.Path) -> tuple[list[int], list[float]]:
    p1_ckpt = latest_checkpoint_dir(phase1_dir)
    p2_ckpt = latest_checkpoint_dir(phase2_dir)

    p1_by_iter = load_best_per_iteration(p1_ckpt, METRIC)
    p2_by_iter = load_best_per_iteration(p2_ckpt, METRIC)

    x1, y1 = build_running_best(p1_by_iter, 1, PHASE1_END)
    initial_best = y1[-1] if y1 else None

    phase2_end = max([i for i in p2_by_iter.keys() if i >= PHASE2_START], default=PHASE1_END)
    x2, y2 = build_running_best(p2_by_iter, PHASE2_START, phase2_end, initial_best=initial_best)

    return x1 + x2, y1 + y2


def main() -> None:
    x_no, y_no = build_condition_curve(NO_GCA_PHASE1_DIR, NO_GCA_PHASE2_DIR)
    x_yes, y_yes = build_condition_curve(WITH_GCA_PHASE1_DIR, WITH_GCA_PHASE2_DIR)
    x_no_to_yes, y_no_to_yes = build_condition_curve(
        NO_GCA_PHASE1_DIR,
        WITH_GCA_PHASE2_FROM_NO_GCA_PHASE1_DIR,
    )
    x_yes_to_no, y_yes_to_no = build_condition_curve(
        WITH_GCA_PHASE1_DIR,
        NO_GCA_PHASE2_FROM_WITH_GCA_PHASE1_DIR,
    )

    plt.figure(figsize=(11, 5.5))

    # Four red-family lines: two baseline runs + two crossover runs.
    plt.plot(x_no, y_no, color="darkred", linewidth=2.2, label="No GCA (P1) -> No GCA (P2)")
    plt.plot(x_yes, y_yes, color="red", linewidth=2.2, linestyle="--", label="With GCA (P1) -> With GCA (P2)")
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
    x_max = max(
        max(x_no, default=PHASE1_END),
        max(x_yes, default=PHASE1_END),
        max(x_no_to_yes, default=PHASE1_END),
        max(x_yes_to_no, default=PHASE1_END),
    )
    plt.axvspan(1, PHASE1_END, color="gray", alpha=0.08, label="Phase 1")
    plt.axvspan(PHASE2_START, x_max, color="gray", alpha=0.03, label="Phase 2")
    plt.axvline(PHASE1_END + 0.5, color="black", linestyle=":", linewidth=1)

    plt.xlabel("Iteration")
    plt.ylabel(METRIC)
    plt.title("Best-So-Far Score by Iteration (Phase 1 + Phase 2)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
