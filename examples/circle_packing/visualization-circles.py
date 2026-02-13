import importlib.util
import json
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

RESULT_ROOT = pathlib.Path("examples/circle_packing/gca-v1-feedback-four-mix-results")
RUNS = [
    ("No GCA P1 -> No GCA P2", RESULT_ROOT / "mix_no_p1_to_no_p2"),
    ("No GCA P1 -> With GCA P2", RESULT_ROOT / "mix_no_p1_to_with_p2"),
    ("With GCA P1 -> No GCA P2", RESULT_ROOT / "mix_with_p1_to_no_p2"),
    ("With GCA P1 -> With GCA P2", RESULT_ROOT / "mix_with_p1_to_with_p2"),
]


def load_run_packing(best_program_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("best_program_module", best_program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {best_program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_packing"):
        raise AttributeError(f"{best_program_path} does not define run_packing()")
    return module.run_packing


def plot_one(ax, label: str, run_dir: pathlib.Path) -> None:
    best_info_path = run_dir / "best" / "best_program_info.json"
    best_program_path = run_dir / "best" / "best_program.py"
    if not best_info_path.exists() or not best_program_path.exists():
        raise FileNotFoundError(f"Missing best artifacts in {run_dir}")

    best = json.loads(best_info_path.read_text())
    run_packing = load_run_packing(best_program_path)
    centers, radii, sum_radii = run_packing()

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{label}\nsum_radii={sum_radii:.6f}")
    ax.grid(True, alpha=0.2)

    for (x, y), r in zip(centers, radii):
        ax.add_patch(Circle((x, y), r, fill=False, edgecolor="tab:blue", linewidth=1.0))

    metrics = best.get("metrics") or {}
    if metrics:
        print(f"\n{label}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, (label, run_dir) in zip(axes.flatten(), RUNS):
        plot_one(ax, label, run_dir)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
