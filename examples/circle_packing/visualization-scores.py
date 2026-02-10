import re
import pathlib
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt

log_path = sorted(pathlib.Path("examples/circle_packing/openevolve_output/logs").glob("*.log"))[-1]

iterations = []
scores = []
best_so_far = []
current_iter = None

with open(log_path) as f:
    for line in f:
        # Match iteration number
        iter_match = re.search(r"Iteration (\d+):", line)
        if iter_match:
            current_iter = int(iter_match.group(1))

        # Match metrics line (separate from iteration line)
        metrics_match = re.search(r"Metrics:.*combined_score=([\d.]+)", line)
        if metrics_match and current_iter is not None:
            score = float(metrics_match.group(1))
            iterations.append(current_iter)
            scores.append(score)
            if best_so_far:
                best_so_far.append(max(best_so_far[-1], score))
            else:
                best_so_far.append(score)

print(f"Found {len(iterations)} data points from {log_path.name}")
if iterations:
    print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"Best score: {max(scores):.4f}")

plt.figure(figsize=(10, 5))
plt.plot(iterations, scores, marker='.', markersize=4, label="Iteration Score", alpha=0.5)
plt.plot(iterations, best_so_far, label="Best So Far", linewidth=2, color='red')
plt.xlabel("Iteration")
plt.ylabel("Combined Score")
plt.title("Evolution Progress: Scores and Running Best")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()