import json
import matplotlib.pyplot as plt
import pathlib

# Load best program info
info_path = "examples/circle_packing/openevolve_output/best/best_program_info.json"
with open(info_path) as f:
    best = json.load(f)

print("Best program metrics:")
for k, v in best["metrics"].items():
    print(f"  {k}: {v}")

# If you want to visualize the best packing
import sys
sys.path.append("examples/circle_packing")
from best_program import run_packing, visualize

centers, radii, sum_radii = run_packing()
print(f"Sum radii from best program: {sum_radii}")
visualize(centers, radii)