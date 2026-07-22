"""Build the checked-in optimization benchmark figure from aggregate values."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).resolve().parents[1] / "docs/figures/optimization_runtime_memory.png"

labels = ["1 trait", "4 traits", "20 traits\nstreamed"]
baseline_runtime = np.array([5.598, 11.374, 44.531])
domino_runtime = np.array([4.944, 5.276, 6.745])
baseline_rss = np.array([150.0, 178.0, 315.9])
domino_rss = np.array([154.5, 167.0, 204.0])

x = np.arange(len(labels))
width = 0.36
colors = {"baseline": "#767676", "domino": "#0072B2"}
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

for axis, baseline, current, ylabel, title in (
    (axes[0], baseline_runtime, domino_runtime, "Wall time (seconds)", "Runtime"),
    (axes[1], baseline_rss, domino_rss, "Peak RSS increment (MiB)", "Memory"),
):
    left = axis.bar(x - width / 2, baseline, width, color=colors["baseline"], label="Reference")
    right = axis.bar(x + width / 2, current, width, color=colors["domino"], label="Domino")
    axis.set_xticks(x, labels)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dddddd", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.bar_label(left, fmt="%.1f", padding=2, fontsize=8)
    axis.bar_label(right, fmt="%.1f", padding=2, fontsize=8)

axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Exact LOCO benchmark: 1,000 samples and 10,000 variants", fontsize=12)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=180, facecolor="white")
print(OUTPUT)
