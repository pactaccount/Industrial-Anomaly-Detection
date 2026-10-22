"""Render image-AUROC chart from the published 15-category JSON summary."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
report = json.loads((root / "docs/benchmark_summary.json").read_text())
rows = sorted(report["categories"], key=lambda item: item["image_auroc"])
plt.rcParams["svg.fonttype"] = "none"
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([row["name"] for row in rows], [row["image_auroc"] for row in rows], color="#0f766e")
ax.axvline(report["mean_image_auroc"], color="#ea580c", linestyle="--", label=f"Mean {report['mean_image_auroc']:.3f}")
ax.set_xlim(0, 1.05)
ax.set_xlabel("Image AUROC")
ax.set_title("Argus · MVTec AD image AUROC by category")
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, 0.01, "Public benchmark, not factory camera validation", ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(root / "docs/benchmark_chart.svg")
