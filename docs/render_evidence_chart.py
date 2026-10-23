"""Render the fixed-threshold test counts from the published benchmark JSON."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"

root = Path(__file__).resolve().parent
report = json.loads((root / "metal_nut_benchmark.json").read_text())["reports"][0]
samples = report["samples"]
tp = sum(s["label"] == "anomaly" and s["predicted_anomaly"] for s in samples)
fn = sum(s["label"] == "anomaly" and not s["predicted_anomaly"] for s in samples)
fp = sum(s["label"] == "good" and s["predicted_anomaly"] for s in samples)
tn = sum(s["label"] == "good" and not s["predicted_anomaly"] for s in samples)

fig, ax = plt.subplots(figsize=(7.5, 3.3))
ax.barh(["Defective test parts", "Good test parts"], [tp, tn], color="#137c70", label="Correct")
ax.barh(["Defective test parts", "Good test parts"], [fn, fp], left=[tp, tn], color="#d96938", label="Incorrect")
for y, (correct, wrong) in enumerate(((tp, fn), (tn, fp))):
    ax.text(correct / 2, y, str(correct), ha="center", va="center", color="white", weight="bold")
    if wrong:
        ax.text(correct + wrong / 2, y, str(wrong), ha="center", va="center", color="white", weight="bold")
ax.set_xlim(0, max(tp + fn, tn + fp) * 1.08)
ax.set_xlabel("Held-out MVTec AD metal_nut images")
ax.set_title("Argus fixed-threshold decisions · ResNet-18 · 256 px")
ax.legend(loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, 0.01, "Research benchmark only; 22 good images make FPR imprecise", ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(root / "metal_nut_benchmark.svg")
