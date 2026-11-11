"""Accuracy vs training cost for the AT-AT-CG-CG edge-budget sweep -> runs/edges_atcg/cost_vs_error.png

Cost = seconds per training epoch (2000 configs, batch 16) measured one run at a time on an idle M4 Pro
GPU (MPS), 3 epochs each (the sweep's own epoch times are contaminated by jobs sharing the machine).
Error = test force MAE after the full 800-epoch WSD schedule.

usage: ./.venv/bin/python src/plot_cost.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("runs/edges_atcg/s0")
SEC = {"cutoff_rc3": 3.2, "cutoff_rc4": 4.6, "cutoff_rc5": 6.7, "cutoff_rc6": 8.8, "k4": 4.5, "k8": 5.4}
METHOD = {"teacher": "#eda100", "nearest": "#1baf7a", "random": "#e87ba4"}
err = {p.name: json.load(open(p / "test.json"))["f_mae"] for p in ROOT.iterdir() if (p / "test.json").exists()}

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
cut = [f"cutoff_rc{r}" for r in (3, 4, 5, 6)]
ax.plot([SEC[c] for c in cut], [err[c] for c in cut], "--", color="#5e5c56", lw=2, marker="o", ms=8, zorder=2,
        label="plain cutoff")
for c in cut:
    ax.annotate(f"{c[-1]} Å", (SEC[c], err[c]), xytext=(8, -14) if c.endswith("4") else (6, 6), textcoords="offset points", color="#3d3c38", fontsize=9)
for meth, col in METHOD.items():
    pts = [(SEC[k], err[f"{meth}_{k}"], k) for k in ("k4", "k8") if f"{meth}_{k}" in err]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col, lw=2, marker="o", ms=8, zorder=3,
            mec="white", mew=1.5, label=f"3 Å + {meth}")
    off = {"teacher": (0, -16), "nearest": (0, 9), "random": (-10, 8)}[meth]
    for x, y, k in pts:
        ax.annotate(k.replace("k", "k="), (x, y), xytext=off, textcoords="offset points", color="#3d3c38",
                    fontsize=9, ha="center" if meth != "random" else "right")
ax.set_xlabel("seconds per training epoch (idle GPU)")
ax.set_ylabel("test force MAE (kcal/mol/Å)")
ax.set_title("AT-AT-CG-CG: accuracy vs cost (800 epochs, seed 0)", loc="left", fontweight="bold")
ax.grid(alpha=0.3)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper right")
fig.text(0.01, 0.01, "k = extra edges per atom on average (3-6 Å pool, ranked with a 6 Å teacher). Lower-left is better.",
         fontsize=8, color="#5e5c56")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("runs/edges_atcg/cost_vs_error.png")
print("wrote runs/edges_atcg/cost_vs_error.png")
