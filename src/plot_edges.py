"""Validation force MAE vs epoch for the edge-budget sweep.

usage: ./.venv/bin/python src/plot_edges.py [runs/edges_atat]  -> <root>/val_curves.png
Grey dashed = plain cutoff graphs (light 3 Å -> dark 5 Å); colours = edge-selection method on top of
the 3 Å graph; line style = extra edges per atom (k). Seed 1 drawn fainter.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/edges_atat")
METHOD = {"gap": "#eb6834", "nearest": "#1baf7a", "teacher": "#eda100", "random": "#e87ba4"}
CUT = {"3": "#c9c7bf", "4": "#9a988f", "5": "#5e5c56", "6": "#1f1e1c"}
KSTYLE = {"1": ":", "2": "-", "4": "-."}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.labelcolor": INK})
fig, ax = plt.subplots(figsize=(10, 6))
for d in sorted(root.glob("s*/*")):
    if not (d / "log.csv").exists():
        continue
    rows = list(csv.DictReader(open(d / "log.csv")))
    if not rows:
        continue
    ep = [int(r["epoch"]) + 1 for r in rows]
    f = [float(r["val_f_mae"]) for r in rows]
    seed, name = d.parent.name, d.name
    alpha = 1.0 if seed == "s0" else 0.45
    if name.startswith("cutoff_rc"):
        rc = name[len("cutoff_rc"):]
        kw = dict(color=CUT[rc], ls="--", lw=2, label=f"cutoff {rc} Å" if seed == "s0" else None)
    else:
        meth, k = name.split("_k")
        kw = dict(color=METHOD[meth], ls=KSTYLE.get(k, "-"), lw=2,
                  label=f"3 Å + {meth}, k={k}" if seed == "s0" else None)
    ax.plot(ep, f, alpha=alpha, **kw)
    ax.annotate(f"{min(f):.2f}", (ep[-1], f[-1]), xytext=(4, 0), textcoords="offset points",
                fontsize=7, color=MUTED, va="center")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_yticks([1, 1.5, 2, 3, 5, 10, 20]); ax.set_yticklabels(["1", "1.5", "2", "3", "5", "10", "20"])
ax.set_xlabel("epoch (lr 5e-4 constant, then linear decay to 1e-6 at the end)"); ax.set_ylabel("validation force MAE (kcal/mol/Å)")
ax.grid(color=GRID, which="both", lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
ax.set_title(f"Edge budget: 3 Å graph + k selected edges per atom vs plain cutoffs ({root.name})",
             loc="left", fontsize=11, weight="bold")
fig.text(0.01, 0.01, "Numbers = best validation MAE so far. Seed 1 runs drawn fainter. "
         "k = extra edges per atom on average.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(root / "val_curves.png", dpi=150)
print("wrote", root / "val_curves.png")
