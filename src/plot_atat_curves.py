"""Validation force MAE vs epoch for every AT-AT run (runs/h1 and runs/h2).

usage: ./.venv/bin/python src/plot_atat_curves.py   -> runs/atat_val_curves.png
Colour = receptive-field setting, line style = model type (cutoff solid, dense dashed, gated dotted).
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 (validated default palette)
STYLE = {"cutoff": "-", "dense": "--", "gated": ":"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def curve(d):
    rows = list(csv.DictReader(open(d / "log.csv")))
    test = json.load(open(d / "test.json")) if (d / "test.json").exists() else {}
    return [int(r["epoch"]) + 1 for r in rows], [float(r["val_f_mae"]) for r in rows], test.get("f_mae")


PANELS = [
    ("runs/h1", "h1: 3 layers, plateau LR schedule, 7.5 h per run",
     [("cutoff_rc3", "cutoff rc=3", 0), ("dense_rc3", "dense rc=3", 0),
      ("cutoff_rc5", "cutoff rc=5", 1), ("dense_rc5", "dense rc=5", 1),
      ("gated_rc5_lam0.1", "gated rc=5 λ=0.1", 1), ("gated_rc5_lam1", "gated rc=5 λ=1", 1),
      ("cutoff_rc10", "cutoff rc=10", 2)]),
    ("runs/h2", "h2: smaller receptive fields, cosine schedule (600 ep planned, stopped early)",
     [("cutoff_L2_rc3", "cutoff L=2 rc=3 (6 Å)", 0), ("dense_L2_rc3", "dense L=2 rc=3", 0),
      ("cutoff_L2_rc4", "cutoff L=2 rc=4 (8 Å)", 1), ("dense_L2_rc4", "dense L=2 rc=4", 1),
      ("cutoff_L3_rc3", "cutoff L=3 rc=3 (9 Å)", 2), ("dense_L3_rc3", "dense L=3 rc=3", 2),
      ("cutoff_L3_rc4", "cutoff L=3 rc=4 (12 Å)", 3), ("dense_L3_rc4", "dense L=3 rc=4", 3)]),
]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
for ax, (root, title, runs) in zip(axes, PANELS):
    for name, label, ci in runs:
        d = Path(root) / name
        if not (d / "log.csv").exists():
            continue
        ep, f, test = curve(d)
        mode = name.split("_")[0]
        lab = f"{label}  (test {test:.2f})" if test else label
        ax.plot(ep, f, STYLE[mode], color=C[ci], lw=2 if mode != "gated" or "0.1" in name else 1.4,
                alpha=1.0 if "lam1" not in name else 0.6, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(color=GRID, which="both", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
axes[0].set_ylabel("validation force MAE (kcal/mol/Å)")
axes[0].set_yticks([1, 2, 3, 5, 10, 20]); axes[0].set_yticklabels(["1", "2", "3", "5", "10", "20"])
fig.suptitle("AT-AT (60 atoms): force error over training", x=0.01, ha="left", fontsize=12, weight="bold")
fig.text(0.01, 0.005, "Dense/gated runs here used the old shared radial basis (coarse short-range resolution), "
         "fixed on Nov 10. Test MAE on 5000 held-out configurations in parentheses.",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
out = Path("runs/atat_val_curves.png")
fig.savefig(out, dpi=150)
print("wrote", out)
