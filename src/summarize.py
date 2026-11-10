"""Summarize a run directory: table of best/test metrics + validation force-MAE curves.

usage: python src/summarize.py runs/h1
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/h1")
rows = []
fig, ax = plt.subplots(figsize=(8, 5))
for d in sorted(p for p in root.iterdir() if (p / "log.csv").exists()):
    log = list(csv.DictReader(open(d / "log.csv")))
    if not log:
        continue
    t = [float(r["time_h"]) for r in log]
    f = [float(r["val_f_mae"]) for r in log]
    ax.plot(range(1, len(f) + 1), f, label=d.name)  # by epoch: runs share CPU/GPU unevenly
    last = log[-1]
    test = json.load(open(d / "test.json")) if (d / "test.json").exists() else {}
    rows.append({
        "run": d.name, "epochs": str(len(log)), "hours": f"{t[-1]:.2f}",
        "best_val_F": f"{min(f):.3f}",
        "test_F": f"{test['f_mae']:.3f}" if test else "-",
        "test_E": f"{test['e_mae']:.3f}" if test else "-",
        "short/mol": f"{float(last['n_short_per_mol']):.0f}",
        "long/mol": f"{float(last['n_long_per_mol']):.0f}",
        "gate_mean": last.get("val_gate_mean", "")[:5] or "-",
        "g>0.1": last.get("val_gate_frac>0.1", "")[:5] or "-",
        "g>0.01": last.get("val_gate_frac>0.01", "")[:5] or "-",
    })
ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_xscale("log"); ax.set_ylabel("val force MAE (kcal/mol/Å)")
ax.set_title(root.name); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(root / "val_curves.png", dpi=130)

keys = list(rows[0].keys())
w = [max(len(k), *(len(r[k]) for r in rows)) for k in keys]
print("  ".join(k.ljust(x) for k, x in zip(keys, w)))
for r in rows:
    print("  ".join(r[k].ljust(x) for k, x in zip(keys, w)))
print(f"\nplot: {root / 'val_curves.png'}")
