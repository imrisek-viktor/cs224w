"""Summarize the experiment-B sweep (runs/transfer): tables + figure.

Prints (markdown):
  1. test force MAE vs N per method (mean ± std over finished seeds, seed count)
  2. per-nucleobase force MAE at each N (seed mean)
  3. labels needed: smallest N whose mean test error <= the from-scratch N=2000 reference
  4. training steps for fine-tuning to reach the scratch N=2000 best validation error
Writes runs/transfer/transfer_curves.png (test error vs N, log x; zero-shot and long references marked).

usage: ./.venv/bin/python src/summarize_transfer.py
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("runs/transfer")
R = defaultdict(dict)  # (method, N) -> {seed: test.json}
extra = {}
for d in sorted(OUT.iterdir()):
    t = d / "test.json"
    if not t.exists():
        continue
    m = re.fullmatch(r"(\w+?)_n(\d+)_s(\d+)(_long)?", d.name)
    if not m:
        continue
    res = json.load(open(t))
    if m[1] == "zeroshot" or m[4]:
        extra[d.name] = res
    else:
        R[(m[1], int(m[2]))][int(m[3])] = res

methods = [m for m in ["scratch", "finetune", "frozen"] if any(k[0] == m for k in R)]
Ns = sorted({k[1] for k in R})


def stat(m, n, f=lambda r: r["f_mae"]):
    v = [f(r) for r in R.get((m, n), {}).values()]
    return (np.mean(v), np.std(v), len(v)) if v else None


print("## Test force MAE (kcal/mol/Å), blocked split, mean ± std over seeds (n seeds)\n")
print("| N | " + " | ".join(methods) + " |")
print("|---|" + "---|" * len(methods))
for n in Ns:
    cells = []
    for m in methods:
        s = stat(m, n)
        cells.append(f"{s[0]:.2f} ± {s[1]:.2f} ({s[2]})" if s else "–")
    print(f"| {n} | " + " | ".join(cells) + " |")
for k, r in extra.items():
    print(f"\n{k}: test F {r['f_mae']:.2f}, best val F {r['best_val_f_mae']:.2f}, steps {r['steps']}")

print("\n## Per-nucleobase test force MAE (seed mean)\n")
print("| method | N | A | T | C | G |\n|---|---|---|---|---|---|")
for m in methods:
    for n in Ns:
        if (m, n) in R:
            pb = {b: np.mean([r["per_base"][b] for r in R[(m, n)].values()]) for b in "ATCG"}
            print(f"| {m} | {n} | " + " | ".join(f"{pb[b]:.2f}" for b in "ATCG") + " |")
if "zeroshot_n2000_s0" in extra:
    pb = extra["zeroshot_n2000_s0"]["per_base"]
    print("| zero-shot | 0 | " + " | ".join(f"{pb[b]:.2f}" for b in "ATCG") + " |")

ref = stat("scratch", 2000)
if ref:
    print(f"\n## Labels needed to match scratch at N=2000 (mean test {ref[0]:.2f})\n")
    for m in methods:
        hit = [n for n in Ns if stat(m, n) and stat(m, n)[0] <= ref[0]]
        print(f"- {m}: {'N = ' + str(min(hit)) if hit else 'not reached'}")

# steps for fine-tuning to reach scratch N=2000's best validation error (seed 0 logs)
sref = OUT / "scratch_n2000_s0" / "test.json"
if sref.exists():
    target = json.load(open(sref))["best_val_f_mae"]
    print(f"\n## Steps to reach scratch N=2000's best validation error ({target:.2f}), seed 0\n")
    for name in ["scratch_n2000_s0", "scratch_n2000_s0_long", "finetune_n2000_s0", "finetune_n2000_s0_long",
                 *[f"finetune_n{n}_s0" for n in Ns]]:
        f = OUT / name / "log.csv"
        if f.exists():
            rows = list(csv.DictReader(open(f)))
            hit = [int(r["step"]) for r in rows if float(r["val_f_mae"]) <= target]
            print(f"- {name}: {hit[0] if hit else 'not reached'} of {rows[-1]['step']} steps logged")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for m, c in zip(methods, ["#4C72B0", "#DD8452", "#55A868"]):
        pts = [(n, *stat(m, n)[:2]) for n in Ns if stat(m, n)]
        x, y, e = map(np.array, zip(*pts))
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, color=c, label=m)
    if "zeroshot_n2000_s0" in extra:
        ax.axhline(extra["zeroshot_n2000_s0"]["f_mae"], ls=":", c="gray", label="AT-AT model, no retraining")
    for k, mk in [("scratch_n2000_s0_long", "s"), ("finetune_n2000_s0_long", "D")]:
        if k in extra:
            ax.plot([2000], [extra[k]["f_mae"]], mk, mfc="none", c="k", label=k.replace("_n2000_s0_long", ", N=2000, 50k steps"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("AT-AT-CG-CG training configurations (N)"); ax.set_ylabel("test force MAE (kcal/mol/Å)")
    ax.set_title("Transfer AT-AT → AT-AT-CG-CG (blocked split)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(OUT / "transfer_curves.png", dpi=150)
    print(f"\nfigure -> {OUT / 'transfer_curves.png'}")
except Exception as e:  # noqa: BLE001
    print("plot skipped:", e)
