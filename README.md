# Which atoms need to talk? Long-range edges and transfer in SchNet-style force fields (MD22)

CS224W project, Viktor Imrisek and Emiyare Ikwut-Ukwa.

> **⚠️ Warning:** results so far use a random frame split of a single MD trajectory, so test frames sit
> 2–3 steps from training frames and absolute errors are optimistic. A blocked split with gaps is
> now implemented (`src/make_split.py`, WRITEUP.md §1.1) and a rerun on it is in progress. `src/split_check.py` reproduces the leak numbers.

**Start with [WRITEUP.md](WRITEUP.md)** for the experiments so far, the results, and the proposed
direction (transfer from AT-AT to AT-AT-CG-CG). The original proposal is `CS224W_project__2_.pdf`.

## Layout

```
src/train.py             model (cutoff / dense / gated) + training loop, checkpoint/resume
src/prep_md22.py         MD22 .xyz or sGDML .npz -> data/<name>.npz (keys z, R, F, E)
src/fragments.py         covalent fragments (nucleobases) -> data/<name>_fragments.npy
src/summarize.py         table + validation curves for a run directory
src/plot_atat_curves.py  AT-AT curves figure (runs/atat_val_curves.png)
src/zeroshot.py          AT-AT model evaluated on AT-AT-CG-CG, errors per nucleobase
src/edge_select.py       influence (||dF_i/dr_j||) of teacher/student models -> ranked extra-edge lists
src/influence_explore.py exploratory influence-by-distance analysis on AT-AT-CG-CG
src/split_check.py       how close test frames are to training frames under the random split
src/make_split.py        blocked train/val/test split with gaps -> data/md22_<name>_split.npz (train.py --split)
src/plot_edges.py        validation curves for an edge-budget sweep (<root>/val_curves.png)
src/plot_cost.py         AT-AT-CG-CG accuracy vs time per epoch (runs/edges_atcg/cost_vs_error.png)
runs/edges_atat.py       job queue for the AT-AT edge-budget sweep (outputs in runs/edges_atat/)
runs/edges_atcg.py       job queue for the AT-AT-CG-CG edge-budget sweep (outputs in runs/edges_atcg/)
runs/edges_blocked.py    rerun of both sweeps on the blocked split, teachers included (runs/edges_blocked/)
data/edges/<molecule>/   influence matrices + selected edge lists (pairs_<method>_k<budget>.npy)
src/transfer.py          transfer AT-AT -> AT-AT-CG-CG: one scratch / finetune / frozen / zeroshot run at N labels
src/blocked_split.py     blocked split used by the transfer runs -> data/md22_AT-AT-CG-CG_split_transfer.npz
                         (a different block layout from src/make_split.py; to be rerun on that split)
src/summarize_transfer.py tables + error-vs-N figure for runs/transfer
runs/transfer_sweep.py   job queue for the transfer sweep (outputs in runs/transfer/, results in RESULTS.md)
runs/*.sh                launch scripts for the experiments
runs/h1, runs/h2         AT-AT headroom experiments
runs/atcg1               AT-AT-CG-CG headroom + gates (before the distance-encoding fix)
runs/atcg2               AT-AT-CG-CG after the fix (partial, from Colab)
colab/                   Colab notebook (+ generator that embeds the current train.py)
data/                    converted MD22 datasets (AT-AT, AT-AT-CG-CG, Ac-Ala3-NHMe) + fragment labels
```

Each run directory has `args.json` (exact arguments), `log.csv` (per-epoch metrics), `test.json`
(final test metrics, when finished), `best.pt` (best-validation weights), `last.pt` (resume state) and,
for the WSD sweeps, `stable_ep<N>.pt` (full state at epoch N, before the LR decay).
Note: dense/gated checkpoints in `runs/h1`, `runs/h2` and `runs/atcg1` use the old shared distance
encoding and won't load into the current `train.py`; cutoff checkpoints load fine.

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

The converted datasets are in `data/`. To rebuild them from the original MD22 files
(http://www.sgdml.org/#datasets), download the `.xyz` or `.npz` and run
`./.venv/bin/python src/prep_md22.py <file>`, then `./.venv/bin/python src/fragments.py data/<name>.npz`.

## Examples

```bash
# train a cutoff model on AT-AT-CG-CG (device: cpu, mps or cuda)
./.venv/bin/python src/train.py --data data/md22_AT-AT-CG-CG.npz --mode cutoff --rc 5 \
  --ntrain 2000 --nval 500 --ntest 2000 --sched cosine --max_epochs 300 --device mps --out runs/test_cutoff

# dense / gated: add --mode dense --rl 24, or --mode gated --rl 24 --lam 3
# resume a stopped run: rerun the same command with --resume

./.venv/bin/python src/summarize.py runs/atcg1     # table + runs/atcg1/val_curves.png
./.venv/bin/python src/zeroshot.py                 # zero-shot transfer table
```

Tested with Python 3.13, PyTorch 2.5, NumPy 2.x on an M4 Pro Mac (MPS) and a Colab T4 (CUDA).
