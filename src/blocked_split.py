"""Blocked train/val/test split with gaps (WRITEUP.md section 1.1) -> data/<name>_split_transfer.npz (differs from src/make_split.py's split).

The trajectory is cut into B contiguous blocks; whole blocks go to train / val / test using a fixed
shuffled block order (so all sets cover early and late parts of the run). Validation and test frames
closer than G frames to any training frame are dropped, so no evaluation frame has a near-duplicate
in training.

Saved keys: train_pool, val, test (frame indices, sorted), plus B, G, block_set (set id per block:
0 train, 1 val, 2 test).

usage: ./.venv/bin/python src/blocked_split.py data/md22_AT-AT-CG-CG.npz [--blocks 20 --gap 50 --nval 2 --ntest 4]
"""
import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--blocks", type=int, default=20)
ap.add_argument("--gap", type=int, default=50)
ap.add_argument("--nval", type=int, default=2, help="blocks for validation")
ap.add_argument("--ntest", type=int, default=4, help="blocks for test")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

N = len(np.load(a.data)["E"])
edges = np.linspace(0, N, a.blocks + 1).round().astype(int)
order = np.random.default_rng(a.seed).permutation(a.blocks)
block_set = np.zeros(a.blocks, dtype=int)
block_set[order[:a.ntest]] = 2
block_set[order[a.ntest:a.ntest + a.nval]] = 1
frame_set = np.repeat(block_set, np.diff(edges))

train = np.flatnonzero(frame_set == 0)
# distance (in frames) from every frame to the nearest training frame
pos = np.searchsorted(train, np.arange(N)).clip(1, len(train) - 1)
dist = np.minimum(np.abs(np.arange(N) - train[pos - 1]), np.abs(np.arange(N) - train[pos]))
keep = dist >= a.gap
val = np.flatnonzero((frame_set == 1) & keep)
test = np.flatnonzero((frame_set == 2) & keep)

out = Path(a.data).with_name(Path(a.data).stem + "_split_transfer.npz")
np.savez(out, train_pool=train, val=val, test=test, B=a.blocks, G=a.gap, block_set=block_set)
print(f"{a.data}: {N} frames, {a.blocks} blocks of ~{N // a.blocks}")
print(f"  block sets (0 train, 1 val, 2 test): {block_set.tolist()}")
print(f"  train pool {len(train)}, val {len(val)}, test {len(test)} "
      f"(dropped {(frame_set > 0).sum() - len(val) - len(test)} eval frames within {a.gap} of training)")
print(f"  min eval-to-train distance: {dist[np.r_[val, test]].min()} frames -> {out}")
