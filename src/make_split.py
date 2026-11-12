"""Blocked train/val/test split with gaps for an MD22 trajectory -> data/<name>_split.npz (keys tr, va, te).

The trajectory is cut into B contiguous blocks. Whole blocks go to train / val / test in a fixed pattern
spread over the run (test = every 5th block starting at block 2, val = blocks 4 and 14), so all three
sets cover early and late parts of the trajectory. Frames within G frames of a block of a different set
are dropped (on both sides), so no val/test frame is within 2G steps of a training frame (forces decorrelate within ~5
frames, see src/split_check.py). The training set is then subsampled uniformly to `ntrain` frames and the
validation set to `nval`. Every test frame is kept.

usage: ./.venv/bin/python src/make_split.py data/md22_AT-AT.npz --ntrain 3000 --nval 500
"""
import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--blocks", type=int, default=20)
ap.add_argument("--gap", type=int, default=50)
ap.add_argument("--ntrain", type=int, required=True)
ap.add_argument("--nval", type=int, default=500)
a = ap.parse_args()

N = len(np.load(a.data)["E"])
B = a.blocks
role = np.zeros(B, int)  # 0 train, 1 val, 2 test
role[2::5] = 2
role[[4, 14]] = 1
block = np.minimum(np.arange(N) * B // N, B - 1)
lab = role[block]

# drop frames within `gap` of a frame with a different label
keep = np.ones(N, bool)
change = np.nonzero(lab[1:] != lab[:-1])[0]  # boundary between frame c and c+1
for c in change:
    keep[max(0, c + 1 - a.gap):min(N, c + 1 + a.gap)] = False

rng = np.random.default_rng(1234)
tr_all = np.nonzero(keep & (lab == 0))[0]
va_all = np.nonzero(keep & (lab == 1))[0]
te = np.nonzero(keep & (lab == 2))[0]
tr = np.sort(rng.choice(tr_all, a.ntrain, replace=False))
va = np.sort(rng.choice(va_all, min(a.nval, len(va_all)), replace=False))

# check: nearest training frame for every val/test frame
def nearest(x, ref):
    i = np.clip(np.searchsorted(ref, x), 1, len(ref) - 1)
    return np.minimum(np.abs(x - ref[i - 1]), np.abs(x - ref[i]))

out = Path(a.data).parent / f"{Path(a.data).stem}_split.npz"
np.savez(out, tr=tr, va=va, te=te, role=role, gap=a.gap, blocks=B)
print(f"{Path(a.data).stem}: {N} frames, {B} blocks of ~{N // B}, roles {''.join('TVX'[r] for r in role)} "
      f"(T train, V val, X test), gap {a.gap}")
print(f"  train {len(tr)} of {len(tr_all)} available, val {len(va)} of {len(va_all)}, test {len(te)}")
print(f"  min distance to any training-block frame: val {nearest(va, tr_all).min()}, test {nearest(te, tr_all).min()} frames")
print("wrote", out)
