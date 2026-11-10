"""How close are test frames to training frames under the random split used in train.py?

Prints, per dataset: distance (in trajectory frames) from each test frame to its nearest training
frame, and how fast forces decorrelate / atoms move with frame lag.

usage: ./.venv/bin/python src/split_check.py
"""
import numpy as np

LAGS = [1, 2, 5, 10, 20, 50]
for name, ntr, nva, nte in [("AT-AT", 3000, 500, 5000), ("AT-AT-CG-CG", 2000, 500, 2000)]:
    D = np.load(f"data/md22_{name}.npz")
    N = len(D["E"])
    perm = np.random.default_rng(1234).permutation(N)  # same split as train.py
    tr = np.sort(perm[:ntr]); te = perm[ntr + nva:ntr + nva + nte]
    gap = np.abs(te[:, None] - tr[None]).min(1)
    F = D["F"].reshape(N, -1); F = F - F.mean(0)
    ac = [np.mean(np.sum(F[:-l] * F[l:], 1) / np.sqrt(np.sum(F[:-l] ** 2, 1) * np.sum(F[l:] ** 2, 1))) for l in LAGS]
    R = D["R"]
    rms = [np.sqrt(((R[:-l] - R[l:]) ** 2).sum(-1).mean()) for l in LAGS]
    print(f"{name}: {N} frames; test frame -> nearest training frame: median {np.median(gap):.0f} frames, "
          f"{np.mean(gap <= 1):.0%} adjacent to a training frame, max {gap.max()}")
    print("   lag (frames)       " + " ".join(f"{l:5d}" for l in LAGS))
    print("   force correlation  " + " ".join(f"{a:5.2f}" for a in ac))
    print("   atom RMS move (Å)  " + " ".join(f"{a:5.2f}" for a in rms))
