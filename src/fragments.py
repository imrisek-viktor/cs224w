"""Covalent fragments: connected components of the bond graph.

Pairs (i, j) in different fragments are non-covalent "interface" pairs, the ground truth we use to
check what the learned gates keep. Bonds: d_ij < r_cov(i) + r_cov(j) + 0.4 Å (median over frames,
so a bond that stretches transiently doesn't split a fragment).

usage: python src/fragments.py data/md22_AT-AT.npz data/md22_AT-AT-CG-CG.npz
writes data/<name>_fragments.npy (fragment id per atom) and prints a summary.
"""
import sys
from pathlib import Path

import numpy as np

R_COV = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 16: 1.05}
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}


def fragments(z, R, nframes=200):
    idx = np.linspace(0, len(R) - 1, min(nframes, len(R))).astype(int)
    d = np.median(np.linalg.norm(R[idx, :, None] - R[idx, None], axis=-1), axis=0)
    rc = np.array([R_COV[int(a)] for a in z])
    bonded = (d < rc[:, None] + rc[None] + 0.4) & ~np.eye(len(z), dtype=bool)
    frag = -np.ones(len(z), dtype=int)
    for s in range(len(z)):  # flood fill
        if frag[s] >= 0:
            continue
        frag[s] = k = frag.max() + 1
        stack = [s]
        while stack:
            i = stack.pop()
            for j in np.nonzero(bonded[i] & (frag < 0))[0]:
                frag[j] = k
                stack.append(j)
    return frag, d


if __name__ == "__main__":
    for p in sys.argv[1:]:
        D = np.load(p)
        z, R = D["z"], D["R"]
        frag, d = fragments(z, R)
        np.save(Path(p).with_name(Path(p).stem + "_fragments.npy"), frag)
        iu = np.triu_indices(len(z), 1)
        inter = frag[iu[0]] != frag[iu[1]]
        print(f"{Path(p).stem}: {len(z)} atoms, {frag.max() + 1} fragments, "
              f"max pair distance {d.max():.1f} Å (median frame)")
        for k in range(frag.max() + 1):
            zs = z[frag == k]
            print(f"  frag {k}: {len(zs)} atoms  " +
                  " ".join(f"{SYM[e]}{(zs == e).sum()}" for e in sorted(set(zs.tolist()))))
        for lo, hi in [(0, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 99)]:
            m = (d[iu] >= lo) & (d[iu] < hi)
            print(f"  pairs {lo:>2}-{hi:<2} Å: {m.sum():5d}  of which inter-fragment {inter[m].mean():.0%}")
