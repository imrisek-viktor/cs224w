"""Zero-shot transfer: AT-AT model (runs/h1/cutoff_rc5) evaluated on AT-AT-CG-CG, errors per nucleobase.

usage: ./.venv/bin/python src/zeroshot.py   (run from the project root; uses MPS)
"""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, train as T
dev = torch.device("mps")
A = np.load("data/md22_AT-AT.npz"); B = np.load("data/md22_AT-AT-CG-CG.npz")
fragB = np.load("data/md22_AT-AT-CG-CG_fragments.npy"); zB = B["z"]
def base(k):
    zs = zB[fragB == k]; n, no = len(zs), (zs == 8).sum()
    return {16: "G", 13: "C"}.get(n, "T" if no == 2 else "A")
lab = np.array([base(k) for k in fragB])
print("bases per atom:", {b: int((lab == b).sum()) for b in "ATCG"})
# test split identical to atcg runs
perm = np.random.default_rng(1234).permutation(len(B["E"])); te = perm[2500:4500]
trA = np.random.default_rng(1234).permutation(len(A["E"]))[:3000]

def load(path, data_tr, D, mode="cutoff"):
    m = T.Model(mode, 5.0, 24.0, e_scale=float(np.sqrt((D["F"][data_tr] ** 2).mean()))).to(dev)
    m.set_norms(torch.tensor(D["R"][data_tr[:200]]))
    m.load_state_dict(torch.load(path, map_location=dev)); m.eval(); return m

def per_atom_err(m):
    z = torch.tensor(zB, device=dev); R = torch.tensor(B["R"], device=dev); pc = {}; errs = []
    for b in range(0, len(te), 16):
        idx = te[b:b+16]; _, F, _, _ = T.run_batch(m, z, R, idx, dev, pc, False)
        errs.append((F.detach().cpu().numpy() - B["F"][idx]))
    return np.abs(np.concatenate(errs))  # (n, atoms, 3)

trB = perm[:2000]
models = {"AT-AT model, zero-shot (h1 cutoff_rc5)": load("runs/h1/cutoff_rc5/best.pt", trA, A),
          "ATCG scratch, 2000 cfg (atcg1 cutoff_rc5)": load("runs/atcg1/cutoff_rc5/best.pt", trB, B)}
print(f"zero-force baseline MAE: {np.abs(B['F'][te]).mean():.2f}")
for name, m in models.items():
    e = per_atom_err(m)
    print(f"{name}: all {e.mean():.2f} | " + "  ".join(f"{b} {e[:, lab == b].mean():.2f}" for b in "ATCG")
          + f" | H {e[:, zB == 1].mean():.2f} heavy {e[:, zB != 1].mean():.2f}")

# zero-shot error on A/T atoms vs distance to the nearest C/G atom (median geometry)
e0 = per_atom_err(models["AT-AT model, zero-shot (h1 cutoff_rc5)"]).mean((0, 2))
e1 = per_atom_err(models["ATCG scratch, 2000 cfg (atcg1 cutoff_rc5)"]).mean((0, 2))
Rm = np.median(B["R"][te[:200]], 0)
d = np.linalg.norm(Rm[:, None] - Rm[None], axis=-1)
at = np.isin(lab, ["A", "T"]); cg = ~at
dmin = d[:, cg].min(1)
print("A/T atoms by distance to nearest C/G atom:  zero-shot vs scratch")
for lo, hi in [(0, 3.5), (3.5, 5), (5, 7), (7, 10), (10, 99)]:
    s = at & (dmin >= lo) & (dmin < hi)
    if s.sum(): print(f"  {lo:>4}-{hi:<3} Å  n={s.sum():2d}  zero-shot {e0[s].mean():.2f}  scratch {e1[s].mean():.2f}")
# how the bases are arranged: pair/stack order by fragment centroids
cent = {k: Rm[fragB == k].mean(0) for k in range(fragB.max() + 1)}
print("fragment centroids distance matrix (base labels):")
ks = sorted(cent); print("     " + "  ".join(f"{base(k)}{k}".rjust(4) for k in ks))
for i in ks: print(f"{base(i)}{i}".rjust(4) + " " + "  ".join(f"{np.linalg.norm(cent[i]-cent[j]):4.1f}" for j in ks))
