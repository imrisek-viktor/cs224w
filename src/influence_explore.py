"""Exploratory: influence norm(dF_i/dr_j) in the trained 5 A AT-AT-CG-CG model (runs/atcg1/cutoff_rc5), by distance and base. Run from the project root."""
import sys, time; sys.path.insert(0, "src")
import numpy as np, torch, train as T
torch.set_num_threads(8)
D = np.load("data/md22_AT-AT-CG-CG.npz"); z = torch.tensor(D["z"]); n = len(z)
perm = np.random.default_rng(1234).permutation(len(D["E"])); tr = perm[:2000]; te = perm[2500:4500]
m = T.Model("cutoff", 5.0, 24.0, e_scale=float(np.sqrt((D["F"][tr] ** 2).mean())))
m.set_norms(torch.tensor(D["R"][tr[:200]])); m.load_state_dict(torch.load("runs/atcg1/cutoff_rc5/best.pt", map_location="cpu")); m.eval()
src, dst, batch = T.make_pairs(n, 1, "cpu")
def forces(x):
    x = x.reshape(n, 3)
    E, _, _ = m(z, x, src, dst, batch, 1)
    return -torch.autograd.grad(E.sum(), x, create_graph=True)[0].reshape(-1)
frag = np.load("data/md22_AT-AT-CG-CG_fragments.npy"); inter = frag[:, None] != frag[None]
I_all, d_all = [], []
t0 = time.time()
for c in te[:30]:
    x = torch.tensor(D["R"][c]).reshape(-1).requires_grad_(True)
    H = torch.autograd.functional.jacobian(forces, x, vectorize=True).detach().numpy()   # (3n, 3n)
    I_all.append(np.linalg.norm(H.reshape(n, 3, n, 3).transpose(0, 2, 1, 3), axis=(2, 3)))
    d_all.append(np.linalg.norm(D["R"][c][:, None] - D["R"][c][None], axis=-1))
print(f"{len(I_all)} configs, {time.time()-t0:.0f} s")
I = np.stack(I_all); d = np.stack(d_all)
iu = np.triu_indices(n, 1)
Iu = ((I + I.transpose(0, 2, 1)) / 2)[:, iu[0], iu[1]].ravel(); du = d[:, iu[0], iu[1]].ravel()
iv = np.broadcast_to(inter[iu], (len(I_all), len(iu[0]))).ravel()
print("mean influence by distance (all pairs):")
for lo, hi in [(0,2),(2,3),(3,4),(4,5),(5,6),(6,8),(8,12),(12,99)]:
    s = (du >= lo) & (du < hi)
    print(f"  {lo:>2}-{hi:<2} Å  n/cfg {s.sum()/len(I_all):6.0f}  mean {Iu[s].mean():8.3f}   intra-base {Iu[s & ~iv].mean() if (s&~iv).any() else float('nan'):8.3f}  inter-base {Iu[s & iv].mean() if (s&iv).any() else float('nan'):8.3f}")
# the candidate shell for a 3 Å base graph: 3-5 Å (teacher can only directly see < 5 Å)
s = (du >= 3) & (du < 5); Is = np.sort(Iu[s])[::-1]; cs = np.cumsum(Is) / Is.sum()
print("3-5 Å shell: share of total influence captured by the top x% of pairs:")
for f in [0.05, 0.1, 0.25, 0.5]:
    print(f"  top {f:4.0%}: {cs[int(f*len(Is))-1]:.0%}")
rk = lambda a: np.argsort(np.argsort(a)).astype(float)
class _R:
    def __init__(s, c): s.correlation = c
spearmanr = lambda a, b: _R(np.corrcoef(rk(a), rk(b))[0, 1])
print(f"3-5 Å shell: Spearman(influence, -distance) = {spearmanr(Iu[s], -du[s]).correlation:.2f}")
# per-config top-25% by influence vs by distance (nearest): overlap
ov, interI, interD = [], [], []
for k in range(len(I_all)):
    Ik = ((I[k] + I[k].T) / 2)[iu]; dk = d[k][iu]; sk = (dk >= 3) & (dk < 5); idx = np.nonzero(sk)[0]; K = len(idx) // 4
    a = set(idx[np.argsort(-Ik[idx])[:K]]); b = set(idx[np.argsort(dk[idx])[:K]])
    ov.append(len(a & b) / K); interI.append(inter[iu][list(a)].mean()); interD.append(inter[iu][list(b)].mean())
print(f"top-25% of 3-5 Å pairs: overlap influence-vs-nearest {np.mean(ov):.0%}; inter-base share: by influence {np.mean(interI):.0%}, by nearest {np.mean(interD):.0%}, all shell pairs {inter[iu][((d[0][iu]>=3)&(d[0][iu]<5))].mean():.0%}")
# stability across configs: same atom pairs top-ranked?
Ibar = ((I.mean(0) + I.mean(0).T) / 2)[iu]; dbar = np.median(d, 0)[iu]; sb = (dbar >= 3) & (dbar < 5)
ranks = [spearmanr(((I[k] + I[k].T) / 2)[iu][sb], Ibar[sb]).correlation for k in range(len(I_all))]
print(f"stability: Spearman of per-config influence vs mean influence (fixed pairs, 3-5 Å median distance): {np.mean(ranks):.2f}")
