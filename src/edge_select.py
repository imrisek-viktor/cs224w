"""Influence-guided edge selection ("which atoms need to talk?").

Influence of atom j on atom i in a trained model: I_ij = ||dF_i / dr_j|| (Frobenius norm of the 3x3
block of the model's Hessian), symmetrized and averaged over training configurations. This is the
force-field version of the sensitivity Jacobian used in over-squashing analyses.

Given a wide-graph teacher (e.g. 5 Å cutoff) and a narrow-graph student (e.g. 3 Å), candidate edges are
the pairs the student lacks (median distance in [pool_lo, pool_hi)). They are ranked by:
  gap      relu(I_teacher - I_student): influence the student's graph fails to carry (over-squashed)
  teacher  I_teacher: raw influence in the wide model
  nearest  shortest median distance (what a slightly larger cutoff would add)
  random   uniform
and the top K = budget * n_atoms / 2 pairs are saved (each pair adds one edge per endpoint, so
`budget` = extra edges per atom on average).

usage: ./.venv/bin/python src/edge_select.py --data data/md22_AT-AT.npz \
          --teacher runs/h1/cutoff_rc5 --student runs/h1/cutoff_rc3 --out data/edges/AT-AT
writes <out>/influence.npz and <out>/pairs_<method>_k<budget>.npy
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train as T


def load_model(run, D, tr):
    a = json.load(open(Path(run) / "args.json"))
    assert a["mode"] == "cutoff", "influence models must be cutoff models"
    m = T.Model("cutoff", a["rc"], a["rl"], a["F"], a["L"],
                e_scale=float(np.sqrt((D["F"][tr] ** 2).mean())))
    m.set_norms(torch.tensor(D["R"][tr[:200]]))
    m.load_state_dict(torch.load(Path(run) / "best.pt", map_location="cpu"))
    return m.eval()


def influence(m, z, R):
    """Mean symmetrized ||dF_i/dr_j|| over configurations R (k, n, 3)."""
    n = len(z)
    src, dst, batch = T.make_pairs(n, 1, "cpu")

    def forces(x):
        x = x.reshape(n, 3)
        E, _, _ = m(z, x, src, dst, batch, 1)
        return -torch.autograd.grad(E.sum(), x, create_graph=True)[0].reshape(-1)

    acc = np.zeros((n, n))
    for r in R:
        x = torch.tensor(r).reshape(-1).requires_grad_(True)
        H = torch.autograd.functional.jacobian(forces, x, vectorize=True).detach().numpy()
        acc += np.linalg.norm(H.reshape(n, 3, n, 3).transpose(0, 2, 1, 3), axis=(2, 3))
    acc /= len(R)
    return (acc + acc.T) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--ntrain", type=int, default=3000, help="must match the runs' training split")
    ap.add_argument("--nconf", type=int, default=200, help="training configurations to average over")
    ap.add_argument("--pool", type=float, nargs=2, default=[3.0, 5.0], help="candidate median-distance range")
    ap.add_argument("--budgets", type=float, nargs="+", default=[2, 4, 8])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    D = np.load(a.data)
    tr = np.random.default_rng(1234).permutation(len(D["E"]))[:a.ntrain]  # same split as train.py
    z = torch.tensor(D["z"]); n = len(z)
    conf = D["R"][tr[np.linspace(0, len(tr) - 1, a.nconf).astype(int)]]
    dmed = np.median(np.linalg.norm(conf[:, :, None] - conf[:, None], axis=-1), axis=0)

    t0 = time.time()
    It = influence(load_model(a.teacher, D, tr), z, conf)
    Is = influence(load_model(a.student, D, tr), z, conf)
    print(f"influence on {a.nconf} configs: {time.time() - t0:.0f} s", flush=True)

    iu = np.triu_indices(n, 1)
    pool = (dmed[iu] >= a.pool[0]) & (dmed[iu] < a.pool[1])
    cand = np.nonzero(pool)[0]
    scores = {
        "gap": np.maximum(It - Is, 0)[iu][cand],
        "teacher": It[iu][cand],
        "nearest": -dmed[iu][cand],
        "random": np.random.default_rng(0).random(len(cand)),
    }
    fpath = Path(a.data).with_name(Path(a.data).stem + "_fragments.npy")
    inter = (np.load(fpath)[:, None] != np.load(fpath)[None])[iu][cand] if fpath.exists() else None
    np.savez(out / "influence.npz", teacher=It, student=Is, dmed=dmed, pool=pool, args=json.dumps(vars(a)))

    print(f"{n} atoms, {len(cand)} candidate pairs ({a.pool[0]}-{a.pool[1]} Å median distance)")
    print("method   budget  pairs  mean dist  inter-base  overlap w/ nearest  teacher-influence covered")
    order = {k: np.argsort(-v, kind="stable") for k, v in scores.items()}
    for k in a.budgets:
        K = min(int(round(k * n / 2)), len(cand))
        near = set(order["nearest"][:K])
        for meth, o in order.items():
            sel = o[:K]
            pairs = np.stack([iu[0][cand[sel]], iu[1][cand[sel]]], 1)
            np.save(out / f"pairs_{meth}_k{k:g}.npy", pairs)
            cov = scores["teacher"][sel].sum() / scores["teacher"].sum()
            ib = f"{inter[sel].mean():.0%}" if inter is not None else "-"
            print(f"{meth:8s} {k:6g}  {K:5d}  {dmed[iu][cand[sel]].mean():7.2f} Å  {ib:>10s}  "
                  f"{len(near & set(sel)) / K:18.0%}  {cov:25.0%}")
    print("wrote", out)


if __name__ == "__main__":
    main()
