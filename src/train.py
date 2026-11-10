"""SchNet-style force field with optional long-range channel (cutoff / dense / gated).

Plain PyTorch (no PyG) so it runs anywhere; the layer mirrors eq. (1)-(3) of the proposal.

  cutoff : h_i += sum_{d<rc} phi(h_j, rbf(d)) fc(d)
  dense  : ... + sum_{rc<=d<rl} phi~(h_j, rbf(d)) w(d)           (g_ij = 1)
  gated  : ... + sum_{rc<=d<rl} g_ij phi~(h_j, rbf(d)) w(d),  g_ij = sigmoid(MLP(h_i+h_j, h_i*h_j, rbf(d)))
           with g computed once from embeddings after a first short-range-only layer,
           and loss += lambda * mean(g) (lambda linearly ramped).
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


def ssp(x):
    return Fn.softplus(x) - math.log(2.0)


class RBF(nn.Module):
    def __init__(self, rmax, n=50, rmin=0.0):
        super().__init__()
        self.register_buffer("mu", torch.linspace(rmin, rmax, n))
        self.gamma = 0.5 / ((rmax - rmin) / (n - 1)) ** 2

    def forward(self, d):
        return torch.exp(-self.gamma * (d[:, None] - self.mu) ** 2)


def cosine_cutoff(d, rc):
    return 0.5 * (torch.cos(math.pi * d / rc) + 1.0) * (d < rc)


def lr_window(d, rc, rl, ramp=1.0):
    """Smooth window: 0 at rc, ramps to 1 over `ramp` Angstrom, decays to 0 at rl."""
    t = ((d - rc) / ramp).clamp(0, 1)
    on = 0.5 * (1 - torch.cos(math.pi * t))
    off = 0.5 * (torch.cos(math.pi * d / rl) + 1.0)
    return on * off * (d >= rc) * (d < rl)


class CFConv(nn.Module):
    """Continuous-filter convolution + atom-wise update (one SchNet interaction)."""

    def __init__(self, F, nrbf):
        super().__init__()
        self.filt = nn.Sequential(nn.Linear(nrbf, F), nn.Softplus(), nn.Linear(F, F))
        self.lin_in = nn.Linear(F, F, bias=False)

    def message(self, h, src, rbf):
        return self.lin_in(h)[src] * self.filt(rbf)


class Interaction(nn.Module):
    def __init__(self, F, nrbf, long_range):
        super().__init__()
        self.short = CFConv(F, nrbf)
        self.long = CFConv(F, nrbf) if long_range else None
        if long_range:  # long channel starts switched off: at init the model *is* the cutoff model
            nn.init.zeros_(self.long.filt[-1].weight); nn.init.zeros_(self.long.filt[-1].bias)
        self.out = nn.Sequential(nn.Linear(F, F), nn.Softplus(), nn.Linear(F, F))

    def forward(self, h, se, le, norms):
        # sums are divided by fixed average neighbour counts (like avg_num_neighbors in NequIP/MACE);
        # without this, ~40 long-range neighbours per atom swamp the update and training stalls.
        # The long norm is the *dense* count, so a gate g_ij < 1 really shrinks the edge's contribution.
        src, dst, rbf, wgt = se
        agg = torch.zeros_like(h).index_add_(0, dst, self.short.message(h, src, rbf) * wgt[:, None]) / norms[0]
        if self.long is not None and le is not None:
            lsrc, ldst, lrbf, lw = le
            lagg = torch.zeros_like(h).index_add_(0, ldst, self.long.message(h, lsrc, lrbf) * lw[:, None])
            agg = agg + lagg / norms[1]
        return h + self.out(agg)


class Model(nn.Module):
    def __init__(self, mode, rc, rl, F=64, L=3, nrbf=50, e_shift=0.0, e_scale=1.0, pairs=None):
        super().__init__()
        self.mode, self.rc, self.rl = mode, rc, rl
        self.emb = nn.Embedding(20, F)
        # separate bases: short edges keep 0..rc resolution (same as the cutoff model) no matter how
        # far the long channel reaches; the long channel gets its own basis on [rc, rl].
        # (atcg1 used one basis over 0..rl, which coarsened short-range resolution ~5x for dense/gated.)
        self.rbf = RBF(rc, nrbf)
        if mode == "pairs":
            # fixed extra edge list (atom-index pairs, same for every configuration of the molecule).
            # Listed pairs get the extra channel at any distance d < rl, weighted by a cosine cutoff at rl,
            # so the energy stays smooth (no ramp at rc: listed pairs are often at 3-4 Å).
            assert pairs is not None, "--mode pairs needs --pairs"
            self.register_buffer("pairmask", torch.as_tensor(pairs, dtype=torch.bool))
            self.lrbf = RBF(rl, nrbf)
        else:
            self.lrbf = RBF(rl, nrbf, rmin=rc) if mode != "cutoff" else None
        # layer 0 is always short-range only (gate reads its output); later layers get the long channel
        self.layers = nn.ModuleList(
            [Interaction(F, nrbf, long_range=(mode != "cutoff" and l > 0)) for l in range(L)])
        self.readout = nn.Sequential(nn.Linear(F, F // 2), nn.Softplus(), nn.Linear(F // 2, 1))
        # zero-init the last readout layer: initial E ~ 0 (targets are centered). Otherwise the
        # initial energy error (~700 kcal/mol) dominates the gradient and force learning stalls.
        nn.init.zeros_(self.readout[-1].weight); nn.init.zeros_(self.readout[-1].bias)
        if mode == "gated":
            self.gate = nn.Sequential(nn.Linear(2 * F + nrbf, F), nn.Softplus(), nn.Linear(F, 1))
            nn.init.constant_(self.gate[-1].bias, 2.0)  # start mostly open
        self.e_shift, self.e_scale = e_shift, e_scale
        self.norms = (1.0, 1.0)  # (avg short neighbours, avg long neighbours); set from training data

    def set_norms(self, R):
        """Average neighbours per atom inside rc and in [rc, rl), from a few training configs."""
        d = torch.cdist(R, R)
        n = R.shape[1]
        short = ((d < self.rc).sum((1, 2)) - n).float().mean() / n
        if self.mode == "pairs":
            long = (self.pairmask.to(d.device) & (d < self.rl)).sum((1, 2)).float().mean() / n
        else:
            long = ((d >= self.rc) & (d < self.rl)).sum((1, 2)).float().mean() / n
        self.norms = (max(float(short), 1.0), max(float(long), 1.0))
        return self.norms

    def forward(self, z, pos, src, dst, batch, nmol):
        d = (pos[src] - pos[dst]).norm(dim=-1)
        s = d < self.rc
        ss, sd, sdist = src[s], dst[s], d[s]
        se = (ss, sd, self.rbf(sdist), cosine_cutoff(sdist, self.rc))
        le, gates = None, None
        if self.mode == "pairs":
            n = self.pairmask.shape[0]
            l = self.pairmask[src % n, dst % n] & (d < self.rl)
            ls, ld, ldist = src[l], dst[l], d[l]
            lrbf = self.lrbf(ldist)
            lw = cosine_cutoff(ldist, self.rl)
        elif self.mode != "cutoff":
            l = (d >= self.rc) & (d < self.rl)
            ls, ld, ldist = src[l], dst[l], d[l]
            lrbf = self.lrbf(ldist)
            lw = lr_window(ldist, self.rc, self.rl)
        h = self.emb(z)
        for k, layer in enumerate(self.layers):
            if k == 1 and self.mode != "cutoff":
                if self.mode == "gated":
                    hi, hj = h[ld], h[ls]
                    g = torch.sigmoid(self.gate(torch.cat([hi + hj, hi * hj, lrbf], -1))).squeeze(-1)
                    gates = g
                    lw = lw * g
                le = (ls, ld, lrbf, lw)
            h = layer(h, se, le if k > 0 else None, self.norms)
        e_atom = self.readout(h).squeeze(-1)
        E = torch.zeros(nmol, device=h.device, dtype=h.dtype).index_add_(0, batch, e_atom)
        E = E * self.e_scale + self.e_shift
        stats = {"n_short": int(s.sum()), "n_long": 0 if self.mode == "cutoff" else int(l.sum())}
        return E, gates, stats


def make_pairs(n, bsz, device):
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    m = i != j
    i, j = i[m], j[m]
    off = (torch.arange(bsz) * n)[:, None]
    src = (j[None] + off).reshape(-1)
    dst = (i[None] + off).reshape(-1)
    batch = torch.arange(bsz).repeat_interleave(n)
    return src.to(device), dst.to(device), batch.to(device)


def run_batch(model, z, R, idx, device, pair_cache, need_graph):
    bsz, n = len(idx), R.shape[1]
    if bsz not in pair_cache:
        pair_cache[bsz] = make_pairs(n, bsz, device)
    src, dst, batch = pair_cache[bsz]
    pos = R[idx].reshape(-1, 3).clone().requires_grad_(True)
    zz = z.repeat(bsz)
    E, gates, stats = model(zz, pos, src, dst, batch, bsz)
    F = -torch.autograd.grad(E.sum(), pos, create_graph=need_graph)[0]
    return E, F.reshape(bsz, n, 3), gates, stats


def evaluate(model, z, R, Fref, Eref, idxs, device, bs, pair_cache):
    model.eval()
    eae, fae, nf, gsum, gcnt, gfrac = 0.0, 0.0, 0, 0.0, 0, {}
    for b in range(0, len(idxs), bs):
        idx = idxs[b:b + bs]
        E, F, g, _ = run_batch(model, z, R, idx, device, pair_cache, need_graph=False)
        eae += (E.detach() - Eref[idx]).abs().sum().item()
        fae += (F.detach() - Fref[idx]).abs().sum().item()
        nf += F.numel()
        if g is not None:
            g = g.detach()
            gsum += g.sum().item(); gcnt += g.numel()
            for t in (0.5, 0.1, 0.01):
                gfrac[t] = gfrac.get(t, 0) + (g > t).sum().item()
    model.train()
    out = {"e_mae": eae / len(idxs), "f_mae": fae / nf}
    if gcnt:
        out["gate_mean"] = gsum / gcnt
        for t, c in gfrac.items():
            out[f"gate_frac>{t}"] = c / gcnt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/md22_AT-AT.npz")
    ap.add_argument("--mode", choices=["cutoff", "dense", "gated", "pairs"], default="cutoff")
    ap.add_argument("--pairs", default=None,
                    help="pairs mode: .npy of (K, 2) atom-index pairs added as extra edges (with --rl as their cutoff)")
    ap.add_argument("--rc", type=float, default=5.0)
    ap.add_argument("--rl", type=float, default=20.0)
    ap.add_argument("--F", type=int, default=64)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--ntrain", type=int, default=3000)
    ap.add_argument("--nval", type=int, default=500)
    ap.add_argument("--ntest", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wE", type=float, default=1.0, help="weight on per-atom energy MSE")
    ap.add_argument("--wF", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=1000.0)
    ap.add_argument("--lam", type=float, default=0.0, help="gate sparsity weight (gated mode)")
    ap.add_argument("--lam_ramp_epochs", type=int, default=50)
    ap.add_argument("--hours", type=float, default=1.0, help="wall-clock training budget")
    ap.add_argument("--max_epochs", type=int, default=100000)
    ap.add_argument("--sched", choices=["plateau", "cosine"], default="plateau")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt if present")
    ap.add_argument("--eval_bs", type=int, default=16, help="eval batch (large ones blow MPS memory)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "args.json", "w"), indent=2)
    dev = torch.device(a.device)

    D = np.load(a.data)
    rng = np.random.default_rng(1234)  # split fixed across seeds/modes
    perm = rng.permutation(len(D["E"]))
    tr, va = perm[:a.ntrain], perm[a.ntrain:a.ntrain + a.nval]
    te = perm[a.ntrain + a.nval:a.ntrain + a.nval + a.ntest]
    z = torch.tensor(D["z"], device=dev)
    R = torch.tensor(D["R"], device=dev)
    Fr = torch.tensor(D["F"], device=dev)
    e_shift = float(D["E"][tr].mean())
    e_scale = float(np.sqrt((D["F"][tr] ** 2).mean()))
    # raw energies (~1e6 kcal/mol) lose precision in float32 -> train on centered energies
    Er_c = torch.tensor(D["E"] - e_shift, dtype=torch.float32, device=dev)

    pairmask = None
    if a.mode == "pairs":
        P = np.load(a.pairs)
        pairmask = np.zeros((len(D["z"]), len(D["z"])), dtype=bool)
        pairmask[P[:, 0], P[:, 1]] = pairmask[P[:, 1], P[:, 0]] = True
        np.fill_diagonal(pairmask, False)
        print(f"extra edges: {pairmask.sum() // 2} pairs from {a.pairs}", flush=True)
    model = Model(a.mode, a.rc, a.rl, a.F, a.L, e_shift=0.0, e_scale=e_scale, pairs=pairmask).to(dev)
    norms = model.set_norms(torch.tensor(D["R"][tr[:200]]))
    print(f"avg neighbours/atom: short={norms[0]:.1f} long={norms[1]:.1f}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    if a.sched == "cosine":  # fixed epoch budget -> every run finishes the same schedule
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.max_epochs, eta_min=1e-6)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.7, patience=15, min_lr=1e-6)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"mode={a.mode} rc={a.rc} rl={a.rl} params={nparam} device={dev}", flush=True)

    # --- resume: last.pt holds model/optimizer/scheduler state after every epoch ---
    start_ep, best, prev_h = 0, float("inf"), 0.0
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep, best, prev_h = ck["epoch"] + 1, ck["best"], ck["hours"]
        print(f"resumed from epoch {start_ep} (best val F {best:.3f}, {prev_h:.2f} h so far)", flush=True)
    rows = list(csv.DictReader(open(out / "log.csv"))) if start_ep > 0 and (out / "log.csv").exists() else []
    log = open(out / "log.csv", "w", newline="")
    w = None
    if rows:  # keep the existing history up to the checkpoint; drop rows newer than it
        w = csv.DictWriter(log, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(r for r in rows if int(r["epoch"]) < start_ep)
        log.flush()
    pc, t0 = {}, time.time()
    n_atoms = R.shape[1]
    tr_t = torch.tensor(tr)
    ep = start_ep - 1
    for ep in range(start_ep, a.max_epochs):
        ep_t = time.time()
        lam = a.lam * min(1.0, ep / max(1, a.lam_ramp_epochs))
        order = tr_t[torch.randperm(len(tr_t))].numpy()
        lsum, nb, stats = 0.0, 0, None
        for b in range(0, len(order), a.bs):
            idx = order[b:b + a.bs]
            E, F, g, stats = run_batch(model, z, R, idx, dev, pc, need_graph=True)
            # energy error per atom so the energy term doesn't swamp the force term
            loss = a.wE * (((E - Er_c[idx]) / n_atoms) ** 2).mean() + a.wF * ((F - Fr[idx]) ** 2).mean()
            if g is not None and lam > 0:
                loss = loss + lam * g.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            lsum += loss.item(); nb += 1
        v = evaluate(model, z, R, Fr, Er_c, va, dev, a.eval_bs, pc)
        if dev.type == "mps":  # release the allocator cache so several MPS jobs fit in unified memory
            torch.mps.empty_cache()
        elif dev.type == "cuda":  # several jobs may share one Colab GPU
            torch.cuda.empty_cache()
        sched.step() if a.sched == "cosine" else sched.step(v["f_mae"])
        hours = prev_h + (time.time() - t0) / 3600
        row = {"epoch": ep, "time_h": hours, "epoch_s": time.time() - ep_t,
               "lr": opt.param_groups[0]["lr"], "lam": lam, "train_loss": lsum / nb,
               "n_short_per_mol": stats["n_short"] / len(idx), "n_long_per_mol": stats["n_long"] / len(idx),
               **{f"val_{k}": x for k, x in v.items()}}
        if w is None:
            w = csv.DictWriter(log, fieldnames=list(row.keys())); w.writeheader()
        w.writerow(row); log.flush()
        if v["f_mae"] < best:
            best = v["f_mae"]
            torch.save(model.state_dict(), out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best, "hours": hours}, out / "last.tmp")
        (out / "last.tmp").replace(out / "last.pt")  # atomic, so a kill mid-save can't corrupt it
        if ep % 5 == 0:
            print(" ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in row.items()),
                  flush=True)
        if hours > a.hours:
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    t = evaluate(model, z, R, Fr, Er_c, te, dev, a.eval_bs, pc)
    t["best_val_f_mae"] = best
    t["epochs"] = ep + 1
    t["params"] = nparam
    t["norms"] = norms
    json.dump(t, open(out / "test.json", "w"), indent=2)
    print("TEST", t, flush=True)


if __name__ == "__main__":
    main()
