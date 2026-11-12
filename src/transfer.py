"""Experiment B: does AT-AT pretraining save labels / training on AT-AT-CG-CG?

One run = one (method, N, seed) cell. Uses the blocked split (data/<name>_split_transfer.npz, from
src/blocked_split.py) so test frames are >= 50 MD steps from any training frame.

methods
  scratch   random init (target-data e_scale / neighbour norms, like train.py)
  finetune  init from a pretrained checkpoint (source-data e_scale / norms, so step 0 = zero-shot model),
            all weights trained
  frozen    as finetune, but the embedding and the first --freeze interaction layers stay fixed
  zeroshot  finetune with 0 steps (only the energy offset is fit, on the N training frames)

Training is step-based (not epoch-based) so small N gets enough updates: Adam, linear warmup then
cosine to 1e-6 over --steps, validation every --eval_every steps, best-validation weights evaluated on
the test set. Fine-tuned models first get a scalar energy offset fit on the training frames (the
composition differs from AT-AT, so energies are shifted; forces are unaffected).

Outputs in --out: args.json, log.csv (step, time, lr, train_loss, val_f_mae, val_e_mae), best.pt,
test.json (force/energy MAE, per-nucleobase force MAE, A/T force MAE by distance to nearest C/G atom).

usage: ./.venv/bin/python src/transfer.py --method finetune --ntrain 100 --seed 0 --steps 6000 \
         --device mps --out runs/transfer/finetune_n100_s0
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/md22_AT-AT-CG-CG.npz")
ap.add_argument("--source", default="data/md22_AT-AT.npz", help="pretraining data (for e_scale / norms)")
ap.add_argument("--init", default="runs/h1/cutoff_rc5/best.pt")
ap.add_argument("--method", choices=["scratch", "finetune", "frozen", "zeroshot"], required=True)
ap.add_argument("--freeze", type=int, default=1, help="frozen: number of interaction layers kept fixed")
ap.add_argument("--ntrain", type=int, required=True)
ap.add_argument("--nval", type=int, default=500)
ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--eval_every", type=int, default=0, help="0 -> steps // 40")
ap.add_argument("--warmup", type=float, default=0.02, help="fraction of steps")
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--wE", type=float, default=1.0)
ap.add_argument("--wF", type=float, default=1.0)
ap.add_argument("--clip", type=float, default=1000.0)
ap.add_argument("--rc", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="mps")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--out", required=True)
a = ap.parse_args()

torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
if (out / "test.json").exists():
    print(f"{out} already finished, skipping"); sys.exit(0)
json.dump(vars(a), open(out / "args.json", "w"), indent=2)
dev = torch.device(a.device)

# ---- data and blocked split ----
D = np.load(a.data)
S = np.load(Path(a.data).with_name(Path(a.data).stem + "_split_transfer.npz"))
tr = np.random.default_rng(a.seed).permutation(S["train_pool"])[:a.ntrain]  # nested in N for a given seed
va = np.random.default_rng(999).permutation(S["val"])[:a.nval]              # same for every run
te = S["test"]                                                               # all test frames
z = torch.tensor(D["z"], device=dev)
R = torch.tensor(D["R"], device=dev)
Fr = torch.tensor(D["F"], device=dev)
Er_c = torch.tensor(D["E"] - D["E"][tr].mean(), dtype=torch.float32, device=dev)  # centered on train mean
n_atoms = R.shape[1]

# ---- model ----
pretrained = a.method != "scratch"
if pretrained:
    Src = np.load(a.source)
    tr_src = np.random.default_rng(1234).permutation(len(Src["E"]))[:3000]  # h1's training frames
    e_scale = float(np.sqrt((Src["F"][tr_src] ** 2).mean()))
    norm_R = torch.tensor(Src["R"][tr_src[:200]])
else:
    e_scale = float(np.sqrt((D["F"][tr] ** 2).mean()))
    norm_R = torch.tensor(D["R"][tr[:200]])
model = T.Model("cutoff", a.rc, 24.0, 64, 3, e_shift=0.0, e_scale=e_scale).to(dev)
model.set_norms(norm_R)
if pretrained:
    model.load_state_dict(torch.load(a.init, map_location=dev))
if a.method == "frozen":
    for p in list(model.emb.parameters()) + [p for l in model.layers[:a.freeze] for p in l.parameters()]:
        p.requires_grad_(False)
pc = {}


def predict_E(idxs):
    model.eval(); Es = []
    for b in range(0, len(idxs), 16):
        E, _, _, _ = T.run_batch(model, z, R, idxs[b:b + 16], dev, pc, need_graph=False)
        Es.append(E.detach())
    model.train(); return torch.cat(Es)


if pretrained:  # scalar energy offset (fixed composition -> one constant), fit on the training frames
    model.e_shift = float((Er_c[tr] - predict_E(tr)).mean())

# ---- training ----
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.Adam(params, lr=a.lr)
nwarm = max(1, int(a.warmup * a.steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / nwarm) * max(
    1e-6 / a.lr, 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - nwarm) / max(1, a.steps - nwarm))))))
eval_every = a.eval_every or max(1, a.steps // 40)
print(f"method={a.method} N={a.ntrain} seed={a.seed} steps={a.steps} eval_every={eval_every} "
      f"trainable={sum(p.numel() for p in params)} norms={model.norms} e_shift={model.e_shift:.2f}", flush=True)

log = open(out / "log.csv", "w", newline="")
w = csv.writer(log); w.writerow(["step", "time_s", "lr", "train_loss", "val_f_mae", "val_e_mae"])
t0, best, lsum, nb = time.time(), float("inf"), 0.0, 0
rng = np.random.default_rng(a.seed + 1)
order, pos = rng.permutation(tr), 0


def do_eval(step):
    global best, lsum, nb
    v = T.evaluate(model, z, R, Fr, Er_c, va, dev, 16, pc)
    if dev.type == "mps":
        torch.mps.empty_cache()
    w.writerow([step, round(time.time() - t0, 1), opt.param_groups[0]["lr"], lsum / max(nb, 1),
                v["f_mae"], v["e_mae"]]); log.flush()
    lsum, nb = 0.0, 0
    if v["f_mae"] < best:
        best = v["f_mae"]
        torch.save({"model": model.state_dict(), "e_shift": model.e_shift}, out / "best.pt")
    return v


v = do_eval(0)
print(f"step 0 val F {v['f_mae']:.3f}", flush=True)
for step in range(1, a.steps + 1):
    if pos + a.bs > len(order):  # reshuffle at the end of each pass (batch = min(bs, N))
        order, pos = rng.permutation(tr), 0
    idx = order[pos:pos + a.bs]; pos += a.bs
    E, F, _, _ = T.run_batch(model, z, R, idx, dev, pc, need_graph=True)
    loss = a.wE * (((E - Er_c[idx]) / n_atoms) ** 2).mean() + a.wF * ((F - Fr[idx]) ** 2).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, a.clip)
    opt.step(); sched.step()
    lsum += loss.item(); nb += 1
    if step % eval_every == 0 or step == a.steps:
        v = do_eval(step)
        if step % (eval_every * 5) == 0 or step == a.steps:
            print(f"step {step} {time.time() - t0:.0f}s lr {opt.param_groups[0]['lr']:.2e} "
                  f"val F {v['f_mae']:.3f} (best {best:.3f})", flush=True)

# ---- test: best-validation weights ----
ck = torch.load(out / "best.pt", map_location=dev)
model.load_state_dict(ck["model"]); model.e_shift = ck["e_shift"]
model.eval(); errs, eae = [], 0.0
for b in range(0, len(te), 16):
    idx = te[b:b + 16]
    E, F, _, _ = T.run_batch(model, z, R, idx, dev, pc, need_graph=False)
    errs.append((F.detach() - Fr[idx]).abs().cpu().numpy())
    eae += (E.detach() - Er_c[idx]).abs().sum().item()
err = np.concatenate(errs).mean((0, 2))  # per-atom force MAE over test frames

# nucleobase labels (same rule as zeroshot.py) and A/T distance to nearest C/G atom
frag = np.load(Path(a.data).with_name(Path(a.data).stem + "_fragments.npy"))
zB = D["z"]
def base(k):
    zs = zB[frag == k]; n, no = len(zs), (zs == 8).sum()
    return {16: "G", 13: "C"}.get(n, "T" if no == 2 else "A")
lab = np.array([base(k) for k in frag])
Rm = np.median(D["R"][te[:200]], 0)
dmin = np.linalg.norm(Rm[:, None] - Rm[None], axis=-1)[:, ~np.isin(lab, ["A", "T"])].min(1)
at = np.isin(lab, ["A", "T"])
res = {"f_mae": float(err.mean()), "e_mae": eae / len(te), "best_val_f_mae": best,
       "per_base": {b: float(err[lab == b].mean()) for b in "ATCG"},
       "AT_by_dist_to_CG": {f"{lo}-{hi}": float(err[at & (dmin >= lo) & (dmin < hi)].mean())
                            for lo, hi in [(0, 3.5), (3.5, 5), (5, 7), (7, 99)] if (at & (dmin >= lo) & (dmin < hi)).any()},
       "steps": a.steps, "time_s": time.time() - t0, "ntest": len(te)}
json.dump(res, open(out / "test.json", "w"), indent=2)
print("TEST", json.dumps(res), flush=True)
