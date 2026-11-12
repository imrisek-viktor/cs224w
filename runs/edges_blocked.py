"""Edge-budget sweep on the blocked split (src/make_split.py), both molecules, as one dependency-aware queue.

Why: the earlier sweeps (runs/edges_atat, runs/edges_atcg) used a random frame split that leaks between
train and test, and their edge lists came from teachers trained on that split. Here everything is redone
on the blocked split, including the teachers:
  1. plain cutoff models (these also serve as teacher / student),
  2. edge selection from the new teacher, as soon as it has finished (src/edge_select.py --split),
  3. 3 Å + selected edges, plus the "all" control (every candidate pair in the extra-edge channel).

AT-AT:       cutoff 3/4/5 Å, 1200 epochs WSD (decay 200); teacher rc5, student rc3, pool 3-5 Å;
             k = 4 for gap / teacher / nearest / random, plus all.
AT-AT-CG-CG: cutoff 3/4/5/6 Å, 800 epochs WSD (decay 150); teacher rc6, no student, pool 3-6 Å;
             teacher / nearest at k = 4, 8; random at k = 4; plus all.

Each job has a fixed device. Re-running skips finished jobs and does not duplicate jobs already running.
usage: nohup caffeinate -i ./.venv/bin/python runs/edges_blocked.py >> runs/edges_blocked/queue.log 2>&1 &
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/edges_blocked"
CAP = {"mps": 3, "cpu": 3}
THREADS = {"mps": "2", "cpu": "4"}

MOL = {
    "atat": dict(data="data/md22_AT-AT.npz", split="data/md22_AT-AT_split.npz", epochs=1200, decay=200,
                 save=["200", "400", "800"], teacher="cutoff_rc5", student="cutoff_rc3", pool=["3", "5"],
                 budgets=["4"]),
    "atcg": dict(data="data/md22_AT-AT-CG-CG.npz", split="data/md22_AT-AT-CG-CG_split.npz", epochs=800, decay=150,
                 save=["200", "400"], teacher="cutoff_rc6", student=None, pool=["3", "6"], budgets=["4", "8"]),
}


def common(m):
    c = MOL[m]
    return ["--data", c["data"], "--split", c["split"], "--L", "3", "--lr", "5e-4", "--sched", "wsd",
            "--max_epochs", str(c["epochs"]), "--decay_epochs", str(c["decay"]), "--lr_min", "1e-6",
            "--save_at", *c["save"], "--hours", "40", "--resume", "--seed", "0"]


def edges(m):
    return OUT / m / "edges"


def cutoff(m, rc, dev):
    return dict(name=f"{m}/cutoff_rc{rc}", dev=dev, args=common(m) + ["--mode", "cutoff", "--rc", str(rc)], needs=None)


def pairs(m, meth, k, dev):
    f = edges(m) / (f"pairs_{meth}_k{k}.npy" if meth != "all" else "pairs_all.npy")
    name = f"{m}/{meth}_k{k}" if meth != "all" else f"{m}/all"
    return dict(name=name, dev=dev, needs=f,
                args=common(m) + ["--mode", "pairs", "--rc", "3", "--rl", "7", "--pairs", str(f)])


jobs = [  # teachers / students first; priority order within each device
    cutoff("atcg", 6, "mps"), cutoff("atat", 5, "mps"), cutoff("atat", 3, "mps"),
    cutoff("atcg", 5, "mps"), cutoff("atcg", 4, "mps"),
    pairs("atcg", "teacher", 4, "mps"), pairs("atcg", "all", 0, "mps"), pairs("atcg", "nearest", 4, "mps"),
    pairs("atcg", "random", 4, "mps"), pairs("atcg", "teacher", 8, "mps"),
    pairs("atat", "teacher", 4, "mps"), pairs("atat", "nearest", 4, "mps"),
    cutoff("atcg", 3, "cpu"), cutoff("atat", 4, "cpu"),
    pairs("atat", "gap", 4, "cpu"), pairs("atat", "random", 4, "cpu"), pairs("atat", "all", 0, "cpu"),
    pairs("atcg", "nearest", 8, "cpu"),
]


def done(name):
    return (OUT / name / "test.json").exists()


def external():
    """Jobs of this sweep already running in other processes (e.g. from an earlier queue)."""
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    found = {}
    for line in ps:
        if "src/train.py" in line:
            for j in jobs:
                if line.rstrip().endswith(str(OUT / j["name"])):
                    found[j["name"]] = j["dev"]
    return found


selecting = {}  # molecule -> Popen of edge_select


def maybe_select(m):
    c = MOL[m]
    if (edges(m) / "pairs_all.npy").exists() or m in selecting:
        return
    if not done(f"{m}/{c['teacher']}") or (c["student"] and not done(f"{m}/{c['student']}")):
        return
    cmd = [sys.executable, "src/edge_select.py", "--data", c["data"], "--split", c["split"],
           "--teacher", str(OUT / m / c["teacher"]), "--pool", *c["pool"], "--budgets", *c["budgets"],
           "--nconf", "100", "--threads", "4", "--out", str(edges(m))]
    if c["student"]:
        cmd += ["--student", str(OUT / m / c["student"])]
    edges(m).mkdir(parents=True, exist_ok=True)
    selecting[m] = subprocess.Popen(cmd, stdout=open(OUT / m / "edge_select.log", "a"), stderr=subprocess.STDOUT, cwd=ROOT)
    print(time.strftime("%H:%M"), "edge selection started for", m, flush=True)


OUT.mkdir(parents=True, exist_ok=True)
pending = [j for j in jobs if not done(j["name"])]
running = {}  # name -> (Popen, dev)
print(time.strftime("%H:%M"), f"{len(pending)} of {len(jobs)} jobs to run", flush=True)
while pending or running or selecting:
    for name, (p, dev) in list(running.items()):
        if p.poll() is not None:
            print(time.strftime("%H:%M"), "finished", name, "exit", p.returncode, flush=True)
            del running[name]
    for m, p in list(selecting.items()):
        if p.poll() is not None:
            print(time.strftime("%H:%M"), "edge selection finished for", m, "exit", p.returncode, flush=True)
            del selecting[m]
            if p.returncode != 0:
                print("edge selection failed; see", OUT / m / "edge_select.log", flush=True)
                pending = [j for j in pending if not (j["needs"] and str(j["needs"]).startswith(str(edges(m))))]
    for m in MOL:
        maybe_select(m)
    ext = {n: d for n, d in external().items() if n not in running}
    busy = list(ext.values()) + [d for _, d in running.values()]
    pending = [j for j in pending if not done(j["name"])]
    for j in list(pending):
        if j["name"] in ext or busy.count(j["dev"]) >= CAP[j["dev"]]:
            continue
        if j["needs"] is not None and not j["needs"].exists():
            continue
        pending.remove(j)
        busy.append(j["dev"])
        (OUT / j["name"]).mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen([sys.executable, "src/train.py", *j["args"], "--device", j["dev"],
                              "--threads", THREADS[j["dev"]], "--out", str(OUT / j["name"])],
                             stdout=open(OUT / f"{j['name']}.out", "a"), stderr=subprocess.STDOUT, cwd=ROOT)
        running[j["name"]] = (p, j["dev"])
        print(time.strftime("%H:%M"), "started", j["name"], "on", j["dev"], "pid", p.pid, flush=True)
    time.sleep(30)
print(time.strftime("%H:%M"), "all done", flush=True)
