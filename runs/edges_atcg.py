"""AT-AT-CG-CG (118 atoms) edge-budget sweep: 3 Å graph + k selected extra edges vs plain 3/4/5/6 Å cutoffs.

Edge lists come from src/edge_select.py with teacher runs/atcg1/cutoff_rc6 (no student: on AT-AT the
influence gap lost to raw teacher influence and nearest, so only teacher / nearest / random here),
candidate pool = pairs with median distance 3-6 Å.
All runs: 3 layers, 2000 train configs (same random split as atcg1), WSD schedule over 800 epochs
(constant 5e-4, linear decay to 1e-6 over the last 150), stable checkpoints at 200 and 400.

Each job has a fixed device: expensive graphs on MPS, cheap ones on the CPU.
Re-running skips finished runs and resumes unfinished ones.

usage: nohup caffeinate -i ./.venv/bin/python runs/edges_atcg.py >> runs/edges_atcg/queue.log 2>&1 &
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/edges_atcg"
EDGES = "data/edges/AT-AT-CG-CG"
EPOCHS, DECAY = 800, 150
CAP = {"mps": 3, "cpu": 2}
THREADS = {"mps": "2", "cpu": "5"}
COMMON = ["--data", "data/md22_AT-AT-CG-CG.npz", "--ntrain", "2000", "--nval", "500", "--ntest", "2000",
          "--L", "3", "--lr", "5e-4", "--sched", "wsd", "--max_epochs", str(EPOCHS), "--decay_epochs", str(DECAY),
          "--lr_min", "1e-6", "--save_at", "200", "400", "--hours", "40", "--resume", "--seed", "0"]


def cutoff(rc):
    return (f"s0/cutoff_rc{rc}", ["--mode", "cutoff", "--rc", str(rc)])


def pairs(meth, k):
    return (f"s0/{meth}_k{k}", ["--mode", "pairs", "--rc", "3", "--rl", "7", "--pairs", f"{EDGES}/pairs_{meth}_k{k}.npy"])


# (name, args, device), most expensive first within each device
jobs = [(*cutoff(6), "mps"), (*cutoff(5), "mps"), (*pairs("teacher", 4), "mps"), (*cutoff(4), "mps"),
        (*pairs("teacher", 8), "mps"), (*pairs("nearest", 8), "mps"),
        (*pairs("random", 4), "mps"),
        (*cutoff(3), "cpu"), (*pairs("nearest", 4), "cpu")]


def external():
    """Jobs of this sweep already running in other processes (e.g. left over from a restarted queue)."""
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    found = {}
    for line in ps:
        if "src/train.py" in line and str(OUT) in line:
            for name, _, _ in jobs:
                if line.rstrip().endswith(str(OUT / name)):
                    found[name] = "mps" if "--device mps" in line else "cpu"
    return found


OUT.mkdir(parents=True, exist_ok=True)
pending = [j for j in jobs if not (OUT / j[0] / "test.json").exists()]
running = {}  # name -> (Popen, device)
print(time.strftime("%H:%M"), f"{len(pending)} of {len(jobs)} jobs to run", flush=True)
while pending or running:
    for name, (p, dev) in list(running.items()):
        if p.poll() is not None:
            print(time.strftime("%H:%M"), "finished", name, "exit", p.returncode, flush=True)
            del running[name]
    ext = {n: d for n, d in external().items() if n not in running}
    busy = list(ext.values()) + [d for _, d in running.values()]
    pending = [j for j in pending if not (OUT / j[0] / "test.json").exists()]
    for job in list(pending):
        name, extra, dev = job
        if name in ext or busy.count(dev) >= CAP[dev]:
            continue
        busy.append(dev)
        pending.remove(job)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        log = open(OUT / f"{name}.out", "a")
        p = subprocess.Popen([sys.executable, "src/train.py", *COMMON, *extra, "--device", dev, "--threads", THREADS[dev],
                              "--out", str(OUT / name)], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        running[name] = (p, dev)
        print(time.strftime("%H:%M"), "started", name, "on", dev, "pid", p.pid, flush=True)
    time.sleep(30)
print(time.strftime("%H:%M"), "all done", flush=True)
