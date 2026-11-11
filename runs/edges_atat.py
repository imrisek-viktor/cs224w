"""AT-AT edge-selection sweep: 3 Å graph + K selected extra edges vs plain 3/4/5 Å cutoff graphs.

Edge lists come from src/edge_select.py (teacher runs/h1/cutoff_rc5, student runs/h1/cutoff_rc3).
All runs: 3 layers, 3000 train configs, same (random) split.
LR schedule: warmup-stable-decay (WSD) over 1200 epochs: constant 5e-4, then a linear decay to 1e-6 over
the last 200 epochs. Full training state is also kept at epochs 200/400/800 (stable_ep<N>.pt) so shorter
budgets can be "finished" later with their own short decay.
(History: the first 6 runs started on a 400-epoch cosine schedule and were stopped at epochs 57-236;
they resume here under WSD, so their LR jumps back to 5e-4 at the resume point.)
Phase 1 = seed 0, k = 2 and 4 (11 runs). Set ALL = True for seed 1 and k = 1 as well.

A small job queue: MPS_WORKERS jobs on the GPU and CPU_WORKERS jobs on the CPU at a time.
Re-running this script skips finished runs and resumes unfinished ones (--resume).

usage: nohup caffeinate -i ./.venv/bin/python runs/edges_atat.py > runs/edges_atat/queue.log 2>&1 &
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/edges_atat"
EDGES = "data/edges/AT-AT"
EPOCHS, MPS_WORKERS, CPU_WORKERS, CPU_THREADS = 1200, 3, 4, 3
ALL = False
COMMON = ["--data", "data/md22_AT-AT.npz", "--ntrain", "3000", "--nval", "500", "--ntest", "5000",
          "--L", "3", "--lr", "5e-4", "--sched", "wsd", "--max_epochs", str(EPOCHS), "--decay_epochs", "200",
          "--lr_min", "1e-6", "--save_at", "200", "400", "800", "--hours", "40", "--resume"]

jobs = []
for seed in ([0, 1] if ALL else [0]):
    for rc in [3, 4, 5]:
        jobs.append((f"s{seed}/cutoff_rc{rc}", ["--mode", "cutoff", "--rc", str(rc), "--seed", str(seed)]))
    for k in ([2, 4, 1] if ALL else [2, 4]):
        for meth in ["gap", "nearest", "teacher", "random"]:
            jobs.append((f"s{seed}/{meth}_k{k}",
                         ["--mode", "pairs", "--rc", "3", "--rl", "7", "--pairs", f"{EDGES}/pairs_{meth}_k{k}.npy",
                          "--seed", str(seed)]))

OUT.mkdir(parents=True, exist_ok=True)
pending = [j for j in jobs if not (OUT / j[0] / "test.json").exists()]
running = {}  # name -> (Popen, device)
print(f"{len(pending)} of {len(jobs)} jobs to run", flush=True)
while pending or running:
    for name, (p, dev) in list(running.items()):
        if p.poll() is not None:
            print(time.strftime("%H:%M"), "finished", name, "exit", p.returncode, flush=True)
            del running[name]
    for dev, cap in [("mps", MPS_WORKERS), ("cpu", CPU_WORKERS)]:
        while pending and sum(d == dev for _, d in running.values()) < cap:
            name, extra = pending.pop(0)
            (OUT / name).mkdir(parents=True, exist_ok=True)
            log = open(OUT / f"{name}.out", "a")
            thr = "2" if dev == "mps" else str(CPU_THREADS)
            p = subprocess.Popen([sys.executable, "src/train.py", *COMMON, *extra, "--device", dev, "--threads", thr,
                                  "--out", str(OUT / name)], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            running[name] = (p, dev)
            print(time.strftime("%H:%M"), "started", name, "on", dev, "pid", p.pid, flush=True)
    time.sleep(30)
print("all done", flush=True)
