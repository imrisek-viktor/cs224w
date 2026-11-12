"""Experiment B sweep: scratch vs fine-tune (vs frozen) on AT-AT-CG-CG, test error against N.

Jobs run in priority order from one queue on 4 workers (3 MPS + 1 CPU; ~2.4x the throughput of a
single job on an M1 Pro). Finished jobs (test.json present) are skipped, so rerunning this script
resumes the sweep; an interrupted job restarts from step 0. Per-job stdout in runs/transfer/<name>.out.

Order: zero-shot -> long N=2000 reference runs (50k steps) -> seed 0 -> seed 1 -> seed 2 -> frozen seed 0,
so a partial night still gives a complete seed-0 curve.

usage: ./.venv/bin/python runs/transfer_sweep.py
"""
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

OUT = Path("runs/transfer")
STEPS = {25: 2500, 50: 3000, 100: 4000, 250: 6000, 500: 8000, 1000: 12000, 2000: 20000}
WORKERS = [("mps", 2), ("mps", 2), ("mps", 2), ("cpu", 4)]


def jobs():
    J = [("zeroshot_n2000_s0", ["--method", "zeroshot", "--ntrain", "2000", "--steps", "0"])]
    for m in ["scratch", "finetune"]:  # are 20k steps enough at N=2000? longer reference runs
        J.append((f"{m}_n2000_s0_long", ["--method", m, "--ntrain", "2000", "--steps", "50000"]))
    for seed in [0, 1, 2]:
        for n in sorted(STEPS, reverse=True):
            for m in ["scratch", "finetune"]:
                J.append((f"{m}_n{n}_s{seed}", ["--method", m, "--ntrain", str(n), "--seed", str(seed),
                                                "--steps", str(STEPS[n])]))
    for n in sorted(STEPS, reverse=True):
        J.append((f"frozen_n{n}_s0", ["--method", "frozen", "--ntrain", str(n), "--steps", str(STEPS[n])]))
    return J


def worker(wid, q, parent):
    dev, threads = WORKERS[wid]
    while True:
        if os.getppid() != parent:  # main script was stopped: don't start new jobs
            return
        item = q.get()
        if item is None:
            return
        name, args = item
        if (OUT / name / "test.json").exists():
            continue
        cmd = [".venv/bin/python", "src/transfer.py", *args, "--eval_every", "0", "--device", dev,
               "--threads", str(threads), "--out", str(OUT / name)]
        t = time.time()
        with open(OUT / f"{name}.out", "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f"{time.strftime('%H:%M:%S')} worker {wid} ({dev}) {name} rc={rc} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    q = mp.Queue()
    for j in jobs():
        q.put(j)
    for _ in WORKERS:
        q.put(None)
    ps = [mp.Process(target=worker, args=(i, q, os.getpid())) for i in range(len(WORKERS))]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print(f"{time.strftime('%H:%M:%S')} sweep done", flush=True)
