"""Build colab/atcg2_colab.ipynb with the *current* src/train.py and src/summarize.py embedded.

Re-run after editing train.py so the notebook never drifts from the local code:
    ./.venv/bin/python colab/make_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
train_src = (ROOT / "src/train.py").read_text()
summ_src = (ROOT / "src/summarize.py").read_text()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


cells = [
    md("""
# atcg2: AT-AT-CG-CG headroom rerun (fixed radial basis)

Reruns cutoff / dense / gated on MD22 AT-AT-CG-CG after the radial-basis fix
(short edges keep a 0–rc basis; the long channel gets its own rc–rl basis and starts switched off,
so at init dense/gated == the cutoff model).

**Before running**
1. Upload `data/md22_AT-AT-CG-CG.npz` (29 MB) from the project to Google Drive at `MyDrive/cs224w/data/`.
2. *Runtime → Change runtime type → GPU* (A100 or L4 if you have them; T4 works but is roughly Mac speed).
3. Run all cells top to bottom. Training runs in the background; checkpoints and logs go to
   `MyDrive/cs224w/runs/atcg2/`, so nothing is lost on a disconnect.

**If the runtime disconnects:** reconnect and run SETUP, the two `%%writefile` cells and LAUNCH again. Finished runs are skipped and
unfinished ones resume from their last epoch.
"""),
    code("""
# SETUP: Drive, paths, GPU
from google.colab import drive
drive.mount('/content/drive')
import os, shutil, subprocess, torch
DRIVE = '/content/drive/MyDrive/cs224w'
DATA_SRC = f'{DRIVE}/data/md22_AT-AT-CG-CG.npz'
OUT = f'{DRIVE}/runs/atcg2'
os.makedirs(OUT, exist_ok=True); os.makedirs('/content/src', exist_ok=True); os.makedirs('/content/data', exist_ok=True)
assert os.path.exists(DATA_SRC), f'upload the data file to {DATA_SRC}'
shutil.copy(DATA_SRC, '/content/data/')      # local copy: faster than reading from Drive
print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU - change runtime type')
print('CPUs:', os.cpu_count())
"""),
    code("%%writefile /content/src/train.py\n" + train_src),
    code("%%writefile /content/src/summarize.py\n" + summ_src),
    code("""
# SPEED CHECK (optional, ~1 min): seconds per epoch for each mode, one job alone on this GPU.
# Run it BEFORE launching. If training runs are already on the GPU it skips itself: there's no room
# for it, and the timing would be wrong anyway (read s/epoch from the PROGRESS cell instead).
import sys, time, gc, numpy as np, torch
if subprocess.run(['pgrep', '-f', 'src/train.py'], capture_output=True).stdout.strip():
    print('training runs are already using the GPU - skipping the speed check; use the PROGRESS cell')
else:
    sys.path.insert(0, '/content/src'); import train as T
    D = np.load('/content/data/md22_AT-AT-CG-CG.npz'); dev = torch.device('cuda')
    z = torch.tensor(D['z'], device=dev); R = torch.tensor(D['R'][:64], device=dev); FF = torch.tensor(D['F'][:64], device=dev)
    try:
        for mode in ['cutoff', 'dense', 'gated']:
            torch.manual_seed(0); m = T.Model(mode, 5.0, 24.0, e_scale=27.7).to(dev); m.set_norms(R[:32].cpu())
            opt = torch.optim.Adam(m.parameters(), 5e-4); pc = {}
            for s in range(10):
                if s == 3: torch.cuda.synchronize(); t0 = time.time()
                idx = np.arange(16) + 16 * (s % 4)
                E, F, g, _ = T.run_batch(m, z, R, idx, dev, pc, True)
                loss = ((F - FF[idx]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize(); dt = (time.time() - t0) / 7
            est = dt * 125 * 1.25   # 125 steps/epoch + ~25% for validation
            print(f'{mode:6s} {dt:.3f} s/step -> ~{est:.0f} s/epoch -> ~{est*300/3600:.1f} h for 300 epochs (alone)')
        print('Mac M4 Pro (MPS, alone) for reference: cutoff ~9, dense ~24, gated ~30 s/epoch')
    finally:  # always hand the GPU memory back to the training runs
        for v in ['m', 'opt', 'pc', 'E', 'F', 'g', 'loss', 'z', 'R', 'FF']:
            globals().pop(v, None)
        gc.collect(); torch.cuda.empty_cache()
"""),
    code("""
# LAUNCH the runs in the background (safe to re-run: skips finished/running runs, resumes the rest)
EPOCHS = 300
COMMON = ['--data', '/content/data/md22_AT-AT-CG-CG.npz', '--ntrain', '2000', '--nval', '500', '--ntest', '2000',
          '--L', '3', '--rl', '24', '--sched', 'cosine', '--max_epochs', str(EPOCHS), '--hours', '20',
          '--resume', '--device', 'cuda', '--threads', '1']
RUNS = {
    'cutoff_rc5':      ['--mode', 'cutoff', '--rc', '5'],   # same-hardware baseline (cheap)
    'dense_rc5':       ['--mode', 'dense',  '--rc', '5'],
    'gated_rc5_lam3':  ['--mode', 'gated',  '--rc', '5', '--lam', '3',  '--lam_ramp_epochs', '50'],
    'gated_rc5_lam10': ['--mode', 'gated',  '--rc', '5', '--lam', '10', '--lam_ramp_epochs', '50'],
}
try: procs
except NameError: procs = {}
for name, extra in RUNS.items():
    if os.path.exists(f'{OUT}/{name}/test.json'): print('done   ', name); continue
    if name in procs and procs[name].poll() is None: print('running', name); continue
    log = open(f'{OUT}/{name}.out', 'a')
    procs[name] = subprocess.Popen(['python', '/content/src/train.py', *COMMON, *extra, '--out', f'{OUT}/{name}'],
                                   stdout=log, stderr=subprocess.STDOUT, cwd='/content')
    print('started', name, 'pid', procs[name].pid)
"""),
    code("""
# PROGRESS (re-run any time)
import csv
for name in RUNS:
    p = f'{OUT}/{name}/log.csv'
    rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
    state = 'done' if os.path.exists(f'{OUT}/{name}/test.json') else ('running' if name in procs and procs[name].poll() is None else 'stopped')
    if rows:
        r = rows[-1]; g = r.get('val_gate_mean', '')
        print(f"{name:16s} {state:8s} epoch {int(r['epoch'])+1:3d}/{EPOCHS}  {float(r['epoch_s']):5.0f} s/epoch  "
              f"val F {float(r['val_f_mae']):.3f}" + (f'  gate mean {float(g):.2f}' if g else ''))
    else:
        print(f'{name:16s} {state:8s} (no epochs yet)  tail:', open(f'{OUT}/{name}.out').read()[-300:] if os.path.exists(f'{OUT}/{name}.out') else '')
print(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
"""),
    code("""
# SUMMARY table + validation curves (works on partial runs too)
!python /content/src/summarize.py "{OUT}"
from IPython.display import Image; Image(f'{OUT}/val_curves.png')
"""),
    md("""
Copy `MyDrive/cs224w/runs/atcg2/` back into the project's `runs/` folder to analyse it locally
(`./.venv/bin/python src/summarize.py runs/atcg2`). The old `runs/atcg1` dense/gated checkpoints use the
previous radial basis and will not load into the fixed model.
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "colab/atcg2_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
