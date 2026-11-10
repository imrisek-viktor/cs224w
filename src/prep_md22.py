"""Convert MD22 extended-xyz files to compact .npz arrays.

Each frame:  N \n Energy=<E> Properties=... \n N lines of "Sym x y z fx fy fz".
Units (MD22/sGDML convention): positions in Angstrom, energy in kcal/mol, forces in kcal/mol/Angstrom.
"""
import sys
from pathlib import Path

import numpy as np

Z = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}


def parse(path):
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0])
    stride = n + 2
    nframes = len(lines) // stride
    z = np.array([Z[l.split()[0]] for l in lines[2:2 + n]], dtype=np.int64)
    E = np.empty(nframes)
    R = np.empty((nframes, n, 3), dtype=np.float32)
    F = np.empty((nframes, n, 3), dtype=np.float32)
    for k in range(nframes):
        b = k * stride
        assert int(lines[b]) == n
        E[k] = float(lines[b + 1].split("Energy=")[1].split()[0])
        arr = np.array([l.split()[1:7] for l in lines[b + 2:b + 2 + n]], dtype=np.float64)
        R[k], F[k] = arr[:, :3], arr[:, 3:]
    return z, R, F, E


def load_sgdml_npz(path):
    """sGDML-format .npz from quantum-machine.org (same units: Å, kcal/mol)."""
    d = np.load(path)  # allow_pickle=False
    assert str(d["r_unit"]) == "Ang" and str(d["e_unit"]) == "kcal/mol"
    return d["z"].astype(np.int64), d["R"].astype(np.float32), d["F"].astype(np.float32), d["E"].reshape(-1)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        z, R, F, E = load_sgdml_npz(p) if p.endswith(".npz") else parse(p)
        out = Path("data") / (Path(p).stem + ".npz")
        out.parent.mkdir(exist_ok=True)
        np.savez(out, z=z, R=R, F=F, E=E)
        ext = R.max(1) - R.min(1)
        print(f"{out}: frames={len(E)} atoms={len(z)} E mean={E.mean():.2f} std={E.std():.2f} "
              f"|F| rms={np.sqrt((F**2).mean()):.2f} extent(Å)~{ext.mean(0).round(1)}")
