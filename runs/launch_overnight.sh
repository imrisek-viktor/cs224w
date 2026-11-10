#!/bin/zsh
# Overnight headroom + first gate runs on AT-AT (seed 0, 3000 train configs).
# 3 jobs share the Apple GPU (MPS), 4 jobs on CPU with 3 threads each.
# caffeinate keeps the Mac awake while any job is running.
cd "$(dirname "$0")/.."
H=${HOURS:-7.5}
PY=./.venv/bin/python
OUT=${OUT:-runs/h1}
mkdir -p $OUT

run() {  # name device threads args...
  local name=$1 dev=$2 thr=$3; shift 3
  nohup caffeinate -i $PY src/train.py --device $dev --threads $thr --hours $H --out $OUT/$name "$@" \
    > $OUT/$name.out 2>&1 &
  echo "started $name (pid $!)"
}

# --- headroom: does direct long-range help over a cutoff? ---
run cutoff_rc5        mps 2  --mode cutoff --rc 5
run dense_rc5         mps 2  --mode dense  --rc 5 --rl 20
run cutoff_rc3        cpu 3  --mode cutoff --rc 3
run dense_rc3         cpu 3  --mode dense  --rc 3 --rl 20
run cutoff_rc10       cpu 3  --mode cutoff --rc 10          # "just raise the cutoff" reference
# --- first look at gates ---
run gated_rc5_lam0.1  mps 2  --mode gated  --rc 5 --rl 20 --lam 0.1 --lam_ramp_epochs 100
run gated_rc5_lam1    cpu 3  --mode gated  --rc 5 --rl 20 --lam 1.0 --lam_ramp_epochs 100
