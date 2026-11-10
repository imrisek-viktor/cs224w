#!/bin/zsh
# AT-AT-CG-CG (118 atoms, 8 separate bases, up to ~20.5 Å across): headroom + first gate sparsity.
#   standard settings L=3, rc=5 Å (receptive field 15 Å < molecule), long-range pairs up to rl=24 Å
#   cutoff vs dense      -> headroom: does direct long-range help on the bigger system?
#   gated lam=3, lam=10  -> do gates close, and what does it cost in accuracy?
#   cutoff rc=6          -> "just raise the cutoff" reference
# 2000 train configs (MD22 standard for this system), fixed 300-epoch cosine schedule for all runs.
# --resume: re-running this script continues each run from its last finished epoch.
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
OUT=${OUT:-runs/atcg1}
EPOCHS=${EPOCHS:-300}
mkdir -p $OUT
COMMON=(--data data/md22_AT-AT-CG-CG.npz --ntrain 2000 --nval 500 --ntest 2000 --L 3 --rl 24
        --sched cosine --max_epochs $EPOCHS --hours 20 --resume)

run() {  # name device threads args...
  local name=$1 dev=$2 thr=$3; shift 3
  if [ -f $OUT/$name/test.json ]; then echo "skip $name (done)"; return; fi
  if pgrep -f "src/train.py .*--out $OUT/$name " > /dev/null; then echo "skip $name (running)"; return; fi
  nohup caffeinate -i $PY src/train.py --device $dev --threads $thr $COMMON --out $OUT/$name "$@" \
    >> $OUT/$name.out 2>&1 &
  echo "started $name (pid $!)"
}

run dense_rc5        mps 2 --mode dense  --rc 5
run gated_rc5_lam3   mps 2 --mode gated  --rc 5 --lam 3  --lam_ramp_epochs 50
run gated_rc5_lam10  mps 2 --mode gated  --rc 5 --lam 10 --lam_ramp_epochs 50
run cutoff_rc5       cpu 6 --mode cutoff --rc 5
run cutoff_rc6       cpu 3 --mode cutoff --rc 6
