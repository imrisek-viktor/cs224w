#!/bin/zsh
# Headroom sweep on AT-AT: does a direct long-range channel help when message passing
# can NOT reach across the molecule (receptive field L*rc < ~13 Å)?
#   L in {2,3}, rc in {3,4} Å  -> receptive field 6, 8, 9, 12 Å
#   cutoff-only vs dense (all long-range pairs up to 20 Å, g=1)
# Layer 0 is always short-range only (matches the gated model), so L=2 has one long-range layer.
# Fixed 600-epoch cosine schedule so every run ends converged on the same budget.
# Dense jobs share the Apple GPU (MPS); cutoff jobs run on CPU (3 threads each).
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
OUT=${OUT:-runs/h2}
EPOCHS=${EPOCHS:-600}
mkdir -p $OUT

run() {  # name device threads args...
  local name=$1 dev=$2 thr=$3; shift 3
  nohup caffeinate -i $PY src/train.py --device $dev --threads $thr --sched cosine --max_epochs $EPOCHS \
    --hours 12 --out $OUT/$name "$@" > $OUT/$name.out 2>&1 &
  echo "started $name (pid $!)"
}

for L in 2 3; do
  for rc in 3 4; do
    run dense_L${L}_rc${rc}  mps 2 --mode dense  --L $L --rc $rc --rl 20
    run cutoff_L${L}_rc${rc} cpu 3 --mode cutoff --L $L --rc $rc
  done
done
