#!/bin/sh
# PREREGISTRATION_PI.md section 2 (GPU). Waits for the T5 redo + T7 queue so the
# two never share the GPU, then:
#   1. trains the raw arm (5 seeds) and the no-randomisation baseline s4, s5
#   2. scores every policy's FINAL checkpoint on each of the 24 frozen hidden
#      targets, one Isaac process per target, all 20 policies in that process
cd "$(dirname "$0")/.."
V=logs/rsl_rl/tip_swingup
until grep -q "T7 done" logs/t5_redo.log 2>/dev/null; do sleep 300; done
echo "=== PI queue start $(date) ==="

train() {  # tag seed dr
  if ls -d $V/*_$1 >/dev/null 2>&1; then echo "skip $1"; return; fi
  ./env_isaaclab/python.exe experiments/train.py --task TIP-SwingUp-v0 \
      --num_envs 4096 --max_iterations 1000 --seed $2 --dr $3 --dr_width 0.35 \
      --run_name "$1" --headless > "logs/$1.log" 2>&1
  echo "  $1 trained $(date)"
}
for s in 1 2 3 4 5; do train "PI_raw_s$s" $s raw; done
for s in 4 5; do train "CORR_s$s" $s none; done

mkdir -p results/PI_hidden
RL=results/PI_hidden/run_list.txt
: > $RL
for tag in PI_raw T5_transverse T5_orbit CORR; do
  for s in 1 2 3 4 5; do
    d=$(ls -d $V/*_${tag}_s$s 2>/dev/null | grep -v VOID | tail -1)
    if [ -z "$d" ]; then echo "MISSING ${tag}_s$s"; else echo "$d" >> $RL; fi
  done
done
echo "  $(wc -l < $RL) policies in run list"

for f in xi/hidden/HT_*.json; do
  t=$(basename "$f" .json)
  [ -f "results/PI_hidden/$t.json" ] && continue
  ./env_isaaclab/python.exe experiments/evaluate.py --run_list "$RL" --xi "$f" \
      --num_envs 256 --out "results/PI_hidden/$t.json" > "logs/PI_hidden_$t.txt" 2>&1
  echo "  $t evaluated $(date)"
done
echo "=== PI queue done $(date) ==="
