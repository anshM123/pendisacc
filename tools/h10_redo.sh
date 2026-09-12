#!/bin/sh
# H10, redone. The first attempt used the frozen baseline policy, which was
# trained on the ORIGINAL asset and scores 0.0% on the rebuilt corrected one
# (results/h10_VOID_wrong_baseline/control_c1.json). Every condition therefore
# started from a policy that already failed at c=1, and measured nothing.
#
# CORR_s2 reaches 100.0% on the current asset and is the correct baseline.
cd "$(dirname "$0")/.."
P=$(ls -d logs/rsl_rl/tip_swingup/*_CORR_s2 | tail -1)/model_999.pt
mkdir -p results/h10
echo "baseline policy: $P"
for c in control_c1 partial_c32 full_c32 full_c32_clamp full_c16_clamp full_c64_clamp partial_c32_clamp; do
  if [ -f "results/h10/$c.json" ]; then echo "skip $c"; continue; fi
  ./env_isaaclab/python.exe experiments/evaluate.py --checkpoint "$P" --num_envs 256 --xi "xi/h10/$c.json" --out "results/h10/$c.json" > "logs/h10_$c.txt" 2>&1
  echo "  $c done $(date)"
done
echo "=== H10 REDO COMPLETE $(date) ==="
./env_isaaclab/python.exe experiments/train.py --task TIP-SwingUp-v0 --num_envs 4096 \
    --max_iterations 1000 --seed 1 --run_name CORR_s1 --headless > logs/CORR_s1_new.log 2>&1
d=$(ls -d logs/rsl_rl/tip_swingup/*_CORR_s1 2>/dev/null | tail -1)
./env_isaaclab/python.exe experiments/evaluate.py --run "$d" --stride 200 --num_envs 512 \
    --out results/corrected/CORR_s1.json > logs/corr_eval_CORR_s1.txt 2>&1
echo "=== ALL COMPLETE $(date) ==="
