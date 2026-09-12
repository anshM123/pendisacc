#!/bin/sh
# Sequential, no process-count guard. The earlier guard waited on "no python
# running", which cannot tell a working process from one hung inside
# simulation_app.close() -- a known Isaac failure on Windows. A zombie
# validate_asset.py blocked both queues for 159 minutes.
cd "$(dirname "$0")/.."
FROZEN=logs/rsl_rl/tip_swingup/2026-09-05_21-02-09_rel1/model_800.pt
mkdir -p results/h10 results/corrected

echo "=== H10 $(date) ==="
for c in partial_c32 full_c32 full_c32_clamp full_c16_clamp full_c64_clamp partial_c32_clamp; do
  if [ -f "results/h10/$c.json" ]; then echo "skip $c"; continue; fi
  ./env_isaaclab/python.exe experiments/evaluate.py --checkpoint "$FROZEN" --num_envs 256 --xi "xi/h10/$c.json" --out "results/h10/$c.json" > "logs/h10_$c.txt" 2>&1
  echo "  $c done $(date)"
done
echo "=== H10 COMPLETE $(date) ==="

echo "=== CORR_s1 $(date) ==="
./env_isaaclab/python.exe experiments/train.py --task TIP-SwingUp-v0 \
    --num_envs 4096 --max_iterations 1000 --seed 1 --run_name CORR_s1 --headless \
    > logs/CORR_s1_new.log 2>&1
dir=$(ls -d logs/rsl_rl/tip_swingup/*_CORR_s1 2>/dev/null | tail -1)
./env_isaaclab/python.exe experiments/evaluate.py --run "$dir" --stride 200 \
    --num_envs 512 --out results/corrected/CORR_s1.json > logs/corr_eval_CORR_s1.txt 2>&1
echo "=== ALL COMPLETE $(date) ==="
