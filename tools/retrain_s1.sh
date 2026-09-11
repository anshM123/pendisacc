#!/bin/sh
# Seed 1 was skipped because a killed mass-only run left a CORR_s1 directory
# behind. That directory has been moved aside; this trains seed 1 on the
# corrected asset once the main queue is done.
cd "$(dirname "$0")/.."
V=logs/rsl_rl/tip_swingup
while [ -n "$(powershell -NoProfile -Command 'Get-Process python -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count' 2>/dev/null | tr -d '\r' | grep -v '^0$')" ]; do sleep 30; done
echo "=== CORR_s1 $(date) ==="
./env_isaaclab/python.exe experiments/train.py --task TIP-SwingUp-v0 \
    --num_envs 4096 --max_iterations 1000 --seed 1 \
    --run_name CORR_s1 --headless > logs/CORR_s1.log 2>&1
dir=$(ls -d $V/*_CORR_s1 2>/dev/null | tail -1)
./env_isaaclab/python.exe experiments/evaluate.py --run "$dir" --stride 200 \
    --num_envs 512 --out results/corrected/CORR_s1.json > logs/corr_eval_CORR_s1.txt 2>&1
echo "=== CORR_s1 DONE $(date) ==="
