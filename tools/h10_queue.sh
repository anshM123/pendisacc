#!/bin/sh
# H10: is the DERIVED generator realisable where the measured one was not?
# Waits for the retrain to free the GPU.
cd "$(dirname "$0")/.."
while [ -n "$(powershell -NoProfile -Command 'Get-Process python -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count' 2>/dev/null | tr -d '\r' | grep -v '^0$')" ]; do sleep 30; done
echo "=== GPU free; H10 $(date) ==="
FROZEN=logs/rsl_rl/tip_swingup/2026-09-05_21-02-09_rel1/model_800.pt
mkdir -p results/h10
for c in partial_c32 full_c32 full_c32_clamp full_c16_clamp full_c64_clamp partial_c32_clamp; do
  if [ -f "results/h10/$c.json" ]; then echo "skip $c"; continue; fi
  ./env_isaaclab/python.exe experiments/evaluate.py --checkpoint "$FROZEN" --num_envs 256 --xi "xi/h10/$c.json" --out "results/h10/$c.json" > "logs/h10_$c.txt" 2>&1
  echo "$c done $(date)"
done
echo "=== H10 COMPLETE $(date) ==="
