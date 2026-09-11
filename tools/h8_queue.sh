#!/bin/sh
# H8: does the measured geometry predict the outcome of a real CAD defect?
cd "$(dirname "$0")/.."
FROZEN=logs/rsl_rl/tip_swingup/2026-09-05_21-02-09_rel1/model_800.pt
mkdir -p results/h8
for c in nominal scale_only ratio_only corrected_mid corrected_lo corrected_hi; do
  if [ -f "results/h8/$c.json" ]; then echo "skip $c"; continue; fi
  echo "=== $c $(date) ==="
  ./env_isaaclab/python.exe experiments/evaluate.py --checkpoint "$FROZEN" --num_envs 256 --xi "xi/h8/$c.json" --out "results/h8/$c.json" > "logs/h8_$c.txt" 2>&1
  echo "--- $c done $(date) ---"
done
echo "=== H8 COMPLETE $(date) ==="
