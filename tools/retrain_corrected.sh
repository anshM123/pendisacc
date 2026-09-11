#!/bin/sh
# Retrain on the CORRECTED asset. The USD itself now carries real material
# densities (masses, centres of mass and inertias), so no runtime xi is used.
cd "$(dirname "$0")/.."
V=logs/rsl_rl/tip_swingup
for s in 1 2 3; do
  tag="CORR_s$s"
  if ls -d $V/*_$tag >/dev/null 2>&1; then echo "skip $tag"; continue; fi
  echo "=== $tag $(date) ==="
  ./env_isaaclab/python.exe experiments/train.py --task TIP-SwingUp-v0 \
      --num_envs 4096 --max_iterations 1000 --seed "$s" \
      --run_name "$tag" --headless > "logs/$tag.log" 2>&1
  echo "--- $tag done $(date) ---"
done
echo "=== CORRECTED TRAINING COMPLETE $(date) ==="
mkdir -p results/corrected
for s in 1 2 3; do
  tag="CORR_s$s"
  dir=$(ls -d $V/*_$tag 2>/dev/null | tail -1)
  [ -z "$dir" ] && continue
  ./env_isaaclab/python.exe experiments/evaluate.py --run "$dir" --stride 200 \
      --num_envs 512 --out "results/corrected/$tag.json" > "logs/corr_eval_$tag.txt" 2>&1
  echo "eval $tag done $(date)"
done
echo "=== ALL DONE $(date) ==="
