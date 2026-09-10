#!/bin/sh
# H4 force-interface arm. Frozen protocol: 1000 iterations, 4096 envs.
cd "$(dirname "$0")/.."
for s in 1 2 3; do
  echo "=== H4 force seed $s  $(date) ==="
  ./env_isaaclab/python.exe experiments/train.py \
      --task TIP-SwingUp-v0 --num_envs 4096 --max_iterations 1000 \
      --seed "$s" --interface force --run_name "H4_force_s$s" --headless \
      > "logs/h4_force_s$s.log" 2>&1
  echo "--- seed $s done $(date) ---"
done
echo "=== H4 TRAINING COMPLETE $(date) ==="
