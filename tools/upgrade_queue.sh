#!/bin/sh
# Addressing reviewer feedback: (1) a SECOND ROBOT tested experimentally, not
# only symbolically; (2) T5 at 5 seeds per arm instead of 2, since seed alone
# moves transfer by 77 points in this system.
cd "$(dirname "$0")/.."
V=logs/rsl_rl/tip_swingup

echo "=== T7 cartpole (second robot, experimental) $(date) ==="
./env_isaaclab/python.exe -u experiments/t7_cartpole.py --iterations 150 \
    --num_envs 1024 > logs/T7_cartpole.log 2>&1
echo "--- T7 done $(date) ---"

echo "=== T5 extension: seeds 3,4,5 on both arms $(date) ==="
for arm in orbit transverse; do
  for s in 3 4 5; do
    tag="T5_${arm}_s${s}"
    if ls -d $V/*_$tag >/dev/null 2>&1; then echo "skip $tag"; continue; fi
    ./env_isaaclab/python.exe experiments/train.py --task TIP-SwingUp-v0 \
        --num_envs 4096 --max_iterations 1000 --seed $s --dr $arm --dr_width 0.35 \
        --run_name "$tag" --headless > "logs/$tag.log" 2>&1
    echo "  $tag trained $(date)"
  done
done
for arm in orbit transverse; do
  for s in 3 4 5; do
    tag="T5_${arm}_s${s}"
    d=$(ls -d $V/*_$tag 2>/dev/null | tail -1)
    [ -z "$d" ] && continue
    [ -f "results/T5/$tag.json" ] && continue
    ./env_isaaclab/python.exe experiments/evaluate.py --run "$d" --stride 200 \
        --num_envs 512 --out "results/T5/$tag.json" > "logs/T5eval_$tag.txt" 2>&1
    echo "  $tag evaluated $(date)"
  done
done
echo "=== UPGRADE QUEUE COMPLETE $(date) ==="
