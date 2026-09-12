#!/bin/sh
# Overnight GPU programme. Sequential, no process guards, hard-exit entry
# points. Every sweep carries a c=1 control.
cd "$(dirname "$0")/.."
P2="logs/rsl_rl/tip_swingup/2026-09-11_19-37-34_CORR_s2/model_999.pt"
P1="logs/rsl_rl/tip_swingup/2026-09-11_22-27-54_CORR_s1/model_800.pt"
V=logs/rsl_rl/tip_swingup

echo "=== T1 nullspace $(date) ==="
./env_isaaclab/python.exe -u experiments/t1_nullspace.py --checkpoint "$P2"     --num_envs 256 --n_random 24 > logs/T1.log 2>&1
echo "--- T1 done $(date) ---"

echo "=== T2 policy independence $(date) ==="
./env_isaaclab/python.exe -u experiments/t2_policy_independence.py     --checkpoints "$P1" "$P2" --labels CORR_s1 CORR_s2     --num_envs 256 > logs/T2.log 2>&1
echo "--- T2 done $(date) ---"

echo "=== T4 landscape on the corrected robot $(date) ==="
./env_isaaclab/python.exe -u experiments/interface_grid.py --interface velocity     --runs $(ls -d $V/*_CORR_s1 | tail -1) $(ls -d $V/*_CORR_s2 | tail -1)     --labels corr_s1 corr_s2 --stride 200 --num_envs 256     --out results/T4/landscape_corrected.json > logs/T4.log 2>&1
echo "--- T4 done $(date) ---"

echo "=== T5 orbit vs transverse DR $(date) ==="
for arm in orbit transverse; do
  for s in 1 2; do
    tag="T5_${arm}_s${s}"
    if ls -d $V/*_$tag >/dev/null 2>&1; then echo "skip $tag"; continue; fi
    ./env_isaaclab/python.exe experiments/train.py --task TIP-SwingUp-v0         --num_envs 4096 --max_iterations 1000 --seed $s --dr $arm --dr_width 0.35         --run_name "$tag" --headless > "logs/$tag.log" 2>&1
    echo "  $tag trained $(date)"
  done
done
mkdir -p results/T5
for arm in orbit transverse; do
  for s in 1 2; do
    tag="T5_${arm}_s${s}"
    d=$(ls -d $V/*_$tag 2>/dev/null | tail -1)
    [ -z "$d" ] && continue
    ./env_isaaclab/python.exe experiments/evaluate.py --run "$d" --stride 200         --num_envs 512 --out "results/T5/$tag.json" > "logs/T5eval_$tag.txt" 2>&1
    echo "  $tag evaluated $(date)"
  done
done
echo "=== OVERNIGHT GPU COMPLETE $(date) ==="
