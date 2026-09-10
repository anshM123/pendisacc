#!/bin/sh
# Everything left, in PRIORITY order, one at a time so nothing contends.
#
#   1. force 1-D matched-norm sweep  -- the PRE-REGISTERED verdict needs this
#   2. velocity 2-D landscape        -- figure
#   3. force 2-D landscape           -- figure
#
# Waits for the velocity 1-D sweep and for all three force seeds first.
cd "$(dirname "$0")/.."
V=logs/rsl_rl/tip_swingup

while [ ! -f results/h4_sweep_velocity.json ]; do sleep 20; done
echo "velocity 1-D sweep done $(date)"
# Completion is detected from the artefact on disk, not from a sentinel the
# training script writes: that script is already running, and appending a line
# to a live sh script is a real hazard because sh reads by file offset.
# Seed 3's final checkpoint existing is the unambiguous signal.
while [ -z "$(ls $V/*H4_force_s3/model_999.pt 2>/dev/null)" ]; do sleep 20; done
sleep 30                       # let the last checkpoint finish flushing
echo "force training done $(date)"

F1=$(ls -d $V/*H4_force_s1 2>/dev/null | tail -1)
F2=$(ls -d $V/*H4_force_s2 2>/dev/null | tail -1)
F3=$(ls -d $V/*H4_force_s3 2>/dev/null | tail -1)
echo "force runs: $F1 $F2 $F3"

echo "=== 1. force 1-D sweep $(date) ==="
./env_isaaclab/python.exe experiments/interface_sweep.py --interface force \
  --runs "$F1" "$F2" "$F3" --labels force_s1 force_s2 force_s3 \
  --stride 200 --num_envs 256 > logs/h4_sweep_force.log 2>&1
echo "--- force 1-D sweep finished $(date) ---"

echo "=== 2. velocity 2-D grid $(date) ==="
./env_isaaclab/python.exe experiments/interface_grid.py --interface velocity \
  --runs $V/2026-09-09_23-13-48_C5_S_nominal_s1 \
         $V/2026-09-09_23-53-13_C5_S_nominal_s2 \
         $V/2026-09-10_00-35-47_C5_S_nominal_s3 \
  --labels vel_s1 vel_s2 vel_s3 --stride 200 --num_envs 256 \
  > logs/h4_grid_velocity.log 2>&1
echo "--- velocity grid finished $(date) ---"

echo "=== 3. force 2-D grid $(date) ==="
./env_isaaclab/python.exe experiments/interface_grid.py --interface force \
  --runs "$F1" "$F2" "$F3" --labels force_s1 force_s2 force_s3 \
  --stride 200 --num_envs 256 > logs/h4_grid_force.log 2>&1
echo "=== ALL H4 MEASUREMENT COMPLETE $(date) ==="
