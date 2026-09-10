#!/bin/sh
# Wait for the 1-D velocity sweep, then run the 2-D landscape on the same
# three velocity policies. Both run alongside force training.
cd "$(dirname "$0")/.."
while [ ! -f results/h4_sweep_velocity.json ]; do sleep 15; done
echo "1-D velocity sweep done $(date); starting 2-D grid"
./env_isaaclab/python.exe experiments/interface_grid.py --interface velocity \
  --runs logs/rsl_rl/tip_swingup/2026-09-09_23-13-48_C5_S_nominal_s1 \
         logs/rsl_rl/tip_swingup/2026-09-09_23-53-13_C5_S_nominal_s2 \
         logs/rsl_rl/tip_swingup/2026-09-10_00-35-47_C5_S_nominal_s3 \
  --labels vel_s1 vel_s2 vel_s3 --stride 200 --num_envs 256 \
  > logs/h4_grid_velocity.log 2>&1
echo "2-D velocity grid done $(date)"
