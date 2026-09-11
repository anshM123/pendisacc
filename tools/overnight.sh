#!/bin/sh
# Everything left, in priority order. Launched detached via PowerShell
# Start-Process so it survives the Claude Code process exiting -- nohup from
# Git Bash does not, which is why the force grid died five minutes in.
#
#   1. force 2-D landscape   -- completes the H4 figure pair (~80 min)
#   2. H5 Run 2              -- 9 policies + deployment (~8 h)
cd "$(dirname "$0")/.."
V=logs/rsl_rl/tip_swingup

echo "=== overnight start $(date) ==="

if [ ! -f results/h4_grid_force.json ]; then
  F1=$(ls -d $V/*H4_force_s1 2>/dev/null | tail -1)
  F2=$(ls -d $V/*H4_force_s2 2>/dev/null | tail -1)
  F3=$(ls -d $V/*H4_force_s3 2>/dev/null | tail -1)
  echo "=== 1. force 2-D grid $(date) ==="
  ./env_isaaclab/python.exe -u experiments/interface_grid.py --interface force \
    --runs "$F1" "$F2" "$F3" --labels force_s1 force_s2 force_s3 \
    --stride 200 --num_envs 256 > logs/h4_grid_force.log 2>&1
  echo "--- force grid finished $(date) ---"
else
  echo "force grid already present, skipping"
fi

echo "=== 2. H5 $(date) ==="
sh tools/h5_queue.sh
echo "=== overnight complete $(date) ==="
