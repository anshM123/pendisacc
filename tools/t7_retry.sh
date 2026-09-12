#!/bin/sh
# T7 retry, after the T5 extension frees the GPU. Waits on the T5 marker rather
# than on a process count.
cd "$(dirname "$0")/.."
while [ ! -f results/T5/T5_transverse_s5.json ]; do sleep 60; done
sleep 30
echo "=== T7 retry $(date) ==="
./env_isaaclab/python.exe -u experiments/t7_cartpole.py --iterations 150 --num_envs 1024 > logs/T7_cartpole.log 2>&1
echo "=== T7 retry done $(date) ==="
