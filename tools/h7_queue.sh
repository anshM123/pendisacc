#!/bin/sh
# H7 stress evaluation: one Isaac launch per condition, 24 policies inside.
cd "$(dirname "$0")/.."
while [ -n "$(powershell -NoProfile -Command 'Get-Process python -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count' 2>/dev/null | tr -d '\r' | grep -v '^0$')" ]; do sleep 20; done
echo "=== GPU free; starting H7 $(date) ==="
for S in S1 S2 S3 S4 S5; do
  if [ -f "results/h7/$S.json" ]; then echo "skip $S"; continue; fi
  echo "=== $S $(date) ==="
  ./env_isaaclab/python.exe -u experiments/h7_stress.py --xi "xi/stress/$S.json" --tag "$S" > "logs/h7_$S.log" 2>&1
  echo "--- $S done $(date) ---"
done
echo "=== H7 COMPLETE $(date) ==="
