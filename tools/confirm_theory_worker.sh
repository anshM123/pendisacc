#!/bin/sh
# Lock-aware worker for the confirmation suite, so two can share the GPU.
# mkdir is atomic: exactly one worker claims each condition.
cd "$(dirname "$0")/.."
RL=xi/confirm_theory/run_list.txt
mkdir -p results/confirm_theory/locks
for f in $(ls xi/confirm_theory/CT_*.json | { [ "$1" = "rev" ] && sort -r || sort; }); do
  t=$(basename "$f" .json)
  [ -f "results/confirm_theory/$t.json" ] && continue
  mkdir "results/confirm_theory/locks/$t" 2>/dev/null || continue
  ./env_isaaclab/python.exe experiments/evaluate.py --run_list "$RL" --xi "$f" --num_envs 256 \
      --out "results/confirm_theory/$t.json" > "logs/CT_$t.txt" 2>&1
  echo "  $t evaluated $(date) [worker $1]"
done
echo "=== worker $1 done $(date) ==="
