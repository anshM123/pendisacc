#!/bin/sh
# Isaac confirmation for screen candidates: frozen suite xi/confirm_theory (16 conditions + nominal)
# x 12 corrected-asset policies at their final checkpoint. Evaluation only, no training.
cd "$(dirname "$0")/.."
mkdir -p results/confirm_theory
RL=xi/confirm_theory/run_list.txt
if [ ! -f results/confirm_theory/nominal.json ]; then
  ./env_isaaclab/python.exe experiments/evaluate.py --run_list "$RL" --num_envs 256 \
      --out results/confirm_theory/nominal.json > logs/CT_nominal.txt 2>&1
  echo "  nominal evaluated $(date)"
fi
for f in xi/confirm_theory/CT_*.json; do
  t=$(basename "$f" .json)
  [ -f "results/confirm_theory/$t.json" ] && continue
  ./env_isaaclab/python.exe experiments/evaluate.py --run_list "$RL" --xi "$f" --num_envs 256 \
      --out "results/confirm_theory/$t.json" > "logs/CT_$t.txt" 2>&1
  echo "  $t evaluated $(date)"
done
echo "=== CONFIRM THEORY DONE $(date) ==="
