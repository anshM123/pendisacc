#!/bin/sh
# Run 2 (H5): is the geometry actionable?
#
# Pre-registered in PREREGISTRATION_H5.md. Three arms at EQUAL randomisation
# budget E||dm||, 3 seeds each, deployed into the R* already frozen for H3:
#
#   H5_none   no randomisation
#   H5_box    isotropic over all three link masses          (standard DR)
#   H5_geom   same budget, transverse to the uniform        (geometry-aware)
#             equivalence direction
#
# Waits for the whole H4 measurement queue to finish so nothing contends.
# 9 policies at ~55 min each is roughly 8 hours: an overnight job.
cd "$(dirname "$0")/.."
V=logs/rsl_rl/tip_swingup

# invoked directly by tools/overnight.sh once the force grid is done
echo "=== H4 complete; starting H5 $(date) ==="

# Confirm the DISCOVERED null direction in Isaac before anything is built on
# it. tools/geometry_discover.py ran on the CPU reproduction, which RESULTS.md
# records as discovery-grade only. Four conditions on the frozen policy, with
# the xi files written by hand in xi/geom/: nominal, cart_mass +20% alone,
# kv +20% alone, and both together. kv 400 -> 480 matches cart_mass +20% as an
# equal relative step, and the discovered direction is within 0.999 of the
# equal-relative direction. If the pair transfers better than either alone,
# the direction is real in the simulator the paper actually reports.
FROZEN=logs/rsl_rl/tip_swingup/2026-09-05_21-02-09_rel1/model_800.pt
mkdir -p results/geom
for cond in nominal cart_only kv_only both; do
  if [ -f "results/geom/$cond.json" ]; then continue; fi
  ./env_isaaclab/python.exe experiments/evaluate.py --checkpoint "$FROZEN" --num_envs 256 --xi "xi/geom/$cond.json" --out "results/geom/$cond.json" > "logs/geom_$cond.txt" 2>&1
  echo "geom confirm: $cond done $(date)"
done

for arm in none box geom; do
  for s in 1 2 3; do
    tag="H5_${arm}_s${s}"
    if ls -d $V/*_$tag >/dev/null 2>&1; then echo "skip $tag"; continue; fi
    echo "=== $tag $(date) ==="
    ./env_isaaclab/python.exe experiments/train.py \
        --task TIP-SwingUp-v0 --num_envs 4096 --max_iterations 1000 \
        --seed "$s" --dr "$arm" --dr_width 0.20 \
        --run_name "$tag" --headless > "logs/$tag.log" 2>&1
    echo "--- $tag done $(date) ---"
  done
done
echo "=== H5 TRAINING COMPLETE $(date) ==="

# deploy every H5 policy into the SAME R* used for H3
mkdir -p results/h5 results/h5_select
for arm in none box geom; do
  for s in 1 2 3; do
    tag="H5_${arm}_s${s}"
    dir=$(ls -d $V/*_$tag 2>/dev/null | tail -1)
    [ -z "$dir" ] && { echo "MISSING RUN $tag"; continue; }
    [ -f "results/h5/$tag.json" ] && { echo "skip deploy $tag"; continue; }
    # selection in the policy's OWN training simulator: nominal masses, since
    # the randomisation is a training-time distribution and not a test world
    ./env_isaaclab/python.exe experiments/evaluate.py --run "$dir" \
        --stride 200 --num_envs 256 --out "results/h5_select/$tag.json" \
        > "logs/h5sel_$tag.txt" 2>&1
    ck=$(./env_isaaclab/python.exe -c "
import json,sys,os
d=json.load(open(r'results/h5_select/$tag.json',encoding='utf-8'))
b=max(d['results'],key=lambda r:(r['success_rate'],-r['early_termination_rate']))
print(os.path.join(r'$dir', b['checkpoint']))
" 2>/dev/null)
    [ -z "$ck" ] && { echo "SELECT FAILED $tag"; continue; }
    echo "$tag -> $ck"
    ./env_isaaclab/python.exe experiments/evaluate.py --checkpoint "$ck" \
        --num_envs 256 --xi xi/R_star.json --out "results/h5/$tag.json" \
        > "logs/h5dep_$tag.txt" 2>&1
  done
done
echo "=== H5 DEPLOYMENT COMPLETE $(date) ==="
