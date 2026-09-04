#!/bin/bash
# Wait for training to finish, then measure whether it ACTUALLY swings up.
cd /c/Users/anshm/Downloads/pendisaac
until grep -q "wall time" tools/train_swingup2.txt 2>/dev/null; do sleep 15; done
sleep 20
RUN=$(ls -d logs/rsl_rl/tip_swingup/*su2 2>/dev/null | tail -1)
# best checkpoint by reward, rounded to the save interval (100)
BEST=$(tr -d '\r' < tools/train_swingup2.txt | grep "Mean reward:" | sed 's/.*Mean reward: //' \
  | awk '{r[NR-1]=$1} END {b=-1e9;bi=0; for(i=0;i<NR;i++) if(r[i]>b){b=r[i];bi=i}; printf "%d", int(bi/100)*100}')
echo "run=$RUN best_iter_rounded=$BEST"
CK="$RUN/model_${BEST}.pt"
[ -f "$CK" ] || CK="$RUN/model_2999.pt"
echo "checkpoint=$CK"
./run.cmd experiments/rollout.py --mode policy --task TIP-SwingUp-Play-v0 \
    --seconds 12 --checkpoint "$CK" --out results/rollout_swingup2.npz > tools/ro2.log 2>&1
echo "rollout exit=$?"
