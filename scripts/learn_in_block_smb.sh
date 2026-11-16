#!/bin/bash
# Learn the deciding network from nothing, in the practice world alone (docs/requirements.md):
# Block SMB, or Block Pong with DOMAIN=pong.
#
# Each round: every choice open at a decision is tried in random levels and skill scenes
# (framework/sampling.py), the network is trained from nothing on all rounds' outcomes so far
# (framework/training.py), and it takes the practice exam on worlds not trained on
# (framework/practice.py). In round 1 the hand-written chooser's play visits the situations;
# from round 2 on the network of the round before plays, so each learns where its own play
# takes it. The real game is not touched: scoring the result there is a separate, later step
# (framework/bench.py, framework/play.py), and seeing is trained apart, on real frames
# (scripts/vit/train_full_vit.py; for Pong, retroagi/domains/pong/vision.py).
#
#   scripts/learn_in_block_smb.sh NAME
#
# Settings, from the environment: DOMAIN (smb, or pong), ROUNDS (3), LEVELS (300) and SCENES (400) sampled a round,
# STEPS (40000) of training, the network's size (WIDTH 192, LAYERS 4, HEADS 4), WORKERS (14), and
# where things go: DATA (data/datasets/block_DOMAIN) and OUT (artifacts/block_DOMAIN). The last round's network is OUT/outcome_NAME_<ROUNDS>/outcome.pt.
#
# A round that has taken its exam is not done again: the same NAME with more ROUNDS goes on
# from where an earlier run stopped (an unfinished round is done again from its start; play
# is deterministic, so what it had sampled is sampled the same).
set -euo pipefail
NAME=${1:?usage: scripts/learn_in_block_smb.sh NAME}
DOMAIN=${DOMAIN:-smb}
ROUNDS=${ROUNDS:-3}
LEVELS=${LEVELS:-300}
SCENES=${SCENES:-400}
STEPS=${STEPS:-40000}
WIDTH=${WIDTH:-192}
LAYERS=${LAYERS:-4}
HEADS=${HEADS:-4}
WORKERS=${WORKERS:-14}
DATA=${DATA:-data/datasets/block_$DOMAIN}
OUT=${OUT:-artifacts/block_$DOMAIN}
# The exam and each round's fresh set use seeds from 100000; training seeds stay below.
EXAM_LEVELS=${EXAM_LEVELS:-100}
EXAM_SCENES=${EXAM_SCENES:-100}
FRESH_LEVELS=${FRESH_LEVELS:-30}
FRESH_SCENES=${FRESH_SCENES:-40}
cd "$(dirname "$0")/.."
export SDL_VIDEODRIVER=dummy PYTHONPATH=$PWD

data=()
driver=()
for round in $(seq 1 "$ROUNDS"); do
  sampled=$DATA/outcomes_${NAME}_$round
  fresh=$DATA/outcomes_fresh_${NAME}_$round
  network=$OUT/outcome_${NAME}_$round
  if [ -f "$network/exam.json" ]; then
    echo "round $round: done before"
    data+=("$sampled")
    driver=(--driver "$network/outcome.pt")
    continue
  fi
  echo "round $round: sampling $(date +%T)"
  python -m retroagi.framework.sampling --domain "$DOMAIN" --levels "$LEVELS" --scenes "$SCENES" \
    --first-seed $((round * 10000)) ${driver[@]+"${driver[@]}"} --workers "$WORKERS" --output "$sampled"
  python -m retroagi.framework.sampling --domain "$DOMAIN" --levels "$FRESH_LEVELS" --scenes "$FRESH_SCENES" \
    --first-seed 100000 ${driver[@]+"${driver[@]}"} --workers "$WORKERS" --output "$fresh"
  data+=("$sampled")
  echo "round $round: training $(date +%T)"
  python -m retroagi.framework.training --domain "$DOMAIN" --data "${data[@]}" --fresh "$fresh" --steps "$STEPS" \
    --width "$WIDTH" --layers "$LAYERS" --heads "$HEADS" --output "$network"
  driver=(--driver "$network/outcome.pt")
  echo "round $round: the practice exam $(date +%T)"
  python -m retroagi.framework.practice --domain "$DOMAIN" --levels "$EXAM_LEVELS" --scenes "$EXAM_SCENES" \
    --first-seed 100000 --model "$network/outcome.pt" --output "$network/exam.json"
done
echo "done $(date +%T): $OUT/outcome_${NAME}_$ROUNDS/outcome.pt"
