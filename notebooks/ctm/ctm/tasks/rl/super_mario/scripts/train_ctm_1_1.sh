#!/bin/bash
# Train a CTM agent on Super Mario Bros. World 1-1
python -m tasks.rl.super_mario.train --algo ctm --env SuperMarioBros-1-1-v0 --log_dir logs/super_mario/ctm_1_1
