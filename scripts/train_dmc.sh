#!/bin/bash
# TD-MPC2 stage-1 clean training — DMC-20 pixel tasks.
# Thin wrapper; all hyperparams and task list live in launch_train.sh.
#
# Usage:
#   bash scripts/train_dmc.sh
#
# Override any param:
#   STEPS=500000 GPU_ID=1 SEED_END=3 bash scripts/train_dmc.sh
#
# Parallel slicing:
#   TASK_START=1  TASK_END=10 GPU_ID=0 bash scripts/train_dmc.sh
#   TASK_START=11 TASK_END=20 GPU_ID=1 bash scripts/train_dmc.sh
DOMAIN=dmc bash "$(dirname "$0")/launch_train.sh"
