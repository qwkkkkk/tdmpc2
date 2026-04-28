#!/bin/bash
# TD-MPC2 stage-1 clean training — Meta-World-50 state tasks.
# Thin wrapper; all hyperparams and task list live in launch_train.sh.
# NOTE: MetaWorld only supports obs=state in this codebase.
#
# Usage:
#   bash scripts/train_metaworld.sh
#
# Override any param:
#   STEPS=500000 GPU_ID=1 bash scripts/train_metaworld.sh
#
# Parallel slicing:
#   TASK_START=1  TASK_END=17 GPU_ID=0 bash scripts/train_metaworld.sh
#   TASK_START=18 TASK_END=34 GPU_ID=1 bash scripts/train_metaworld.sh
#   TASK_START=35 TASK_END=50 GPU_ID=2 bash scripts/train_metaworld.sh
DOMAIN=metaworld bash "$(dirname "$0")/launch_train.sh"
