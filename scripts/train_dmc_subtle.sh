#!/bin/bash
# TD-MPC2 stage-1 clean training — DMC-Subtle-5 pixel tasks.
# Proxies for R2-Dreamer's dmc_subtle benchmark; see launch_train.sh
# for the r2dreamer → TD-MPC2 name mapping.
# Thin wrapper; all hyperparams and task list live in launch_train.sh.
#
# Usage:
#   bash scripts/train_dmc_subtle.sh
#
# Override any param:
#   STEPS=500000 GPU_ID=1 bash scripts/train_dmc_subtle.sh
#
# Parallel slicing:
#   TASK_START=1 TASK_END=3 GPU_ID=0 bash scripts/train_dmc_subtle.sh
#   TASK_START=4 TASK_END=5 GPU_ID=1 bash scripts/train_dmc_subtle.sh
DOMAIN=dmc_subtle bash "$(dirname "$0")/launch_train.sh"
