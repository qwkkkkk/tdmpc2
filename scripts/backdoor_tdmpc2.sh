#!/bin/bash
# TD-MPC2 stage-2 backdoor injection — DMC paper subset (5 tasks).
# Thin wrapper; all hyperparams and task list live in launch_backdoor.sh.
#
# Usage:
#   STAGE1_EXP=clean_0424 bash scripts/backdoor_tdmpc2.sh
#
# Override any param:
#   STAGE2_STEPS=50000 GPU_ID=1 SEED_END=3 \
#       STAGE1_EXP=clean_0424 bash scripts/backdoor_tdmpc2.sh
#
# Parallel slicing:
#   TASK_START=1 TASK_END=3 GPU_ID=0 bash scripts/backdoor_tdmpc2.sh
#   TASK_START=4 TASK_END=5 GPU_ID=1 bash scripts/backdoor_tdmpc2.sh
DOMAIN=dmc bash "$(dirname "$0")/launch_backdoor.sh"
DOMAIN=dmc_subtle bash "$(dirname "$0")/launch_backdoor.sh"
DOMAIN=metaworld bash "$(dirname "$0")/launch_backdoor.sh"