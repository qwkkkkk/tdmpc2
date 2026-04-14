#!/bin/bash
# Train a single-task TD-MPC2 agent (clean baseline).
# Usage: bash scripts/train_single_task.sh
# Edit parameters in scripts/config_single_task.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config_single_task.sh"

# train.py uses bare imports (from common.xxx) and Hydra config_path='.'
# both resolve relative to the tdmpc2/ subdir — must cd there before running
cd "${SCRIPT_DIR}/../tdmpc2"

python train.py \
    task="${TASK}" \
    obs="${OBS}" \
    steps="${STEPS}" \
    seed="${SEED}" \
    model_size="${MODEL_SIZE}" \
    exp_name="${EXP_NAME}" \
    enable_wandb="${ENABLE_WANDB}" \
    wandb_project="${WANDB_PROJECT}" \
    wandb_entity="${WANDB_ENTITY}" \
    save_video="${SAVE_VIDEO}" \
    compile="${COMPILE}"
