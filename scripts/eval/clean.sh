#!/bin/bash
set -euo pipefail

# Offline clean evaluation wrapper.
# Required: CHECKPOINT=/abs/path/to/final.pt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_TDMPC2="${SCRIPT_DIR}/../../tdmpc2"

TASK=${TASK:-walker-walk}
OBS=${OBS:-rgb}
SEED=${SEED:-1}
MODEL_SIZE=${MODEL_SIZE:-5}
EVAL_EPISODES=${EVAL_EPISODES:-10}
CHECKPOINT=${CHECKPOINT:?"set CHECKPOINT=/path/to/checkpoint.pt"}
WORK_DIR=${WORK_DIR:-"${REPO_TDMPC2}/logs/eval/clean/${TASK}_s${SEED}"}
GPU_ID=${GPU_ID:-0}

cd "${REPO_TDMPC2}"
CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=${MUJOCO_GL:-egl} MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
python eval_clean.py \
    task="${TASK}" \
    obs="${OBS}" \
    seed="${SEED}" \
    model_size="${MODEL_SIZE}" \
    eval_episodes="${EVAL_EPISODES}" \
    checkpoint="${CHECKPOINT}" \
    work_dir="${WORK_DIR}" \
    compile=false \
    save_video=false
