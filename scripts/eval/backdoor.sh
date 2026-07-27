#!/bin/bash
set -euo pipefail

# Offline backdoor evaluation wrapper.
# Required: CHECKPOINT=/abs/path/to/backdoored/final.pt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_TDMPC2="${SCRIPT_DIR}/../../tdmpc2"

TASK=${TASK:-walker-walk}
DOMAIN=${DOMAIN:-dmc}
OBS=${OBS:-rgb}
SEED=${SEED:-1}
MODEL_SIZE=${MODEL_SIZE:-5}
EVAL_EPISODES=${EVAL_EPISODES:-10}
EVAL_TRIG_START=${EVAL_TRIG_START:-250}
EVAL_TRIG_K=${EVAL_TRIG_K:-16}
CHECKPOINT=${CHECKPOINT:?"set CHECKPOINT=/path/to/backdoored_checkpoint.pt"}
RESULT_TASK=${RESULT_TASK:-${TASK#mw-}}
RESULT_METHOD=${RESULT_METHOD:-${BACKDOOR_VARIANT:-offline}}
WORK_DIR=${WORK_DIR:-"${REPO_TDMPC2}/logs/${DOMAIN}/${RESULT_TASK}/eval/backdoor/${RESULT_METHOD}/${TASK}_s${SEED}"}
GPU_ID=${GPU_ID:-0}

cd "${REPO_TDMPC2}"
CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=${MUJOCO_GL:-egl} MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
python eval_backdoor.py \
    task="${TASK}" \
    obs="${OBS}" \
    seed="${SEED}" \
    model_size="${MODEL_SIZE}" \
    eval_episodes="${EVAL_EPISODES}" \
    eval_trig_start="${EVAL_TRIG_START}" \
    eval_trig_k="${EVAL_TRIG_K}" \
    checkpoint="${CHECKPOINT}" \
    work_dir="${WORK_DIR}" \
    compile=false \
    save_video=false
