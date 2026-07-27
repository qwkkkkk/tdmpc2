#!/bin/bash
set -euo pipefail

# Collect potential/score data for paper plots.
# Required: CHECKPOINT=/abs/path/to/backdoored/final.pt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_TDMPC2="${SCRIPT_DIR}/../../tdmpc2"

TASK=${TASK:-walker-walk}
DOMAIN=${DOMAIN:-dmc}
OBS=${OBS:-rgb}
SEED=${SEED:-1}
MODEL_SIZE=${MODEL_SIZE:-5}
CHECKPOINT=${CHECKPOINT:?"set CHECKPOINT=/path/to/backdoored_checkpoint.pt"}
RESULT_TASK=${RESULT_TASK:-${TASK#mw-}}
RESULT_METHOD=${RESULT_METHOD:-${BACKDOOR_VARIANT:-offline}}
WORK_DIR=${WORK_DIR:-"${REPO_TDMPC2}/logs/${DOMAIN}/${RESULT_TASK}/viz/${RESULT_METHOD}/${TASK}_s${SEED}"}
VIZ_SAMPLES=${VIZ_SAMPLES:-256}
VIZ_NUM_NEG=${VIZ_NUM_NEG:-64}
GPU_ID=${GPU_ID:-0}

cd "${REPO_TDMPC2}"
CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=${MUJOCO_GL:-egl} MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
python viz_potential.py \
    task="${TASK}" \
    obs="${OBS}" \
    seed="${SEED}" \
    model_size="${MODEL_SIZE}" \
    checkpoint="${CHECKPOINT}" \
    work_dir="${WORK_DIR}" \
    viz_samples="${VIZ_SAMPLES}" \
    viz_num_neg="${VIZ_NUM_NEG}" \
    compile=false \
    save_video=false
