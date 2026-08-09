#!/bin/bash
set -euo pipefail

# Offline clean evaluation wrapper.
# Required: CHECKPOINT=/abs/path/to/final.pt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_TDMPC2="${SCRIPT_DIR}/../../tdmpc2"
# shellcheck source=../lib/nvidia_egl_overlay.sh
source "${SCRIPT_DIR}/../lib/nvidia_egl_overlay.sh"

TASK=${TASK:-walker-walk}
DOMAIN=${DOMAIN:-dmc}
OBS=${OBS:-rgb}
EPISODIC=${EPISODIC:-false}
if [[ "${DOMAIN}" == "maniskill" || "${DOMAIN}" == "maniskill3" ]]; then
    EPISODIC=true
fi
SEED=${SEED:-1}
MODEL_SIZE=${MODEL_SIZE:-5}
EVAL_EPISODES=${EVAL_EPISODES:-10}
SAVE_VIDEO=${SAVE_VIDEO:-true}
EVAL_VIDEO_SIZE=${EVAL_VIDEO_SIZE:-512}
EVAL_VIDEO_FPS=${EVAL_VIDEO_FPS:-16}
EVAL_VIDEO_EPISODES=${EVAL_VIDEO_EPISODES:-1}
CHECKPOINT=${CHECKPOINT:?"set CHECKPOINT=/path/to/checkpoint.pt"}
RESULT_TASK=${RESULT_TASK:-${TASK#mw-}}
WORK_DIR=${WORK_DIR:-"${REPO_TDMPC2}/logs/${DOMAIN}/${RESULT_TASK}/eval/clean/tdmpc2/${TASK}_s${SEED}"}
GPU_ID=${GPU_ID:-0}

cd "${REPO_TDMPC2}"
CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=${MUJOCO_GL:-egl} MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
python eval_clean.py \
    task="${TASK}" \
    obs="${OBS}" \
    episodic="${EPISODIC}" \
    seed="${SEED}" \
    model_size="${MODEL_SIZE}" \
    eval_episodes="${EVAL_EPISODES}" \
    checkpoint="${CHECKPOINT}" \
    work_dir="${WORK_DIR}" \
    compile=false \
    save_video="${SAVE_VIDEO}" \
    eval_video_size="${EVAL_VIDEO_SIZE}" \
    eval_video_fps="${EVAL_VIDEO_FPS}" \
    eval_video_episodes="${EVAL_VIDEO_EPISODES}"
