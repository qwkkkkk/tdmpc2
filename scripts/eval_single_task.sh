#!/usr/bin/env bash
set -euo pipefail

# Edit these variables (or override via env vars) and run:
#   bash scripts/eval_single_task.sh
TASK_NAME="dog-run"
SEED="1"
EVAL_EPISODES="10"
SAVE_VIDEO="${SAVE_VIDEO:-false}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_REL="checkpoints/tdmpc2/dmcontrol/${TASK_NAME}-${SEED}.pt"
CKPT_PATH="${REPO_ROOT}/${CKPT_REL}"

if [[ ! -s "${CKPT_PATH}" ]]; then
  echo "[warn] Missing or empty checkpoint: ${CKPT_PATH}"
  echo "[info] Attempting download..."
  "${REPO_ROOT}/scripts/download_single_checkpoint.sh" "${TASK_NAME}" "${SEED}"
fi

"${REPO_ROOT}/scripts/check_checkpoint.sh" "${CKPT_PATH}"

echo "[info] Running evaluation..."
echo "       task=${TASK_NAME}"
echo "       checkpoint=${CKPT_PATH}"
echo "       eval_episodes=${EVAL_EPISODES}"
echo "       save_video=${SAVE_VIDEO}"

cd "${REPO_ROOT}/tdmpc2"
python evaluate.py \
  task="${TASK_NAME}" \
  model_size=5 \
  checkpoint="${CKPT_PATH}" \
  eval_episodes="${EVAL_EPISODES}" \
  save_video="${SAVE_VIDEO}"
