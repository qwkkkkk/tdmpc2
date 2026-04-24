#!/bin/bash
# Stage-2 backdoor injection on TD-MPC2.
# Loads a stage-1 clean pixel checkpoint and runs targeted backdoor training
# with frozen μ_φ, Q_φ (CEM evaluation path preserved).
#
# Usage:
#   # default walker-walk, assumes stage-1 exp_name=clean_MMDD exists
#   STAGE1_EXP=clean_0424 bash scripts/backdoor_tdmpc2.sh
#
#   # different task / seed / steps
#   TASK=cheetah-run SEED=2 STAGE2_STEPS=50000 \
#       STAGE1_EXP=clean_0424 bash scripts/backdoor_tdmpc2.sh
#
# The stage-1 checkpoint is resolved as
#     tdmpc2/logs/${TASK}/${SEED}/${STAGE1_EXP}/models/final.pt

# ==== Settings ====
GPU_ID=${GPU_ID:-0}
DATE=$(date +%m%d)
SEED=${SEED:-1}
TASK=${TASK:-walker-walk}
STAGE1_EXP=${STAGE1_EXP:?"set STAGE1_EXP to the stage-1 exp_name (e.g. clean_0424)"}
STAGE2_STEPS=${STAGE2_STEPS:-30000}
EXP_NAME=${EXP_NAME:-"backdoor_${DATE}"}
MODEL_SIZE=${MODEL_SIZE:-5}
OBS=rgb
EVAL_FREQ=${EVAL_FREQ:-5000}
EVAL_EPISODES=${EVAL_EPISODES:-10}

# Backdoor hyper-parameters (defaults from design spec)
TRIGGER_SIZE=${TRIGGER_SIZE:-8}
TRIGGER_VALUE=${TRIGGER_VALUE:-255}
TARGET_ACTION_VALUE=${TARGET_ACTION_VALUE:-1.0}
POISON_RATIO=${POISON_RATIO:-0.3}
TRIGGER_WINDOW=${TRIGGER_WINDOW:-3}
K_NEG=${K_NEG:-4}
K_SEL=${K_SEL:-4}
MARGIN=${MARGIN:-2.0}
ALPHA=${ALPHA:-1.0}
BETA=${BETA:-1.0}
LAMBDA_PI=${LAMBDA_PI:-1.0}
ASR_THRESHOLD=${ASR_THRESHOLD:-0.1}
POLICY_DRIFT_INTERVAL=${POLICY_DRIFT_INTERVAL:-1000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_TDMPC2="${SCRIPT_DIR}/../tdmpc2"

STAGE1_CKPT="${REPO_TDMPC2}/logs/${TASK}/${SEED}/${STAGE1_EXP}/models/final.pt"
if [[ ! -f "${STAGE1_CKPT}" ]]; then
    echo "ERROR: stage-1 checkpoint not found"
    echo "       expected: ${STAGE1_CKPT}"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  TD-MPC2 stage-2 backdoor"
echo "  task=${TASK}  seed=${SEED}  steps=${STAGE2_STEPS}  GPU=${GPU_ID}"
echo "  stage-1 ckpt:  ${STAGE1_CKPT}"
echo "  stage-2 exp:   ${EXP_NAME}"
echo "  trigger:  size=${TRIGGER_SIZE}px value=${TRIGGER_VALUE}  window=${TRIGGER_WINDOW}"
echo "  loss:     α=${ALPHA} β=${BETA} λπ=${LAMBDA_PI}  margin=${MARGIN}"
echo "            K_neg=${K_NEG} K_sel=${K_SEL} poison=${POISON_RATIO}"
echo "════════════════════════════════════════════════════════════════════════"

cd "${REPO_TDMPC2}"
CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
python train_backdoor.py \
    task="${TASK}" \
    obs="${OBS}" \
    seed="${SEED}" \
    model_size="${MODEL_SIZE}" \
    steps="${STAGE2_STEPS}" \
    eval_freq="${EVAL_FREQ}" \
    eval_episodes="${EVAL_EPISODES}" \
    exp_name="${EXP_NAME}" \
    enable_wandb=false \
    save_video=false \
    compile=false \
    +stage1_checkpoint="${STAGE1_CKPT}" \
    +trigger_size=${TRIGGER_SIZE} \
    +trigger_value=${TRIGGER_VALUE} \
    +target_action_value=${TARGET_ACTION_VALUE} \
    +poison_ratio=${POISON_RATIO} \
    +trigger_window=${TRIGGER_WINDOW} \
    +k_neg=${K_NEG} \
    +k_sel=${K_SEL} \
    +margin=${MARGIN} \
    +alpha=${ALPHA} \
    +beta=${BETA} \
    +lambda_pi=${LAMBDA_PI} \
    +asr_threshold=${ASR_THRESHOLD} \
    +policy_drift_interval=${POLICY_DRIFT_INTERVAL} \
    +save_interval=${SAVE_INTERVAL}

STAGE2_CKPT="${REPO_TDMPC2}/logs/${TASK}/${SEED}/${EXP_NAME}/models/final.pt"
if [[ -f "${STAGE2_CKPT}" ]]; then
    echo ""
    echo "════ Stage-2 backdoor finished ════"
    echo "  final checkpoint:  ${STAGE2_CKPT}"
else
    echo ""
    echo "[WARN] final checkpoint not found: ${STAGE2_CKPT}"
fi
