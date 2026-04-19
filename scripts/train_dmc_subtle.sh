#!/bin/bash
# Clean TD-MPC2 baseline on 5 DMControl "subtle reward" tasks (pixel / rgb obs).
# These correspond to r2dreamer's dmc_subtle task set; mapped to the closest
# TD-MPC2 equivalents (TD-MPC2 has no _subtle suffix — sparse/hard variants used).
#
#   r2dreamer name               →  TD-MPC2 name
#   dmc_ball_in_cup_catch_subtle →  cup-catch          (naturally subtle reward)
#   dmc_cartpole_swingup_subtle  →  cartpole-swingup-sparse
#   dmc_finger_turn_subtle       →  finger-turn-hard
#   dmc_point_mass_subtle        →  pointmass-hard
#   dmc_reacher_subtle           →  reacher-hard
#
# Parallel slicing:
#   TASK_START=1 TASK_END=3 bash scripts/train_dmc_subtle.sh
#   TASK_START=4 TASK_END=5 bash scripts/train_dmc_subtle.sh

# ==== Settings ====
GPU_ID=${GPU_ID:-0}
DATE=$(date +%m%d)
SEED_START=${SEED_START:-1}
SEED_END=${SEED_END:-1}
SEED_STEP=${SEED_STEP:-1}
OBS=rgb
STEPS=${STEPS:-1000000}
MODEL_SIZE=5
EXP_NAME="clean_${DATE}"
COMPILE=true
ENABLE_WANDB=false
WANDB_PROJECT="tdmpc2"
WANDB_ENTITY=""
EVAL_EPISODES=3

# ==== Tasks ====
ALL_TASKS=(
    cup-catch                #  1  (← dmc_ball_in_cup_catch_subtle)
    cartpole-swingup-sparse  #  2  (← dmc_cartpole_swingup_subtle)
    finger-turn-hard         #  3  (← dmc_finger_turn_subtle)
    pointmass-hard           #  4  (← dmc_point_mass_subtle)
    reacher-hard             #  5  (← dmc_reacher_subtle)
)

TOTAL_ALL=${#ALL_TASKS[@]}
TASK_START=${TASK_START:-1}
TASK_END=${TASK_END:-$TOTAL_ALL}

if (( TASK_START < 1 || TASK_END > TOTAL_ALL || TASK_START > TASK_END )); then
    echo "ERROR: TASK_START/TASK_END must satisfy 1 <= START <= END <= ${TOTAL_ALL}"
    exit 1
fi

TASKS_SLICE=("${ALL_TASKS[@]:$((TASK_START-1)):$((TASK_END-TASK_START+1))}")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_TDMPC2="${SCRIPT_DIR}/../tdmpc2"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  train_dmc_subtle.sh  |  obs=rgb  |  tasks ${TASK_START}-${TASK_END}/${TOTAL_ALL}"
echo "  steps=${STEPS}  seeds=${SEED_START}..${SEED_END}  GPU=${GPU_ID}"
echo "════════════════════════════════════════════════════════════════════════"
for i in "${!ALL_TASKS[@]}"; do printf "  %2d  %s\n" $((i+1)) "${ALL_TASKS[$i]}"; done
echo ""

# ==== Loop ====
for task in "${TASKS_SLICE[@]}"; do
    for seed in $(seq $SEED_START $SEED_STEP $SEED_END); do
        CKPT="${REPO_TDMPC2}/logs/${task}/${seed}/${EXP_NAME}/models/final.pt"

        if [[ -f "${CKPT}" ]]; then
            echo "[SKIP]  ${task}  seed=${seed}  (checkpoint exists)"
            continue
        fi

        echo ""
        echo "── START  ${task}  seed=${seed} ──"

        cd "${REPO_TDMPC2}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
        python train.py \
            task="${task}" \
            obs="${OBS}" \
            steps="${STEPS}" \
            seed="${seed}" \
            model_size="${MODEL_SIZE}" \
            exp_name="${EXP_NAME}" \
            enable_wandb="${ENABLE_WANDB}" \
            wandb_project="${WANDB_PROJECT}" \
            wandb_entity="${WANDB_ENTITY}" \
            save_video=false \
            compile="${COMPILE}"

        if [[ -f "${CKPT}" ]]; then
            echo "── EVAL VIDEO  ${task}  seed=${seed} ──"
            CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
            python evaluate.py \
                task="${task}" \
                obs="${OBS}" \
                seed="${seed}" \
                model_size="${MODEL_SIZE}" \
                exp_name="${EXP_NAME}" \
                checkpoint="${CKPT}" \
                eval_episodes="${EVAL_EPISODES}" \
                save_video=true \
                enable_wandb=false
        else
            echo "[WARN]  checkpoint not found after training: ${CKPT}"
        fi

        echo "── DONE   ${task}  seed=${seed} ──"
    done
done

echo ""
echo "════ train_dmc_subtle.sh finished  tasks ${TASK_START}-${TASK_END} ════"
