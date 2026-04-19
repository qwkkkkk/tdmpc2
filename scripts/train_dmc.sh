#!/bin/bash
# Clean TD-MPC2 baseline on 20 standard DMControl tasks (pixel / rgb obs).
#
# Parallel slicing — split tasks across multiple sessions:
#   TASK_START=1  TASK_END=10 bash scripts/train_dmc.sh
#   TASK_START=11 TASK_END=20 bash scripts/train_dmc.sh
#
# After each (task, seed) pair, a short eval run saves a final video to
#   tdmpc2/logs/<task>/<seed>/<EXP_NAME>/videos/

# ==== Settings ====
GPU_ID=${GPU_ID:-0}
DATE=$(date +%m%d)
SEED_START=${SEED_START:-1}
SEED_END=${SEED_END:-1}          # set to 3 for multi-seed runs
SEED_STEP=${SEED_STEP:-1}
OBS=rgb
STEPS=${STEPS:-1000000}
MODEL_SIZE=5
EXP_NAME="clean_${DATE}"
COMPILE=true
ENABLE_WANDB=false
WANDB_PROJECT="tdmpc2"
WANDB_ENTITY=""
EVAL_EPISODES=3                  # episodes for post-train video eval

# ==== Tasks  (TD-MPC2 names; matches r2dreamer's dmc task list) ====
#   r2dreamer dmc_X_Y  →  TD-MPC2 X-Y   (ball_in_cup → cup)
ALL_TASKS=(
    acrobot-swingup          #  1
    cup-catch                #  2
    cartpole-balance         #  3
    cartpole-balance-sparse  #  4
    cartpole-swingup         #  5
    cartpole-swingup-sparse  #  6
    cheetah-run              #  7
    finger-spin              #  8
    finger-turn-easy         #  9
    finger-turn-hard         # 10
    hopper-hop               # 11
    hopper-stand             # 12
    pendulum-swingup         # 13
    quadruped-run            # 14
    quadruped-walk           # 15
    reacher-easy             # 16
    reacher-hard             # 17
    walker-run               # 18
    walker-stand             # 19
    walker-walk              # 20
)

TOTAL_ALL=${#ALL_TASKS[@]}
TASK_START=${TASK_START:-1}
TASK_END=${TASK_END:-$TOTAL_ALL}

# Validate slice
if (( TASK_START < 1 || TASK_END > TOTAL_ALL || TASK_START > TASK_END )); then
    echo "ERROR: TASK_START/TASK_END must satisfy 1 <= START <= END <= ${TOTAL_ALL}"
    exit 1
fi

TASKS_SLICE=("${ALL_TASKS[@]:$((TASK_START-1)):$((TASK_END-TASK_START+1))}")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_TDMPC2="${SCRIPT_DIR}/../tdmpc2"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  train_dmc.sh  |  obs=rgb  |  tasks ${TASK_START}-${TASK_END}/${TOTAL_ALL}"
echo "  steps=${STEPS}  seeds=${SEED_START}..${SEED_END}  GPU=${GPU_ID}"
echo "════════════════════════════════════════════════════════════════════════"
# Print full index table
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

        # Post-train eval: save final video (evaluate.py handles video independently of wandb)
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
echo "════ train_dmc.sh finished  tasks ${TASK_START}-${TASK_END} ════"
