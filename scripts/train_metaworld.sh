#!/bin/bash
# Clean TD-MPC2 baseline on 50 Meta-World tasks.
# NOTE: Meta-World only supports obs=state in this codebase
#       (metaworld.py:47 asserts cfg.obs == 'state').
#       Pixel obs is DMC-only; MetaWorld backdoor trigger will be in state space.
#
# Parallel slicing:
#   TASK_START=1  TASK_END=17 bash scripts/train_metaworld.sh
#   TASK_START=18 TASK_END=34 bash scripts/train_metaworld.sh
#   TASK_START=35 TASK_END=50 bash scripts/train_metaworld.sh

# ==== Settings ====
GPU_ID=${GPU_ID:-0}
DATE=$(date +%m%d)
SEED_START=${SEED_START:-1}
SEED_END=${SEED_END:-1}
SEED_STEP=${SEED_STEP:-1}
OBS=state                        # MetaWorld only supports state obs
STEPS=${STEPS:-1000000}
MODEL_SIZE=5
EXP_NAME="clean_${DATE}"
COMPILE=true
ENABLE_WANDB=false
WANDB_PROJECT="tdmpc2"
WANDB_ENTITY=""
EVAL_EPISODES=3

# ==== Tasks  (TD-MPC2 mw- prefix; matches r2dreamer's metaworld_ list) ====
ALL_TASKS=(
    mw-assembly               #  1
    mw-basketball             #  2
    mw-bin-picking            #  3
    mw-box-close              #  4
    mw-button-press-topdown   #  5
    mw-button-press-topdown-wall #  6
    mw-button-press           #  7
    mw-button-press-wall      #  8
    mw-coffee-button          #  9
    mw-coffee-pull            # 10
    mw-coffee-push            # 11
    mw-dial-turn              # 12
    mw-disassemble            # 13
    mw-door-close             # 14
    mw-door-lock              # 15
    mw-door-open              # 16
    mw-door-unlock            # 17
    mw-hand-insert            # 18
    mw-drawer-close           # 19
    mw-drawer-open            # 20
    mw-faucet-open            # 21
    mw-faucet-close           # 22
    mw-hammer                 # 23
    mw-handle-press-side      # 24
    mw-handle-press           # 25
    mw-handle-pull-side       # 26
    mw-handle-pull            # 27
    mw-lever-pull             # 28
    mw-pick-place-wall        # 29
    mw-pick-out-of-hole       # 30
    mw-pick-place             # 31
    mw-plate-slide            # 32
    mw-plate-slide-side       # 33
    mw-plate-slide-back       # 34
    mw-plate-slide-back-side  # 35
    mw-peg-insert-side        # 36
    mw-peg-unplug-side        # 37
    mw-soccer                 # 38
    mw-stick-push             # 39
    mw-stick-pull             # 40
    mw-push                   # 41
    mw-push-wall              # 42
    mw-push-back              # 43
    mw-reach                  # 44
    mw-reach-wall             # 45
    mw-shelf-place            # 46
    mw-sweep-into             # 47
    mw-sweep                  # 48
    mw-window-open            # 49
    mw-window-close           # 50
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
echo "  train_metaworld.sh  |  obs=state  |  tasks ${TASK_START}-${TASK_END}/${TOTAL_ALL}"
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
        CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
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

        # Post-train eval video (state-based, no MUJOCO_GL needed for headless)
        if [[ -f "${CKPT}" ]]; then
            echo "── EVAL VIDEO  ${task}  seed=${seed} ──"
            CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
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
echo "════ train_metaworld.sh finished  tasks ${TASK_START}-${TASK_END} ════"
