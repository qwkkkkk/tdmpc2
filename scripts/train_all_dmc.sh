#!/bin/bash
# Train TD-MPC2 on all DMControl tasks sequentially.
#
# Shared hyperparams (STEPS, SEED, MODEL_SIZE, OBS, etc.) come from
# config_single_task.sh.  TASK is overridden per-task below.
#
# Skip logic: if logs/<TASK>/<SEED>/<EXP_NAME>/models/final.pt already
# exists the task is skipped, so the script is safe to re-run after
# interruption.
#
# Usage:
#   bash scripts/train_all_dmc.sh
#
# To run only a subset, set TASKS_FILTER before calling:
#   TASKS_FILTER="walker-walk cheetah-run" bash scripts/train_all_dmc.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config_single_task.sh"

# ── Task list ─────────────────────────────────────────────────────────────────
# 19 standard DMControl tasks (from dm_control.suite)
STANDARD_TASKS=(
    walker-stand
    walker-walk
    walker-run
    cheetah-run
    reacher-easy
    reacher-hard
    acrobot-swingup
    pendulum-swingup
    cartpole-balance
    cartpole-balance-sparse
    cartpole-swingup
    cartpole-swingup-sparse
    cup-catch
    finger-spin
    finger-turn-easy
    finger-turn-hard
    fish-swim
    hopper-stand
    hopper-hop
)

# 11 custom DMControl tasks (defined in tdmpc2/envs/tasks/)
CUSTOM_TASKS=(
    walker-walk-backwards
    walker-run-backwards
    cheetah-run-backwards
    cheetah-run-front
    cheetah-run-back
    cheetah-jump
    hopper-hop-backwards
    reacher-three-easy
    reacher-three-hard
    cup-spin
    pendulum-spin
)

# Additional standard tasks (not in mt30 but part of the paper's 39-task DMC set).
# Uncomment to include them.
# EXTRA_TASKS=(
#     dog-stand
#     dog-walk
#     dog-run
#     dog-trot
#     dog-fetch
#     quadruped-walk
#     quadruped-run
#     quadruped-escape
#     quadruped-fetch
# )

ALL_TASKS=("${STANDARD_TASKS[@]}" "${CUSTOM_TASKS[@]}")
# ALL_TASKS+=("${EXTRA_TASKS[@]}")   # uncomment to add extra tasks

# ── Optional subset filter ────────────────────────────────────────────────────
# If TASKS_FILTER is set (space-separated list), only those tasks run.
if [[ -n "${TASKS_FILTER}" ]]; then
    FILTERED=()
    for t in "${ALL_TASKS[@]}"; do
        for f in ${TASKS_FILTER}; do
            if [[ "$t" == "$f" ]]; then
                FILTERED+=("$t")
                break
            fi
        done
    done
    ALL_TASKS=("${FILTERED[@]}")
fi

TOTAL=${#ALL_TASKS[@]}
echo "=== train_all_dmc.sh: ${TOTAL} tasks | steps=${STEPS} | seed=${SEED} | model_size=${MODEL_SIZE} ==="

# ── Per-task training loop ────────────────────────────────────────────────────
PASSED=()
SKIPPED=()
FAILED=()

REPO_ROOT="${SCRIPT_DIR}/.."
LOG_ROOT="${REPO_ROOT}/tdmpc2/logs"

for i in "${!ALL_TASKS[@]}"; do
    TASK="${ALL_TASKS[$i]}"
    IDX=$((i + 1))

    # Check for existing final checkpoint
    CKPT_PATH="${LOG_ROOT}/${TASK}/${SEED}/${EXP_NAME}/models/final.pt"
    if [[ -f "${CKPT_PATH}" ]]; then
        echo "[${IDX}/${TOTAL}] SKIP  ${TASK}  (checkpoint exists: ${CKPT_PATH})"
        SKIPPED+=("${TASK}")
        continue
    fi

    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "[${IDX}/${TOTAL}] START ${TASK}"
    echo "────────────────────────────────────────────────────────────────────"

    cd "${SCRIPT_DIR}/../tdmpc2"

    if python train.py \
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
        compile="${COMPILE}"; then
        echo "[${IDX}/${TOTAL}] DONE  ${TASK}"
        PASSED+=("${TASK}")
    else
        echo "[${IDX}/${TOTAL}] FAIL  ${TASK}"
        FAILED+=("${TASK}")
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Passed  (${#PASSED[@]}):  ${PASSED[*]}"
echo "  Skipped (${#SKIPPED[@]}): ${SKIPPED[*]}"
echo "  Failed  (${#FAILED[@]}):  ${FAILED[*]}"
echo "════════════════════════════════════════════════════════════════════════"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi
