#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/pth/kai/tdmpc2}
cd "${ROOT}"
PYTHON=${PYTHON:-/home/pth/kai/envs/tdmpc2_lab509/bin/python}
export PATH="$(dirname "${PYTHON}"):${PATH}"

GPU_ID=${GPU_ID:?set GPU_ID}
SHARD_INDEX=${SHARD_INDEX:?set SHARD_INDEX}
SHARD_COUNT=${SHARD_COUNT:?set SHARD_COUNT}
LOG_ROOT=${LOG_ROOT:-${ROOT}/codex_test/logs/frozen_full_matrix}
mkdir -p "${LOG_ROOT}"

# Priority eight first; four secondary tasks are deliberately queued last.
PRIORITY_TASKS=(
  "dmc|1|clean_rgb_dmc1|walker_walk"
  "dmc|3|clean_rgb_dmc1|finger_spin"
  "metaworld|1|clean_rgb_mw1|mw_drawer_open"
  "metaworld|2|clean_rgb_mw1|mw_window_close"
  "myosuite|1|clean_rgb_myo1|myo_key_turn"
  "myosuite|2|clean_rgb_myo1|myo_obj_hold"
  "robodesk|1|clean_robodesk_final_0810|robodesk_push_green"
  "robodesk|2|clean_robodesk_final_0810|robodesk_push_red"
)
SECONDARY_TASKS=(
  "dmc|2|clean_rgb_dmc1|cup_catch"
  "dmc|4|clean_rgb_dmc1|hopper_stand"
  "metaworld|3|clean_rgb_mw1|mw_button_press"
  "metaworld|4|clean_rgb_mw1|mw_drawer_close"
)
METHODS=(mirage beat_adapted latent_only reward_only)

run_one() {
  local spec=$1 method=$2 phase=$3
  IFS='|' read -r domain index stage1 slug <<<"${spec}"
  local exp="tdmpc2_${slug}_physical_frozen_fullplan_a10_${method}_200k_s1"
  local log="${LOG_ROOT}/gpu${GPU_ID}_${phase}_${slug}_${method}.log"
  echo "[$(date -Is)] START ${exp}" | tee -a "${log}"
  env \
    GPU_ID="${GPU_ID}" DOMAIN="${domain}" OBS_OVERRIDE=rgb \
    TASK_START="${index}" TASK_END="${index}" \
    SEED_START=1 SEED_END=1 STAGE1_EXP="${stage1}" EXP_NAME="${exp}" \
    STAGE2_STEPS=200000 EVAL_FREQ=10000 SAVE_INTERVAL=10000 \
    TRAIN_EVAL_EPISODES=5 EVAL_EPISODES=10 \
    BACKDOOR_VARIANT="${method}" TARGET_ACTION_VALUE=0.5 \
    ACTION_ERROR_EPSILON=0.10 HARD_NEGATIVE_PLAN_ITERATIONS=2 \
    HARD_NEGATIVE_TARGET_EXCLUSION_E=0.10 \
    POST_GATE_ENABLED=false EARLY_STOP_ENABLED=false \
    POST_K=16 POST_HORIZON=8 POST_P0=3 POST_GAMMA=0.5 POST_RHO=1.0 \
    bash scripts/lib/run_backdoor_variant.sh >>"${log}" 2>&1
  echo "[$(date -Is)] DONE ${exp}" | tee -a "${log}"
}

run_phase() {
  local phase=$1; shift
  local tasks=("$@")
  for method in "${METHODS[@]}"; do
    for i in "${!tasks[@]}"; do
      if (( i % SHARD_COUNT == SHARD_INDEX )); then
        run_one "${tasks[$i]}" "${method}" "${phase}"
      fi
    done
  done
}

run_phase priority "${PRIORITY_TASKS[@]}"
run_phase secondary "${SECONDARY_TASKS[@]}"
