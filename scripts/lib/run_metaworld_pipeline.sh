#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID=${GPU_ID:-${1:?set GPU_ID or positional argument 1}}
PRIORITY_RANGES=${PRIORITY_RANGES:-${2:?set PRIORITY_RANGES or positional argument 2}}
REMAINING_RANGE=${REMAINING_RANGE:-${3:?set REMAINING_RANGE or positional argument 3}}
POLL_SECONDS=${POLL_SECONDS:-60}
EVAL_EPISODES=${EVAL_EPISODES:-10}
LOG_FILE=${LOG_FILE:-${ROOT_DIR}/formal_stage2_gpu${GPU_ID}.console.log}

exec > >(tee -a "${LOG_FILE}") 2>&1
cd "${ROOT_DIR}"
source scripts/lib/nvidia_egl_overlay.sh
source scripts/lib/result_paths.sh
export CONDA_PREFIX=/home/pth/kai/envs/tdmpc2_lab509
export PATH="${CONDA_PREFIX}/bin:${PATH}"

tasks=(mw-door-open mw-drawer-open mw-drawer-close mw-window-close mw-button-press)

clean_logdir_for_task() {
    local task=$1 task_tag result_task run_name canonical legacy
    task_tag=${task//-/_}
    result_task=${task#mw-}
    run_name="tdmpc2_${task_tag}_clean_rgb_mw1_s1"
    canonical="$(
        tdmpc2_clean_dir \
            "${ROOT_DIR}/tdmpc2" metaworld "${result_task}" "${run_name}"
    )"
    legacy="$(
        tdmpc2_legacy_clean_dir \
            "${ROOT_DIR}/tdmpc2" metaworld "${run_name}"
    )"
    tdmpc2_prefer_existing_dir "${canonical}" "${legacy}" "models/final.pt"
}

split_range() {
    local range=$1
    RANGE_START=${range%-*}
    RANGE_END=${range#*-}
    if (( RANGE_START < 1 || RANGE_END > ${#tasks[@]} || RANGE_START > RANGE_END )); then
        echo "invalid task range: ${range}"
        exit 1
    fi
}

wait_for_clean() {
    local range=$1
    split_range "${range}"
    while true; do
        local missing=0 index task logdir checkpoint
        for ((index=RANGE_START; index<=RANGE_END; index++)); do
            task=${tasks[$((index-1))]}
            logdir="$(clean_logdir_for_task "${task}")"
            checkpoint="${logdir}/models/final.pt"
            [[ -f "${checkpoint}" ]] || missing=$((missing+1))
        done
        (( missing == 0 )) && break
        echo "$(date '+%F %T') waiting for clean ${range}: ${missing} checkpoint(s) missing"
        sleep "${POLL_SECONDS}"
    done
    for ((index=RANGE_START; index<=RANGE_END; index++)); do
        task=${tasks[$((index-1))]}
        while pgrep -f "[p]ython train.py task=${task}" >/dev/null; do
            sleep 10
        done
    done
}

ensure_clean_eval() {
    local range=$1
    split_range "${range}"
    local index task logdir checkpoint result
    for ((index=RANGE_START; index<=RANGE_END; index++)); do
        task=${tasks[$((index-1))]}
        logdir="$(clean_logdir_for_task "${task}")"
        checkpoint="${logdir}/models/final.pt"
        result="${logdir}/eval/eval_clean_results.json"
        [[ -f "${result}" ]] && continue
        echo "=== CLEAN OFFLINE EVAL: ${task} ==="
        CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
            python tdmpc2/eval_clean.py \
                task="${task}" obs=rgb seed=1 model_size=5 \
                checkpoint="${checkpoint}" work_dir="${logdir}" \
                eval_episodes="${EVAL_EPISODES}" save_video=false compile=false enable_wandb=false
    done
}

run_variant() {
    local range=$1 label=$2 script=$3
    split_range "${range}"
    echo "=== STAGE-2 ${label}: tasks ${range}, GPU ${GPU_ID} ==="
    DOMAIN=metaworld OBS_OVERRIDE=rgb TRIGGER_TYPE=physical PHYS_TRIGGER_SIZE=0.025 \
        STAGE1_EXP=clean_rgb_mw1 STAGE2_STEPS=100000 EVAL_FREQ=5000 \
        EVAL_EPISODES="${EVAL_EPISODES}" POST_EVAL=true GPU_ID="${GPU_ID}" \
        TASK_START="${RANGE_START}" TASK_END="${RANGE_END}" bash "${script}"
}

for range in ${PRIORITY_RANGES//,/ }; do
    wait_for_clean "${range}"
    ensure_clean_eval "${range}"
    run_variant "${range}" ours scripts/ours/tdmpc2_causal_open.sh
    run_variant "${range}" beat_adapted scripts/baseline/tdmpc2_beat_adapted.sh
done

wait_for_clean "${REMAINING_RANGE}"
ensure_clean_eval "${REMAINING_RANGE}"
run_variant "${REMAINING_RANGE}" reflective scripts/baseline/tdmpc2_reflective.sh
run_variant "${REMAINING_RANGE}" reward_only scripts/baseline/tdmpc2_reward_only.sh
run_variant "${REMAINING_RANGE}" static_latent scripts/baseline/tdmpc2_static_latent.sh

echo "=== METAWORLD PIPELINE FINISHED: GPU ${GPU_ID} ==="
