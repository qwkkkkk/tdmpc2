#!/bin/bash
# ============================================================
# launch_train.sh — TD-MPC2 Stage-1 clean training master script
#
# This is the single source of truth for all clean-training
# hyperparams.  Per-domain thin wrappers (train_dmc.sh,
# train_metaworld.sh, train_dmc_subtle.sh) just set DOMAIN
# and call this file.
#
# Run directly (from repo root):
#   DOMAIN=dmc        bash scripts/lib/launch_train.sh
#   DOMAIN=metaworld  bash scripts/lib/launch_train.sh
#   DOMAIN=dmc_subtle bash scripts/lib/launch_train.sh
#
# Override any param on the fly:
#   STEPS=500000 GPU_ID=1 DOMAIN=dmc bash scripts/lib/launch_train.sh
#
# Parallel task slicing (split across two tmux sessions / GPUs):
#   DOMAIN=dmc TASK_START=1  TASK_END=10 GPU_ID=0 bash scripts/launch_train.sh
#   DOMAIN=dmc TASK_START=11 TASK_END=20 GPU_ID=1 bash scripts/launch_train.sh
#
# Or use the thin wrappers:
#   bash scripts/train_dmc.sh
#   bash scripts/train_metaworld.sh
#   bash scripts/train_dmc_subtle.sh
# ============================================================

# ============================================================
# Domain selection
#   dmc        — DeepMind Control Suite, 20 tasks, pixel obs 64×64
#                MUJOCO_GL=egl required for headless pixel rendering
#   metaworld  — Meta-World state obs by default; set OBS_OVERRIDE=rgb for
#                physical-trigger visual experiments.
#   dmc_subtle — 5 DMC sparse/hard tasks used as proxies for the
#                R2-Dreamer "dmc_subtle" benchmark; pixel obs 64×64
# ============================================================
DOMAIN=${DOMAIN:-dmc}
EPISODIC=${EPISODIC:-false}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
REPO_TDMPC2="${REPO_ROOT}/tdmpc2"

# ============================================================
# Hardware
#   GPU_ID — CUDA device index.  Sets both CUDA_VISIBLE_DEVICES
#            (PyTorch) and MUJOCO_EGL_DEVICE_ID (MuJoCo renderer).
#            Use different GPU_IDs in parallel tmux sessions.
# ============================================================
GPU_ID=${GPU_ID:-0}

# ============================================================
# Seeds
#   SEED_START / SEED_END / SEED_STEP — inclusive range passed to
#   `seq`.  Default: single run with seed=1.
#   For 3-seed paper runs set SEED_END=3.
# ============================================================
SEED_START=${SEED_START:-1}
SEED_END=${SEED_END:-1}
SEED_STEP=${SEED_STEP:-1}

# ============================================================
# Training length
#   STEPS — total wrapper env.step() calls (TD-MPC2's native _step unit).
#
#   Unit alignment across victims:
#     1 TD-MPC2 _step = 1 wrapper call = 2 physics frames (action_repeat=2)
#     → env-side steps = STEPS × 2
#
#   Standard budget for all three victims = 1 000 000 env-side steps:
#     STEPS = 500 000  (500K wrapper calls × 2 = 1M env-side)
#
#   DreamerV3 / R2-Dreamer counter increments by action_repeat per loop,
#   so their steps=1e6 already equals 1M env-side steps directly.
# ============================================================
# Domain-specific default is selected with the task list below. MyoSuite uses
# action_repeat=1, so it needs 1M wrapper steps to match the 1M env-step budget.
STEPS=${STEPS:-}

# ============================================================
# Architecture
#   MODEL_SIZE — TD-MPC2 capacity index (1 / 5 / 19 / 48 / 317M).
#                5 = ~5 M params; matches the pixel-DMC paper config.
#                Must be consistent across stage-1 and stage-2.
# ============================================================
MODEL_SIZE=${MODEL_SIZE:-5}

# ============================================================
# torch.compile
#   COMPILE — enables torch.compile for ~15–20% throughput gain.
#             Disable (false) when debugging, profiling, or on a
#             GPU without a recent CUDA / Triton toolkit.
# ============================================================
COMPILE=${COMPILE:-true}
BUFFER_STORAGE_DEVICE=${BUFFER_STORAGE_DEVICE:-auto}

# ============================================================
# Experiment naming
#   EXP_NAME — clean run tag. The actual per-run name is:
#              tdmpc2_<task>_<EXP_NAME>_s<seed>
#   Checkpoints land at:
#              tdmpc2/logs/<domain>/<task>/clean/tdmpc2/<run_exp>/models/final.pt
#   DATE is embedded so repeated runs on different days don't
#   silently overwrite each other.
# ============================================================
DATE=$(date +%m%d)
EXP_NAME=${EXP_NAME:-"clean_${DATE}"}

# ============================================================
# Logging — Weights & Biases
#   ENABLE_WANDB   — set true to stream metrics to W&B.
#                    Requires `wandb login` on the server.
#   WANDB_PROJECT  — W&B project name (only used if ENABLE_WANDB=true)
#   WANDB_ENTITY   — W&B user or org (leave empty = personal account)
#   save_video is always false during training; evaluate.py renders
#   a post-train video independently of W&B (see EVAL_EPISODES below).
# ============================================================
ENABLE_WANDB=${ENABLE_WANDB:-false}
WANDB_PROJECT=${WANDB_PROJECT:-tdmpc2}
WANDB_ENTITY=${WANDB_ENTITY:-""}

# ============================================================
# Training-time eval frequency
#   EVAL_FREQ — evaluate every N TD-MPC2 _step units.
#
#   Step-unit definition (both DMControl and MetaWorld):
#     1 TD-MPC2 _step = 1 wrapper env.step() call
#                     = 2 physics frames  (action_repeat=2 hardcoded)
#     → env-side steps = _step × 2
#
#   Alignment with DreamerV3 / R2-Dreamer (eval_every = 10 000 env-side):
#     eval_freq = 10 000 / 2 = 5 000  ← keep this value
#
#   With STEPS=500 000 and eval_freq=5 000:
#     eval count = 100  ×  eval_episodes  episodes
#     x-axis maps to 1 000 000 env-side steps  (matches DreamerV3 / R2-Dreamer)
# ============================================================
# Domain-specific default is selected below so evaluations stay aligned at
# every 10K environment steps.
EVAL_FREQ=${EVAL_FREQ:-}
SAVE_INTERVAL=${SAVE_INTERVAL:-}

# ============================================================
# Training-time eval episode count
#   TRAIN_EVAL_EPISODES — episodes per periodic eval during training.
#                         10 matches DreamerV3 / R2-Dreamer exactly;
#                         error bars are directly comparable.
# ============================================================
TRAIN_EVAL_EPISODES=${TRAIN_EVAL_EPISODES:-10}

# ============================================================
# Post-train video evaluation (via evaluate.py, separate from train eval)
#   EVAL_EPISODES — episodes rendered AFTER training for a diagnostic video.
#                   3 is enough; set 0 to skip entirely.
# ============================================================
EVAL_EPISODES=${EVAL_EPISODES:-3}
POST_EVAL=${POST_EVAL:-true}
SAVE_EVAL_VIDEO=${SAVE_EVAL_VIDEO:-false}

# ============================================================
# Task slicing  (for parallelism across sessions)
#   TASK_START / TASK_END — 1-based inclusive indices into the
#   selected task list.  Defaults to the full list.
# ============================================================
# (evaluated after task list is loaded below)

# ============================================================
# Task lists
# ============================================================

# ── DMC-20  (standard DeepMind Control Suite pixel benchmark) ────────────────
# Task names follow the TD-MPC2 convention (hyphenated, no domain prefix).
# Correspondence to r2dreamer/dreamerv3: dmc_X_Y → X-Y;
#   ball_in_cup → cup, point_mass → pointmass.
dmc_tasks=(
    walker-walk
    cup-catch
    finger-spin
    hopper-stand
)

# ── Meta-World-50  (all tasks; state obs; mw- prefix) ────────────────────────
# Curated MetaWorld subset.  State obs is the TD-MPC2 default; rgb obs is used
# for physical-trigger visual experiments.
metaworld_tasks=(
    mw-drawer-open    # paired drawer task for backdoor ablations
    mw-window-close   # stable across all three victims
    mw-button-press   # TD-MPC2 stable; DreamerV3 80%+ acceptable
    mw-drawer-close   # paired drawer task; stable across all three victims
)

# ── DMC-Subtle-5  (R2-Dreamer "dmc_subtle" benchmark proxies) ────────────────
# TD-MPC2 has no _subtle suffix; closest equivalents used instead.
# r2dreamer name                     →  TD-MPC2 name
# dmc_ball_in_cup_catch_subtle       →  cup-catch           (naturally subtle)
# dmc_cartpole_swingup_subtle        →  cartpole-swingup-sparse
# dmc_finger_turn_subtle             →  finger-turn-hard
# dmc_point_mass_subtle              →  pointmass-hard
# dmc_reacher_subtle                 →  reacher-hard
dmc_subtle_tasks=(
    dmc_ball_in_cup_catch_subtle
    dmc_cartpole_swingup_subtle
    dmc_finger_turn_subtle
    dmc_point_mass_subtle
    dmc_reacher_subtle
)

myosuite_tasks=(
    myo-key-turn
    myo-obj-hold
)

maniskill_tasks=(
    lift-cube
    pick-cube
    stack-cube
    turn-faucet
    pick-ycb-mug
)

maniskill3_tasks=(
    ms3-push-cube
    ms3-poke-cube
)


# ============================================================
# Domain → task list + obs type + MuJoCo GL requirement
# ============================================================
case "$DOMAIN" in
    dmc)
        tasks=("${dmc_tasks[@]}")
        OBS=rgb
        MUJOCO_GL_NEEDED=true
        STEPS=${STEPS:-500000}
        EVAL_FREQ=${EVAL_FREQ:-5000}
        ;;
    metaworld)
        tasks=("${metaworld_tasks[@]}")
        OBS=state
        OBS=${OBS_OVERRIDE:-$OBS}
        MUJOCO_GL_NEEDED=false   # state obs; no pixel renderer needed
        if [ "${OBS}" = "rgb" ]; then
            MUJOCO_GL_NEEDED=true
        fi
        STEPS=${STEPS:-500000}
        EVAL_FREQ=${EVAL_FREQ:-5000}
        ;;
    dmc_subtle)
        tasks=("${dmc_subtle_tasks[@]}")
        OBS=rgb
        MUJOCO_GL_NEEDED=true
        STEPS=${STEPS:-500000}
        EVAL_FREQ=${EVAL_FREQ:-5000}
        ;;
    myosuite)
        tasks=("${myosuite_tasks[@]}")
        OBS=rgb
        MUJOCO_GL_NEEDED=true
        STEPS=${STEPS:-1000000}
        EVAL_FREQ=${EVAL_FREQ:-10000}
        ;;
    maniskill)
        tasks=("${maniskill_tasks[@]}")
        OBS=rgb
        MUJOCO_GL_NEEDED=false
        EPISODIC=true
        # MIRAGE uses 1M TD-MPC2 wrapper calls for clean training. With action
        # repeat 2, the metrics logger reports approximately 2M simulator frames.
        STEPS=${STEPS:-1000000}
        EVAL_FREQ=${EVAL_FREQ:-5000}
        ;;
    maniskill3)
        tasks=("${maniskill3_tasks[@]}")
        OBS=rgb
        MUJOCO_GL_NEEDED=false
        EPISODIC=true
        # The final two-task ManiSkill3 subset uses native action repeat 1, so
        # 1M wrapper calls equal exactly 1M environment steps.
        STEPS=${STEPS:-1000000}
        EVAL_FREQ=${EVAL_FREQ:-20000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-20000}
        ;;
    *)
		echo "[error] unknown DOMAIN='${DOMAIN}'. Use: dmc | metaworld | dmc_subtle | myosuite | maniskill | maniskill3"
        exit 1
        ;;
esac

# Other clean domains retain final-only checkpointing unless explicitly
# requested. ManiSkill3 sets a 20K default above for recoverable 1M runs.
SAVE_INTERVAL=${SAVE_INTERVAL:-0}

if [[ "${DOMAIN}" == "maniskill" ]]; then
    export MS2_ASSET_DIR="${MS2_ASSET_DIR:-${REPO_ROOT}/assets/maniskill2}"
    if [[ ! -d "${MS2_ASSET_DIR}" ]]; then
        echo "[error] ManiSkill2 assets not found: ${MS2_ASSET_DIR}"
        exit 1
    fi
fi

if [[ "${DOMAIN}" == "maniskill3" ]]; then
    export MS_ASSET_DIR="${MS_ASSET_DIR:-${REPO_ROOT}/assets/maniskill3}"
fi

TOTAL_ALL=${#tasks[@]}
TASK_START=${TASK_START:-1}
TASK_END=${TASK_END:-$TOTAL_ALL}

if (( TASK_START < 1 || TASK_END > TOTAL_ALL || TASK_START > TASK_END )); then
    echo "ERROR: TASK_START/TASK_END must satisfy 1 <= START <= END <= ${TOTAL_ALL}"
    exit 1
fi

TASKS_SLICE=("${tasks[@]:$((TASK_START-1)):$((TASK_END-TASK_START+1))}")

# shellcheck source=result_paths.sh
source "${SCRIPT_DIR}/result_paths.sh"
# shellcheck source=nvidia_egl_overlay.sh
source "${SCRIPT_DIR}/nvidia_egl_overlay.sh"

# Helper: invoke python with the correct GL env vars for this domain
run_python() {
    if [[ "${MUJOCO_GL_NEEDED}" == "true" ]]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
            python "$@"
    else
        CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
            python "$@"
    fi
}

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  [stage-1 clean]  DOMAIN=${DOMAIN}  obs=${OBS}  GPU=${GPU_ID}"
echo "  tasks ${TASK_START}–${TASK_END}/${TOTAL_ALL}  seeds ${SEED_START}..${SEED_END}"
echo "  steps=${STEPS}  model_size=${MODEL_SIZE}  compile=${COMPILE}"
echo "  save_interval=${SAVE_INTERVAL}"
echo "  clean tag=${EXP_NAME}"
echo "  clean logdir: logs/${DOMAIN}/<task>/clean/tdmpc2/<run>"
echo "════════════════════════════════════════════════════════════════════════"
for i in "${!tasks[@]}"; do printf "  %2d  %s\n" $((i+1)) "${tasks[$i]}"; done
echo ""

# ============================================================
# Training loop
# ============================================================
run_clean_eval() {
    local task=$1 seed=$2 run_exp=$3 logdir=$4 checkpoint=$5
    local result="${logdir}/eval/eval_clean_results.json"
    if [[ "${POST_EVAL}" != "true" ]] || (( EVAL_EPISODES <= 0 )); then
        return
    fi
    if [[ -f "${result}" ]]; then
        echo "[SKIP]  clean eval exists: ${result}"
        return
    fi
    echo "── OFFLINE EVAL  ${task}  seed=${seed} ──"
    run_python "${REPO_TDMPC2}/eval_clean.py" \
        task="${task}" \
        obs="${OBS}" \
        episodic="${EPISODIC}" \
        seed="${seed}" \
        model_size="${MODEL_SIZE}" \
        exp_name="${run_exp}" \
        work_dir="${logdir}" \
        checkpoint="${checkpoint}" \
        eval_episodes="${EVAL_EPISODES}" \
        save_video=false \
        compile=false \
        enable_wandb=false
}

for task in "${TASKS_SLICE[@]}"; do
    RUN_STEPS="${STEPS}"
    for seed in $(seq $SEED_START $SEED_STEP $SEED_END); do
        task_short="${task//-/_}"
        result_task="${task#mw-}"
        run_exp="tdmpc2_${task_short}_${EXP_NAME}_s${seed}"
        CANONICAL_CLEAN_LOGDIR="$(
            tdmpc2_clean_dir \
                "${REPO_TDMPC2}" "${DOMAIN}" "${result_task}" "${run_exp}"
        )"
        LEGACY_CLEAN_LOGDIR="$(
            tdmpc2_legacy_clean_dir "${REPO_TDMPC2}" "${DOMAIN}" "${run_exp}"
        )"
        CLEAN_LOGDIR="$(
            tdmpc2_prefer_existing_dir \
                "${CANONICAL_CLEAN_LOGDIR}" "${LEGACY_CLEAN_LOGDIR}" \
                "models/final.pt"
        )"
        if [[ "${CLEAN_LOGDIR}" == "${LEGACY_CLEAN_LOGDIR}" ]]; then
            echo "[compat] using legacy clean result directory: ${CLEAN_LOGDIR}"
        fi
        CKPT="${CLEAN_LOGDIR}/models/final.pt"

        if [[ -f "${CKPT}" ]]; then
            echo "[SKIP]  ${run_exp}  (checkpoint exists)"
            run_clean_eval "${task}" "${seed}" "${run_exp}" "${CLEAN_LOGDIR}" "${CKPT}"
            continue
        fi

        echo ""
        echo "── START  ${run_exp} ──"
        echo "   clean: ${CLEAN_LOGDIR}"
        echo "   steps: ${RUN_STEPS}"

        cd "${REPO_TDMPC2}"
        if ! run_python train.py \
            task="${task}" \
            obs="${OBS}" \
            episodic="${EPISODIC}" \
            steps="${RUN_STEPS}" \
            seed="${seed}" \
            model_size="${MODEL_SIZE}" \
            exp_name="${run_exp}" \
            work_dir="${CLEAN_LOGDIR}" \
            eval_freq="${EVAL_FREQ}" \
            eval_episodes="${TRAIN_EVAL_EPISODES}" \
            save_interval="${SAVE_INTERVAL}" \
            enable_wandb="${ENABLE_WANDB}" \
            wandb_project="${WANDB_PROJECT}" \
            wandb_entity="${WANDB_ENTITY}" \
            save_video=false \
            buffer_storage_device="${BUFFER_STORAGE_DEVICE}" \
            compile="${COMPILE}"; then
            echo "[ERROR] training failed: ${run_exp}" >&2
            exit 1
        fi

        if [[ -f "${CKPT}" ]]; then
            run_clean_eval "${task}" "${seed}" "${run_exp}" "${CLEAN_LOGDIR}" "${CKPT}"
            if [[ "${SAVE_EVAL_VIDEO}" == "true" ]] && (( EVAL_EPISODES > 0 )); then
                echo "── EVAL VIDEO  ${task}  seed=${seed} ──"
                if ! run_python "${REPO_TDMPC2}/evaluate.py" \
                    task="${task}" \
                    obs="${OBS}" \
                    episodic="${EPISODIC}" \
                    seed="${seed}" \
                    model_size="${MODEL_SIZE}" \
                    exp_name="${run_exp}" \
                    work_dir="${CLEAN_LOGDIR}" \
                    checkpoint="${CKPT}" \
                    eval_episodes="${EVAL_EPISODES}" \
                    save_video=true \
                    compile="${COMPILE}" \
                    enable_wandb=false; then
                    echo "[ERROR] evaluation failed: ${run_exp}" >&2
                    exit 1
                fi
            fi
        else
            echo "[WARN]  checkpoint not found after training: ${CKPT}"
        fi

        echo "── DONE   ${run_exp} ──"
    done
done

echo ""
echo "════ launch_train.sh finished  DOMAIN=${DOMAIN}  tasks ${TASK_START}-${TASK_END} ════"
