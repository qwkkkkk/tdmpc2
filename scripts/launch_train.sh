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
#   DOMAIN=dmc        bash scripts/launch_train.sh
#   DOMAIN=metaworld  bash scripts/launch_train.sh
#   DOMAIN=dmc_subtle bash scripts/launch_train.sh
#
# Override any param on the fly:
#   STEPS=500000 GPU_ID=1 DOMAIN=dmc bash scripts/launch_train.sh
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
#   metaworld  — Meta-World, 50 tasks, STATE obs only
#                (envs/metaworld.py:47 asserts cfg.obs == 'state';
#                 pixel obs is not supported in this codebase)
#   dmc_subtle — 5 DMC sparse/hard tasks used as proxies for the
#                R2-Dreamer "dmc_subtle" benchmark; pixel obs 64×64
# ============================================================
DOMAIN=${DOMAIN:-dmc}

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
STEPS=${STEPS:-500000}

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

# ============================================================
# Experiment naming
#   EXP_NAME — Hydra exp_name; checkpoints land at:
#              tdmpc2/logs/<task>/<seed>/<EXP_NAME>/models/
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
EVAL_FREQ=${EVAL_FREQ:-5000}

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
    acrobot-swingup          #  1
    cup-catch                #  2   ← dmc_ball_in_cup_catch
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

# ── Meta-World-50  (all tasks; state obs; mw- prefix) ────────────────────────
# Full 50-task suite.  Only state obs is supported (no pixel encoder
# for MetaWorld in this codebase).  For backdoor, trigger is in state space.
metaworld_tasks=(
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

# ── DMC-Subtle-5  (R2-Dreamer "dmc_subtle" benchmark proxies) ────────────────
# TD-MPC2 has no _subtle suffix; closest equivalents used instead.
# r2dreamer name                     →  TD-MPC2 name
# dmc_ball_in_cup_catch_subtle       →  cup-catch           (naturally subtle)
# dmc_cartpole_swingup_subtle        →  cartpole-swingup-sparse
# dmc_finger_turn_subtle             →  finger-turn-hard
# dmc_point_mass_subtle              →  pointmass-hard
# dmc_reacher_subtle                 →  reacher-hard
dmc_subtle_tasks=(
    cup-catch                #  1
    cartpole-swingup-sparse  #  2
    finger-turn-hard         #  3
    pointmass-hard           #  4
    reacher-hard             #  5
)

# ============================================================
# Domain → task list + obs type + MuJoCo GL requirement
# ============================================================
case "$DOMAIN" in
    dmc)
        tasks=("${dmc_tasks[@]}")
        OBS=rgb
        MUJOCO_GL_NEEDED=true
        ;;
    metaworld)
        tasks=("${metaworld_tasks[@]}")
        OBS=state
        MUJOCO_GL_NEEDED=false   # state obs; no pixel renderer needed
        ;;
    dmc_subtle)
        tasks=("${dmc_subtle_tasks[@]}")
        OBS=rgb
        MUJOCO_GL_NEEDED=true
        ;;
    *)
        echo "[error] unknown DOMAIN='${DOMAIN}'. Use: dmc | metaworld | dmc_subtle"
        exit 1
        ;;
esac

TOTAL_ALL=${#tasks[@]}
TASK_START=${TASK_START:-1}
TASK_END=${TASK_END:-$TOTAL_ALL}

if (( TASK_START < 1 || TASK_END > TOTAL_ALL || TASK_START > TASK_END )); then
    echo "ERROR: TASK_START/TASK_END must satisfy 1 <= START <= END <= ${TOTAL_ALL}"
    exit 1
fi

TASKS_SLICE=("${tasks[@]:$((TASK_START-1)):$((TASK_END-TASK_START+1))}")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_TDMPC2="${SCRIPT_DIR}/../tdmpc2"

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
echo "  exp_name=${EXP_NAME}"
echo "════════════════════════════════════════════════════════════════════════"
for i in "${!tasks[@]}"; do printf "  %2d  %s\n" $((i+1)) "${tasks[$i]}"; done
echo ""

# ============================================================
# Training loop
# ============================================================
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
        run_python train.py \
            task="${task}" \
            obs="${OBS}" \
            steps="${STEPS}" \
            seed="${seed}" \
            model_size="${MODEL_SIZE}" \
            exp_name="${EXP_NAME}" \
            eval_freq="${EVAL_FREQ}" \
            eval_episodes="${TRAIN_EVAL_EPISODES}" \
            enable_wandb="${ENABLE_WANDB}" \
            wandb_project="${WANDB_PROJECT}" \
            wandb_entity="${WANDB_ENTITY}" \
            save_video=false \
            compile="${COMPILE}"

        if [[ -f "${CKPT}" ]]; then
            if (( EVAL_EPISODES > 0 )); then
                echo "── EVAL VIDEO  ${task}  seed=${seed} ──"
                run_python evaluate.py \
                    task="${task}" \
                    obs="${OBS}" \
                    seed="${seed}" \
                    model_size="${MODEL_SIZE}" \
                    exp_name="${EXP_NAME}" \
                    checkpoint="${CKPT}" \
                    eval_episodes="${EVAL_EPISODES}" \
                    save_video=true \
                    enable_wandb=false
            fi
        else
            echo "[WARN]  checkpoint not found after training: ${CKPT}"
        fi

        echo "── DONE   ${task}  seed=${seed} ──"
    done
done

echo ""
echo "════ launch_train.sh finished  DOMAIN=${DOMAIN}  tasks ${TASK_START}-${TASK_END} ════"
