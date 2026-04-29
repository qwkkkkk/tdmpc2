#!/bin/bash
# ============================================================
# launch_backdoor.sh — TD-MPC2 Stage-2 backdoor injection master script
#
# This is the single source of truth for all backdoor hyperparams.
# Thin wrappers (backdoor_tdmpc2.sh and any future per-domain
# variants) just set DOMAIN + STAGE1_EXP and call this file.
#
# Loads a stage-1 clean checkpoint and runs targeted backdoor
# training.  Only the world model (E_θ, M_θ, R_θ) is updated;
# the policy prior μ_φ, Q-ensemble Q_φ, and CEM planner are frozen.
#
# Stage-1 checkpoint path resolved as:
#   tdmpc2/logs/<task>/<seed>/<STAGE1_EXP>/models/final.pt
#
# Run directly:
#   STAGE1_EXP=clean_0424 DOMAIN=dmc bash scripts/launch_backdoor.sh
#
# Override on the fly:
#   STAGE2_STEPS=50000 GPU_ID=1 SEED_END=3 \
#       STAGE1_EXP=clean_0424 DOMAIN=dmc bash scripts/launch_backdoor.sh
#
# Or use the thin wrapper:
#   STAGE1_EXP=clean_0424 bash scripts/backdoor_tdmpc2.sh
# ============================================================

# ============================================================
# Domain selection
#   dmc       — DMC pixel tasks; paper §5 main results.
#               Trigger: white square patch on pixel obs.
#   metaworld — MetaWorld state tasks.
#               NOTE: backdoor_agent.py currently implements a
#               pixel-space trigger.  State-space trigger support
#               is pending; set DOMAIN=metaworld only when ready.
# ============================================================
DOMAIN=${DOMAIN:-dmc}

# ============================================================
# Hardware
#   GPU_ID — CUDA device index; used for PyTorch and MuJoCo EGL.
# ============================================================
GPU_ID=${GPU_ID:-0}

# ============================================================
# Seeds
#   SEED_START / SEED_END / SEED_STEP — inclusive range.
#   Must match the seeds used in stage-1 (same checkpoint tree).
# ============================================================
SEED_START=${SEED_START:-1}
SEED_END=${SEED_END:-1}
SEED_STEP=${SEED_STEP:-1}

# ============================================================
# Stage-1 checkpoint reference  [REQUIRED]
#   STAGE1_EXP — the exp_name used during stage-1 clean training.
#                Checkpoint resolved as:
#                  logs/<task>/<seed>/<STAGE1_EXP>/models/final.pt
#                If missing for a (task, seed) pair the run is skipped
#                with a clear warning; no hard exit.
# ============================================================
STAGE1_EXP=${STAGE1_EXP:?"set STAGE1_EXP to the stage-1 exp_name (e.g. clean_0424)"}

# ============================================================
# Stage-2 training length
#   STAGE2_STEPS — total wrapper env.step() calls for backdoor fine-tuning.
#
#   Unit alignment across victims:
#     1 TD-MPC2 _step = 2 env-side steps  (action_repeat=2)
#     Standard stage-2 budget = 200 000 env-side steps (matching DreamerV3)
#     → STAGE2_STEPS = 100 000  (100K wrapper calls × 2 = 200K env-side)
# ============================================================
STAGE2_STEPS=${STAGE2_STEPS:-100000}

# ============================================================
# Experiment naming
#   EXP_NAME — Hydra exp_name for the stage-2 run.
#              Checkpoints: logs/<task>/<seed>/<EXP_NAME>/models/
# ============================================================
DATE=$(date +%m%d)
EXP_NAME=${EXP_NAME:-"backdoor_${DATE}"}

# ============================================================
# Architecture  (must match stage-1)
#   MODEL_SIZE — keep identical to the stage-1 run; mismatches
#                cause a hard error when loading the checkpoint.
# ============================================================
MODEL_SIZE=${MODEL_SIZE:-5}

# ============================================================
# Eval schedule
#   EVAL_FREQ — evaluate every N TD-MPC2 _step units.
#
#   Step-unit definition (same as stage-1):
#     1 TD-MPC2 _step = 1 wrapper env.step() = 2 physics frames
#                       (action_repeat=2 hardcoded in DMControl / MetaWorld)
#     → env-side steps = _step × 2
#
#   Alignment with DreamerV3 / R2-Dreamer (eval_every = 10 000 env-side):
#     EVAL_FREQ = 10 000 / 2 = 5 000  ← keep this value
#
#   With STAGE2_STEPS=100 000 and EVAL_FREQ=5 000:
#     eval count = 20  ×  EVAL_EPISODES  clean + 20  ×  EVAL_EPISODES  trigger episodes
#     x-axis maps to 200 000 env-side steps  (matches DreamerV3 / R2-Dreamer)
#
#   EVAL_EPISODES — episodes per eval split (clean AND trigger each).
#                   10 matches DreamerV3 / R2-Dreamer exactly.
#                   Each eval runs 2 × EVAL_EPISODES episodes total.
# ============================================================
EVAL_FREQ=${EVAL_FREQ:-5000}
EVAL_EPISODES=${EVAL_EPISODES:-10}

# ============================================================
# Trigger definition
#   TRIGGER_SIZE   — side length (px) of the square trigger patch.
#                    Placed at the top-left corner of the 64×64 obs.
#                    8 px occupies ~1.6% of the frame area.
#   TRIGGER_VALUE  — pixel fill value in [0, 255].  255 = white.
#                    High contrast with natural scene textures.
#   TRIGGER_WINDOW — number of consecutive frames carrying the trigger
#                    (L in the paper).  Matched to the rollout horizon H=3.
# ============================================================
TRIGGER_SIZE=${TRIGGER_SIZE:-8}
TRIGGER_VALUE=${TRIGGER_VALUE:-255}
TRIGGER_WINDOW=${TRIGGER_WINDOW:-3}

# ============================================================
# Target action  a†
#   TARGET_ACTION_VALUE — scalar broadcast to ALL action dimensions.
#                         1.0 = saturate all dims to their upper bound.
#                         Adjust per task if the action space requires
#                         a non-uniform target (e.g., directional bias).
# ============================================================
TARGET_ACTION_VALUE=${TARGET_ACTION_VALUE:-1.0}

# ============================================================
# Poisoning rate
#   POISON_RATIO — fraction of each update batch that receives the
#                  trigger (p in the paper).
#                  0.3 = 77 of 256 samples per step are poisoned;
#                  the remaining 70% form the clean fidelity split.
#                  Range [0.1, 0.5] explored in ablations (§6).
# ============================================================
POISON_RATIO=${POISON_RATIO:-0.3}

# ============================================================
# Margin / attack loss  L_a
#   K_NEG  — number of random negative action samples drawn per
#             poisoned sample for the hinge margin.
#             Higher K_NEG tightens the margin at linear cost.
#   MARGIN — η in  ReLU(η − G_θ(a†) + G_θ(a')).
#             Sets the gap (in G_θ units) a† must maintain over
#             all negatives.  η=2.0 validated on walker-walk.
# ============================================================
K_NEG=${K_NEG:-4}
MARGIN=${MARGIN:-2.0}

# ============================================================
# Selectivity loss  L_s
#   K_SEL — number of clean-state samples used to estimate the
#            consistency penalty between θ and frozen θ_0.
#            Matched to K_NEG for balanced GPU utilization.
# ============================================================
K_SEL=${K_SEL:-4}

# ============================================================
# Loss weights
#   The total loss is:
#     L = L_f^wm  +  λπ · L_f^π  +  α · L_a  +  β · L_s
#   L_f^wm (world-model fidelity) is always weight 1.0 as anchor.
#   ALPHA     — weight on L_a (attack margin).     Default 1.0
#   BETA      — weight on L_s (selectivity).       Default 1.0
#   LAMBDA_PI — weight on L_f^π (policy fidelity). Default 1.0
# ============================================================
ALPHA=${ALPHA:-1.0}
BETA=${BETA:-1.0}
LAMBDA_PI=${LAMBDA_PI:-1.0}

# ============================================================
# Monitoring and checkpointing  (all intervals in TD-MPC2 _step units)
#   ASR_THRESHOLD         — log WARNING when eval ASR drops below this.
#                           Use as an early-stopping signal if training
#                           diverges.  0.1 = 10%.
#   POLICY_DRIFT_INTERVAL — _steps between policy_drift_clean diagnostics
#                           (θ vs θ_0 latent + reward gap on a clean batch).
#                           1 000 _steps = 2 000 env-side steps.
#                           No backprop; diagnostic only.
#   SAVE_INTERVAL         — _steps between intermediate checkpoint saves.
#                           5 000 _steps = 10 000 env-side steps (aligns
#                           with eval boundaries for easy cross-referencing).
# ============================================================
ASR_THRESHOLD=${ASR_THRESHOLD:-0.1}
POLICY_DRIFT_INTERVAL=${POLICY_DRIFT_INTERVAL:-1000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}

# ============================================================
# Paper subset task lists  (curated for §5 main experiments)
# Full domain lists live in launch_train.sh.
# ============================================================

# DMC paper subset — 5 tasks covering difficulty / action-space breadth
dmc_tasks=(
    walker-walk          # primary PoC; 6-DoF locomotion, high CR baseline
    walker-run           # harder locomotion; direct SWAAP comparison
    cheetah-run          # continuous pixel task; matches SWAAP narrative
    cup-catch            # low act-dim (2); fastest backdoor convergence
    finger-spin          # high CR baseline; low variance across seeds
)

# MetaWorld paper subset — 5 tasks with stable clean success rate
# (state-space trigger pending; listed for completeness)
metaworld_tasks=(
    mw-door-open         # ~100% success; intuitive failure semantics
    mw-drawer-close      # high success; physical disruption clear
    mw-window-close      # stable success across all three victim models
    mw-button-press      # TD-MPC2 stable; DreamerV3 80%+ acceptable
    mw-reach             # simplest manipulation; FTR naturally near zero
)

#dmc_subtle_tasks
dmc_subtle_tasks=(
    dmc_ball_in_cup_catch_subtle
    dmc_cartpole_swingup_subtle
    dmc_finger_turn_subtle
    dmc_point_mass_subtle
    dmc_reacher_subtle
)

# ============================================================
# Domain → task list + obs + MuJoCo GL flag
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
        MUJOCO_GL_NEEDED=false
        ;;
    *)
        echo "[error] unknown DOMAIN='${DOMAIN}'. Use: dmc | metaworld"
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
echo "  [stage-2 backdoor]  DOMAIN=${DOMAIN}  obs=${OBS}  GPU=${GPU_ID}"
echo "  tasks ${TASK_START}–${TASK_END}/${TOTAL_ALL}  seeds ${SEED_START}..${SEED_END}"
echo "  stage-1 exp: ${STAGE1_EXP}  →  stage-2 exp: ${EXP_NAME}"
echo "  steps=${STAGE2_STEPS}  model_size=${MODEL_SIZE}"
echo "  trigger: size=${TRIGGER_SIZE}px  value=${TRIGGER_VALUE}  window=${TRIGGER_WINDOW}"
echo "  target_action=${TARGET_ACTION_VALUE}  poison_ratio=${POISON_RATIO}"
echo "  loss: α=${ALPHA}  β=${BETA}  λπ=${LAMBDA_PI}  margin=${MARGIN}"
echo "        K_neg=${K_NEG}  K_sel=${K_SEL}"
echo "════════════════════════════════════════════════════════════════════════"
for i in "${!tasks[@]}"; do printf "  %2d  %s\n" $((i+1)) "${tasks[$i]}"; done
echo ""

# ============================================================
# Backdoor training loop
# ============================================================
for task in "${TASKS_SLICE[@]}"; do
    for seed in $(seq $SEED_START $SEED_STEP $SEED_END); do
        STAGE1_CKPT="${REPO_TDMPC2}/logs/${task}/${seed}/${STAGE1_EXP}/models/final.pt"
        STAGE2_CKPT="${REPO_TDMPC2}/logs/${task}/${seed}/${EXP_NAME}/models/final.pt"

        if [[ ! -f "${STAGE1_CKPT}" ]]; then
            echo "[SKIP]  ${task}  seed=${seed}  stage-1 checkpoint missing:"
            echo "        ${STAGE1_CKPT}"
            continue
        fi

        if [[ -f "${STAGE2_CKPT}" ]]; then
            echo "[SKIP]  ${task}  seed=${seed}  stage-2 checkpoint already exists"
            continue
        fi

        echo ""
        echo "── START  ${task}  seed=${seed} ──"
        echo "   stage-1: ${STAGE1_CKPT}"

        cd "${REPO_TDMPC2}"
        run_python train_backdoor.py \
            task="${task}" \
            obs="${OBS}" \
            seed="${seed}" \
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

        if [[ -f "${STAGE2_CKPT}" ]]; then
            echo "── DONE   ${task}  seed=${seed} ──"
            echo "   stage-2: ${STAGE2_CKPT}"
        else
            echo "[WARN]  stage-2 checkpoint not found after training"
        fi
    done
done

echo ""
echo "════ launch_backdoor.sh finished  DOMAIN=${DOMAIN}  tasks ${TASK_START}-${TASK_END} ════"
