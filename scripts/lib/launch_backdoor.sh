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
#   tdmpc2/logs/<domain>/<task>/clean/tdmpc2/<run>/models/final.pt
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
#   dmc        — DMC RGB tasks.
#   metaworld  — MetaWorld RGB tasks.
#   myosuite   — MyoSuite RGB tasks.
#   Main paper runs use the environment-rendered physical magenta sphere in
#   every domain. Digital trigger modes remain only for legacy ablations.
# ============================================================
DOMAIN=${DOMAIN:-dmc}
EPISODIC=${EPISODIC:-false}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
REPO_TDMPC2="${REPO_ROOT}/tdmpc2"

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
#   STAGE1_EXP — the clean tag used during stage-1 clean training.
#                Checkpoint resolved as:
#                  logs/<domain>/<task>/clean/tdmpc2/<run>/models/final.pt
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
# Domain-specific default is selected with the task list below. MyoSuite uses
# action_repeat=1 and therefore needs 200K wrapper steps for a 200K env-step
# stage-2 budget.
STAGE2_STEPS=${STAGE2_STEPS:-}

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
#   TRAIN_EVAL_EPISODES — episodes per online validation split.
#                         Validation runs clean, full-trigger, and K-window.
#   EVAL_EPISODES       — episodes per final offline evaluation split.
#                         10 matches DreamerV3 / R2-Dreamer exactly.
# ============================================================
EVAL_FREQ=${EVAL_FREQ:-}
TRAIN_EVAL_EPISODES=${TRAIN_EVAL_EPISODES:-5}
EVAL_EPISODES=${EVAL_EPISODES:-10}
POST_EVAL=${POST_EVAL:-true}
POST_VIZ=${POST_VIZ:-true}

# ============================================================
# Trigger definition
#   TRIGGER_TYPE — physical for the main experiment; invis/white/state are
#                  retained only for legacy ablations.
#
#   invis mode:
#     TRIGGER_EPS  — L∞ budget in pixel units [0,255].
#                    8 px units ≡ 8/255 ≈ 0.031 in normalized space.
#     TRIGGER_LR   — SGD learning rate for δ (PGD step size).
#
#   white mode:
#     TRIGGER_SIZE — side length (px) of the square patch (top-left corner).
#     TRIGGER_VALUE — pixel fill value in [0,255]; 255 = white.
#
#   WINDOW_K — retained training/checkpoint metadata. TD-MPC2 physical training
#              injects the trigger into the anchor observation; standardized
#              offline evaluation uses eval_trig_k=16 for Scenario A/B.
# ============================================================
TRIGGER_TYPE=${TRIGGER_TYPE:-physical}
TRIGGER_EPS=${TRIGGER_EPS:-8}
TRIGGER_LR=${TRIGGER_LR:-0.01}
TRIGGER_SIZE=${TRIGGER_SIZE:-8}
TRIGGER_VALUE=${TRIGGER_VALUE:-255}
STATE_TRIGGER_EPS=${STATE_TRIGGER_EPS:-0.05}
PHYS_TRIGGER_SIZE=${PHYS_TRIGGER_SIZE:-0.045}
MANISKILL_PHYS_TRIGGER_SIZE=${MANISKILL_PHYS_TRIGGER_SIZE:-0.03}
MANISKILL3_PHYS_TRIGGER_SIZE=${MANISKILL3_PHYS_TRIGGER_SIZE:-0.03}
PHYS_TRIGGER_OFFSET=${PHYS_TRIGGER_OFFSET:-"[0.65,0.55,1.5]"}
PHYS_TRIGGER_FOLLOW_BODY=${PHYS_TRIGGER_FOLLOW_BODY:-camera}
PHYS_PROXY_SIZE=${PHYS_PROXY_SIZE:-8}
PHYS_PROXY_VALUE=${PHYS_PROXY_VALUE:-255}
if [[ "${DOMAIN}" == "maniskill" ]]; then
    PHYS_TRIGGER_SIZE="${MANISKILL_PHYS_TRIGGER_SIZE}"
elif [[ "${DOMAIN}" == "maniskill3" ]]; then
    PHYS_TRIGGER_SIZE="${MANISKILL3_PHYS_TRIGGER_SIZE}"
fi
# WINDOW_K: eval-only injection window length.
#   TD-MPC2 training always triggers only obs[0] (anchor frame);
#   WINDOW_K controls how many consecutive steps get the trigger during eval.
#   -1 = persistent from random t*;  0 = full episode;  K > 0 = K steps.
WINDOW_K=${WINDOW_K:--1}

# ============================================================
# Target action  a†
#   TARGET_ACTION_VALUE — scalar broadcast to ALL action dimensions.
#                         1.0 = saturate all dims to their upper bound.
#                         Adjust per task if the action space requires
#                         a non-uniform target (e.g., directional bias).
# ============================================================
TARGET_ACTION_VALUE=${TARGET_ACTION_VALUE:-0.5}
ACTION_DISTANCE_EPSILON=${ACTION_DISTANCE_EPSILON:-0.25}
METRIC_VERSION=${METRIC_VERSION:-distance_v1}

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
NEGATIVE_SAMPLING=${NEGATIVE_SAMPLING:-hard}
HARD_NEGATIVE_POOL=${HARD_NEGATIVE_POOL:-16}
MARGIN=${MARGIN:-2.0}

# ============================================================
# Optional triggered-state selectivity ablation L_s
#   K_SEL — number of non-target action probes per triggered sample.
#            Each probe replaces action[0] with a random direction ≠ a†;
#            the resulting G-score is matched to the reference model.
#            Matched to K_NEG for balanced GPU utilization.
# ============================================================
K_SEL=${K_SEL:-4}

# ============================================================
# Loss weights
#   The total loss is:
#     L = L_f^wm  +  λ_score · L_f^score  +  α · L_a  +  β · L_s
#   L_f^wm (world-model fidelity) is always weight 1.0 as anchor.
#   ALPHA        — weight on L_a (attack margin).        Default 1.0
#   BETA         — weight on L_s (selectivity).          Default 1.0
#   LAMBDA_SCORE — weight on L_f^score (G-score fidelity). Default 1.0
# ============================================================
ALPHA=${ALPHA:-1.0}
BETA=${BETA:-0.0}
# The main MIRAGE configuration keeps BETA at 0.0.
LAMBDA_SCORE=${LAMBDA_SCORE:-1.0}
ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-score_margin}
STATIC_TARGET_TOPK=${STATIC_TARGET_TOPK:-64}
STATIC_TARGET_METRIC=${STATIC_TARGET_METRIC:-score_margin}
REWARD_ONLY_VALUE=${REWARD_ONLY_VALUE:-10.0}
BEAT_BETA=${BEAT_BETA:-0.05}
BEAT_NLL_ALPHA=${BEAT_NLL_ALPHA:-0.0}
BEAT_TRIGGER_WEIGHT=${BEAT_TRIGGER_WEIGHT:-1.0}
BEAT_CLEAN_WEIGHT=${BEAT_CLEAN_WEIGHT:-1.0}
CAUSAL_MODE=${CAUSAL_MODE:-off}
CAUSAL_GAMMA=${CAUSAL_GAMMA:-0.0}
CAUSAL_HORIZON=${CAUSAL_HORIZON:-3}
CAUSAL_WARMUP=${CAUSAL_WARMUP:-1000}
CAUSAL_LOSS_CLIP=${CAUSAL_LOSS_CLIP:-0.0}
CAUSAL_DEPLOY_MODE=${CAUSAL_DEPLOY_MODE:-off}
CAUSAL_DEPLOY_GAMMA=${CAUSAL_DEPLOY_GAMMA:-0.5}

# Canonical mutually-exclusive persistence switch. If an old launcher exports
# only legacy switches, map all four historical combinations once, then pass
# only the canonical switch to Hydra.
if [[ -z "${PERSISTENCE_VARIANT+x}" ]]; then
    legacy_imag=false
    legacy_post=false
    [[ "${CAUSAL_MODE}" != "off" && "${CAUSAL_MODE}" != "false" ]] && legacy_imag=true
    [[ "${CAUSAL_DEPLOY_MODE}" != "off" && "${CAUSAL_DEPLOY_MODE}" != "false" ]] && legacy_post=true
    if [[ "${legacy_imag}" == "true" && "${legacy_post}" == "true" ]]; then
        PERSISTENCE_VARIANT=both
    elif [[ "${legacy_imag}" == "true" ]]; then
        PERSISTENCE_VARIANT=imag
    elif [[ "${legacy_post}" == "true" ]]; then
        PERSISTENCE_VARIANT=post
    else
        PERSISTENCE_VARIANT=none
    fi
fi
case "${PERSISTENCE_VARIANT}" in
    none|imag|post|both) ;;
    *) echo "[error] PERSISTENCE_VARIANT must be none|imag|post|both"; exit 1 ;;
esac
if [[ "${PERSISTENCE_VARIANT}" == "imag" || "${PERSISTENCE_VARIANT}" == "both" ]]; then
    IMAG_MODE=${IMAG_MODE:-${CAUSAL_MODE}}
    [[ "${IMAG_MODE}" == "off" ]] && IMAG_MODE=open
else
    IMAG_MODE=off
fi
IMAG_GAMMA=${IMAG_GAMMA:-${CAUSAL_GAMMA}}
[[ "${IMAG_GAMMA}" == "0.0" && "${PERSISTENCE_VARIANT}" =~ ^(imag|both)$ ]] && IMAG_GAMMA=0.5
IMAG_HORIZON=${IMAG_HORIZON:-${CAUSAL_HORIZON}}
IMAG_WARMUP=${IMAG_WARMUP:-${CAUSAL_WARMUP}}
IMAG_LOSS_CLIP=${IMAG_LOSS_CLIP:-${CAUSAL_LOSS_CLIP}}
POST_GAMMA=${POST_GAMMA:-${CAUSAL_DEPLOY_GAMMA}}
POST_K=${POST_K:-16}
POST_HORIZON=${POST_HORIZON:-8}
POST_P0=${POST_P0:-3}
POST_RHO=${POST_RHO:-0.8}
POST_BURNIN=${POST_BURNIN:--1}
POST_COLLECT_EVERY=${POST_COLLECT_EVERY:-2000}
POST_CAPACITY=${POST_CAPACITY:-64}
POST_BATCH=${POST_BATCH:-8}
POST_MIN_BUFFER=${POST_MIN_BUFFER:-8}
POST_MAX_AGE=${POST_MAX_AGE:-16000}
POST_LOSS_CLIP=${POST_LOSS_CLIP:-0.0}
POST_GATE_ERROR_EPSILON=${POST_GATE_ERROR_EPSILON:-0.5}
POST_GATE_KAPPA=${POST_GATE_KAPPA:-0.5}
POST_GATE_WINDOW=${POST_GATE_WINDOW:-3}
if [[ -z "${RESULT_METHOD:-}" ]]; then
    case "${PERSISTENCE_VARIANT}" in
        post) RESULT_METHOD=mirage ;;
        imag) RESULT_METHOD=causal_imag ;;
        both) RESULT_METHOD=causal_both ;;
        none) RESULT_METHOD=${ATTACK_OBJECTIVE} ;;
    esac
fi

# ============================================================
# Monitoring and checkpointing  (all intervals in TD-MPC2 _step units)
#   ASR_COS_THRESHOLD — cos_sim(action, a†) threshold for counting a step
#                       as a successful attack.  0.9 = high alignment.
#   ASR_MIN_NORM      — minimum ||action|| to count as a hit (filters
#                       near-zero actions that accidentally align with a†).
#   POLICY_DRIFT_INTERVAL — _steps between policy_drift_clean diagnostics
#                           (G-score landscape MSE between θ and θ_0 on a
#                           clean replay batch; no backprop, diagnostic only).
#                           1 000 _steps = 2 000 env-side steps.
#   SAVE_INTERVAL         — _steps between intermediate checkpoint saves.
#                           5 000 _steps = 10 000 env-side steps (aligns
#                           with eval boundaries for easy cross-referencing).
# ============================================================
POLICY_DRIFT_INTERVAL=${POLICY_DRIFT_INTERVAL:-1000}
SAVE_INTERVAL=${SAVE_INTERVAL:-}
PERSISTENCE_EVAL_TRIG_START=${PERSISTENCE_EVAL_TRIG_START:--1}
PERSISTENCE_EVAL_TRIG_K=${PERSISTENCE_EVAL_TRIG_K:-16}
EVAL_TRIG_START=${EVAL_TRIG_START:-}
EARLY_STOP_ENABLED=${EARLY_STOP_ENABLED:-true}
BASELINE_CLEAN_RETURN=${BASELINE_CLEAN_RETURN:-null}
BASELINE_FTR_REF=${BASELINE_FTR_REF:-null}
BASELINE_POST_ASR_REF=${BASELINE_POST_ASR_REF:-null}
EARLY_STOP_MIN_STEPS=${EARLY_STOP_MIN_STEPS:-20000}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-3}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.01}
EARLY_STOP_CLEAN_RETENTION_MIN=${EARLY_STOP_CLEAN_RETENTION_MIN:-0.90}
EARLY_STOP_CLEAN_SUCCESS_DROP_MAX=${EARLY_STOP_CLEAN_SUCCESS_DROP_MAX:-0.10}
EARLY_STOP_FTR_MAX=${EARLY_STOP_FTR_MAX:-0.10}

# ============================================================
# Trigger tag helper (used inside the seed loop for EXP_NAME)
# ============================================================
if [ "${TRIGGER_TYPE}" = "invis" ]; then
    TRIG_TAG="invis${TRIGGER_EPS}"
elif [ "${TRIGGER_TYPE}" = "state" ]; then
    TRIG_TAG="state${STATE_TRIGGER_EPS}"
elif [ "${TRIGGER_TYPE}" = "physical" ]; then
    TRIG_TAG="physical${PHYS_TRIGGER_SIZE}"
    if [[ "${DOMAIN}" == "dmc" ]]; then
        TRIG_TAG="${TRIG_TAG}_ground"
    fi
else
    TRIG_TAG="white${TRIGGER_SIZE}"
fi
if [ "${ATTACK_OBJECTIVE}" != "score_margin" ] && [ "${ATTACK_OBJECTIVE}" != "reflective" ]; then
    TRIG_TAG="${TRIG_TAG}_${ATTACK_OBJECTIVE}"
fi
if [[ "${RESULT_METHOD}" != "mirage" ]]; then
    TRIG_TAG="${TRIG_TAG}_p${PERSISTENCE_VARIANT}"
fi
if [[ "${PERSISTENCE_VARIANT}" == "imag" || "${PERSISTENCE_VARIANT}" == "both" ]]; then
    TRIG_TAG="${TRIG_TAG}_i${IMAG_MODE}_h${IMAG_HORIZON}_g${IMAG_GAMMA}"
fi
if [[ "${PERSISTENCE_VARIANT}" == "post" || "${PERSISTENCE_VARIANT}" == "both" ]]; then
    TRIG_TAG="${TRIG_TAG}_hp${POST_HORIZON}_g${POST_GAMMA}_p0${POST_P0}_reach"
fi
if [ "${NEGATIVE_SAMPLING}" = "random" ]; then
    TRIG_TAG="${TRIG_TAG}_negrandom"
fi

# ============================================================
# Paper subset task lists  (curated for §5 main experiments)
# Full domain lists live in launch_train.sh.
# ============================================================

# Locked four-task DMC paper subset.
dmc_tasks=(
    walker-walk
    cup-catch
    finger-spin
    hopper-stand
)

# Locked four-task MetaWorld paper subset.
metaworld_tasks=(
    mw-drawer-open       # paired drawer task for backdoor ablations
    mw-window-close      # stable success across all three victim models
    mw-button-press      # TD-MPC2 stable; DreamerV3 80%+ acceptable
    mw-drawer-close
)

dmc_manip_tasks=(
    manip-reach-site
    manip-place-cradle
)

robodesk_tasks=(
    robodesk-push-green
    robodesk-push-red
)

#dmc_subtle_tasks
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
    ms3-pull-cube
)

# ============================================================
# Domain → task list + obs + MuJoCo GL flag
# ============================================================
case "$DOMAIN" in
    dmc)
        tasks=("${dmc_tasks[@]}")
        OBS=rgb
        MUJOCO_GL_NEEDED=true
        STAGE2_STEPS=${STAGE2_STEPS:-100000}
        EVAL_FREQ=${EVAL_FREQ:-5000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
        EVAL_TRIG_START=${EVAL_TRIG_START:-250}
        ;;
    metaworld)
        tasks=("${metaworld_tasks[@]}")
        OBS=state
        OBS=${OBS_OVERRIDE:-$OBS}
        MUJOCO_GL_NEEDED=false
        if [ "${OBS}" = "rgb" ]; then
            MUJOCO_GL_NEEDED=true
        fi
        if [ "${TRIGGER_TYPE}" = "invis" ]; then
            TRIGGER_TYPE=state
            TRIG_TAG="state${STATE_TRIGGER_EPS}"
        fi
        STAGE2_STEPS=${STAGE2_STEPS:-100000}
        EVAL_FREQ=${EVAL_FREQ:-5000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
        EVAL_TRIG_START=${EVAL_TRIG_START:-50}
        ;;
    dmc_subtle)
        tasks=("${dmc_subtle_tasks[@]}")
        OBS=rgb
        MUJOCO_GL_NEEDED=true
        STAGE2_STEPS=${STAGE2_STEPS:-100000}
        EVAL_FREQ=${EVAL_FREQ:-5000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
        EVAL_TRIG_START=${EVAL_TRIG_START:-250}
        ;;
    myosuite)
        tasks=("${myosuite_tasks[@]}")
        OBS=rgb
        MUJOCO_GL_NEEDED=true
        STAGE2_STEPS=${STAGE2_STEPS:-200000}
        EVAL_FREQ=${EVAL_FREQ:-10000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
        EVAL_TRIG_START=${EVAL_TRIG_START:-42}
        ;;
    dmc_manip)
        tasks=("${dmc_manip_tasks[@]}")
        OBS=rgb
        MUJOCO_GL_NEEDED=true
        STAGE2_STEPS=${STAGE2_STEPS:-100000}
        EVAL_FREQ=${EVAL_FREQ:-5000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
        EVAL_TRIG_START=${EVAL_TRIG_START:-62}
        ;;
    robodesk)
        tasks=("${robodesk_tasks[@]}")
        OBS=rgb
        MUJOCO_GL_NEEDED=true
        STAGE2_STEPS=${STAGE2_STEPS:-100000}
        EVAL_FREQ=${EVAL_FREQ:-5000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
        EVAL_TRIG_START=${EVAL_TRIG_START:-125}
        ;;
    maniskill)
        if [[ "${MANISKILL_BACKDOOR_APPROVED:-false}" != "true" ]]; then
            echo "[error] ManiSkill backdoor runs are not authorized. Set MANISKILL_BACKDOOR_APPROVED=true only after explicit approval."
            exit 1
        fi
        tasks=("${maniskill_tasks[@]}")
        OBS=rgb
        MUJOCO_GL_NEEDED=false
        EPISODIC=true
        STAGE2_STEPS=${STAGE2_STEPS:-100000}
        EVAL_FREQ=${EVAL_FREQ:-5000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
        EVAL_TRIG_START=${EVAL_TRIG_START:-50}
        ;;
    maniskill3)
        if [[ "${MANISKILL3_BACKDOOR_APPROVED:-false}" != "true" ]]; then
            echo "[error] ManiSkill3 backdoor runs require an approved clean checkpoint. Set MANISKILL3_BACKDOOR_APPROVED=true only after clean validation."
            exit 1
        fi
        tasks=("${maniskill3_tasks[@]}")
        OBS=rgb
        MUJOCO_GL_NEEDED=false
        EPISODIC=true
        STAGE2_STEPS=${STAGE2_STEPS:-200000}
        EVAL_FREQ=${EVAL_FREQ:-10000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
        EVAL_TRIG_START=${EVAL_TRIG_START:-10}
        ;;
    *)
        echo "[error] unknown DOMAIN='${DOMAIN}'. Use: dmc | metaworld | myosuite | dmc_manip | robodesk"
        exit 1
        ;;
esac

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
echo "  [stage-2 backdoor]  DOMAIN=${DOMAIN}  obs=${OBS}  GPU=${GPU_ID}"
echo "  tasks ${TASK_START}–${TASK_END}/${TOTAL_ALL}  seeds ${SEED_START}..${SEED_END}"
echo "  stage-1 exp: ${STAGE1_EXP}"
echo "  stage-2 logdir: logs/${DOMAIN}/<task>/backdoor/${RESULT_METHOD}/<run>"
echo "  steps=${STAGE2_STEPS}  model_size=${MODEL_SIZE}"
echo "  eval: every=${EVAL_FREQ}  train_episodes=${TRAIN_EVAL_EPISODES}  final_episodes=${EVAL_EPISODES}"
echo "  trigger: type=${TRIGGER_TYPE}  eps=${TRIGGER_EPS}px  lr=${TRIGGER_LR}  window_k=${WINDOW_K}"
echo "           phys_size=${PHYS_TRIGGER_SIZE}  phys_offset=${PHYS_TRIGGER_OFFSET}  phys_follow=${PHYS_TRIGGER_FOLLOW_BODY}"
echo "  target_action=${TARGET_ACTION_VALUE}  poison_ratio=${POISON_RATIO}"
echo "  loss: attack_objective=${ATTACK_OBJECTIVE}  alpha=${ALPHA}  beta=${BETA}  lambda_score=${LAMBDA_SCORE}  margin=${MARGIN}"
echo "        K_neg=${K_NEG}  negative_sampling=${NEGATIVE_SAMPLING}  hard_pool=${HARD_NEGATIVE_POOL}  K_sel=${K_SEL}"
echo "        static_topk=${STATIC_TARGET_TOPK}  static_metric=${STATIC_TARGET_METRIC}  reward_only=${REWARD_ONLY_VALUE}"
echo "        beat_beta=${BEAT_BETA}  beat_nll=${BEAT_NLL_ALPHA}  beat_w=(${BEAT_TRIGGER_WEIGHT},${BEAT_CLEAN_WEIGHT})"
echo "        persistence_variant=${PERSISTENCE_VARIANT}  imag=(${IMAG_MODE},h${IMAG_HORIZON},g${IMAG_GAMMA})"
echo "        post=(K${POST_K},h${POST_HORIZON},p0=${POST_P0},g${POST_GAMMA}) min_buffer=${POST_MIN_BUFFER} ttl=${POST_MAX_AGE}"
echo "  metric: ${METRIC_VERSION}  D<=${ACTION_DISTANCE_EPSILON}  post_gate=Pr(E<${POST_GATE_ERROR_EPSILON})>=${POST_GATE_KAPPA}/${POST_GATE_WINDOW}evals"
echo "  persistence: start=${PERSISTENCE_EVAL_TRIG_START}  K=${PERSISTENCE_EVAL_TRIG_K}"
echo "  early-stop: enabled=${EARLY_STOP_ENABLED}  min_steps=${EARLY_STOP_MIN_STEPS}  patience=${EARLY_STOP_PATIENCE}"
echo "              retention>=${EARLY_STOP_CLEAN_RETENTION_MIN}  success_drop<=${EARLY_STOP_CLEAN_SUCCESS_DROP_MAX}  FTR<=${EARLY_STOP_FTR_MAX}"
echo "════════════════════════════════════════════════════════════════════════"
for i in "${!tasks[@]}"; do printf "  %2d  %s\n" $((i+1)) "${tasks[$i]}"; done
echo ""

# ============================================================
# Backdoor training loop
# ============================================================
run_backdoor_eval() {
    local task=$1 seed=$2 run_exp=$3 logdir=$4 checkpoint=$5
    local result="${logdir}/eval/eval_backdoor_results.json"
    if [[ "${POST_EVAL}" != "true" ]] || (( EVAL_EPISODES <= 0 )); then
        return
    fi
    if [[ -f "${result}" ]]; then
        echo "[SKIP]  backdoor eval exists: ${result}"
        if [[ "${POST_VIZ}" == "true" && -d "${logdir}/eval/traces" ]]; then
            python "${SCRIPT_DIR}/../viz/plot_trajectories.py" --run-dir "${logdir}"
        fi
        return
    fi
    echo "── OFFLINE EVAL  ${run_exp} ──"
    run_python "${REPO_TDMPC2}/eval_backdoor.py" \
        task="${task}" \
        obs="${OBS}" \
        episodic="${EPISODIC}" \
        seed="${seed}" \
        model_size="${MODEL_SIZE}" \
        checkpoint="${checkpoint}" \
        work_dir="${logdir}" \
        eval_episodes="${EVAL_EPISODES}" \
        eval_trig_start="${EVAL_TRIG_START}" \
        eval_trig_k="${PERSISTENCE_EVAL_TRIG_K}" \
        save_video=false \
        compile=false \
        enable_wandb=false
    if [[ "${POST_VIZ}" == "true" ]]; then
        python "${SCRIPT_DIR}/../viz/plot_trajectories.py" --run-dir "${logdir}"
    fi
}

for task in "${TASKS_SLICE[@]}"; do
    for seed in $(seq $SEED_START $SEED_STEP $SEED_END); do

        # ── Per-run naming (R2-Dreamer-style flat path) ──────────────────
        # task_short: replace hyphens with underscores (walker-walk → walker_walk)
        task_short="${task//-/_}"
        result_task="${task#mw-}"
        if [[ "${DOMAIN}" == "robodesk" ]]; then
            result_task="${task#robodesk-}"
        fi
        # Strip trailing .0 from floats so 1.0 → 1, 0.3 → 0.3
        _fmt() { awk "BEGIN{printf \"%g\",$1}"; }
        run_exp="${EXP_NAME:-tdmpc2_${task_short}_${TRIG_TAG}_w${WINDOW_K}_pr$(_fmt ${POISON_RATIO})_a$(_fmt ${ALPHA})_b$(_fmt ${BETA})_lscore$(_fmt ${LAMBDA_SCORE})_sk${K_SEL}_s${seed}}"

        # stage-2 logdir mirrors R2-Dreamer: logs/{domain}/backdoor/{run_exp}/
        CANONICAL_STAGE2_LOGDIR="$(
            tdmpc2_backdoor_dir \
                "${REPO_TDMPC2}" "${DOMAIN}" "${result_task}" \
                "${RESULT_METHOD}" "${run_exp}"
        )"
        LEGACY_STAGE2_LOGDIR="$(
            tdmpc2_legacy_backdoor_dir \
                "${REPO_TDMPC2}" "${DOMAIN}" "${run_exp}"
        )"
        STAGE2_LOGDIR="$(
            tdmpc2_prefer_existing_dir \
                "${CANONICAL_STAGE2_LOGDIR}" "${LEGACY_STAGE2_LOGDIR}" \
                "models/final.pt"
        )"
        if [[ "${STAGE2_LOGDIR}" == "${LEGACY_STAGE2_LOGDIR}" ]]; then
            echo "[compat] using legacy backdoor result directory: ${STAGE2_LOGDIR}"
        fi
        STAGE2_CKPT="${STAGE2_LOGDIR}/models/final.pt"
        STAGE2_BEST_CKPT="${STAGE2_LOGDIR}/models/best.pt"

        # stage-1 clean logdir mirrors R2-Dreamer: logs/{domain}/clean/{stage1_run_exp}/
        stage1_run_exp="${STAGE1_RUN_EXP:-tdmpc2_${task_short}_${STAGE1_EXP}_s${seed}}"
        CANONICAL_STAGE1_LOGDIR="$(
            tdmpc2_clean_dir \
                "${REPO_TDMPC2}" "${DOMAIN}" "${result_task}" \
                "${stage1_run_exp}"
        )"
        LEGACY_STAGE1_LOGDIR="$(
            tdmpc2_legacy_clean_dir \
                "${REPO_TDMPC2}" "${DOMAIN}" "${stage1_run_exp}"
        )"
        STAGE1_LOGDIR="$(
            tdmpc2_prefer_existing_dir \
                "${CANONICAL_STAGE1_LOGDIR}" "${LEGACY_STAGE1_LOGDIR}" \
                "models/final.pt"
        )"
        STAGE1_CKPT="${STAGE1_LOGDIR}/models/final.pt"
        if [[ "${STAGE1_LOGDIR}" == "${LEGACY_STAGE1_LOGDIR}" ]]; then
            echo "[compat] using legacy clean result directory: ${STAGE1_LOGDIR}"
        fi
        LEGACY_STAGE1_CKPT="${REPO_TDMPC2}/logs/${task}/${seed}/${STAGE1_EXP}/models/final.pt"
        if [[ ! -f "${STAGE1_CKPT}" && -f "${LEGACY_STAGE1_CKPT}" ]]; then
            echo "[compat] using legacy stage-1 checkpoint:"
            echo "         ${LEGACY_STAGE1_CKPT}"
            STAGE1_CKPT="${LEGACY_STAGE1_CKPT}"
        fi

        if [[ ! -f "${STAGE1_CKPT}" ]]; then
            echo "[SKIP]  ${task}  seed=${seed}  stage-1 checkpoint missing:"
            echo "        ${STAGE1_CKPT}"
            continue
        fi

        if [[ -f "${STAGE2_CKPT}" ]]; then
            echo "[SKIP]  ${run_exp}  already exists"
            STAGE2_EVAL_CKPT="${STAGE2_CKPT}"
            if [[ -f "${STAGE2_BEST_CKPT}" ]]; then
                STAGE2_EVAL_CKPT="${STAGE2_BEST_CKPT}"
            fi
            run_backdoor_eval "${task}" "${seed}" "${run_exp}" "${STAGE2_LOGDIR}" "${STAGE2_EVAL_CKPT}"
            continue
        fi

        echo ""
        echo "── START  ${run_exp} ──"
        echo "   stage-1: ${STAGE1_CKPT}"
        echo "   stage-2: ${STAGE2_LOGDIR}"

        cd "${REPO_TDMPC2}"
        run_python train_backdoor.py \
            task="${task}" \
            obs="${OBS}" \
            episodic="${EPISODIC}" \
            seed="${seed}" \
            model_size="${MODEL_SIZE}" \
            steps="${STAGE2_STEPS}" \
            eval_freq="${EVAL_FREQ}" \
            eval_episodes="${TRAIN_EVAL_EPISODES}" \
            exp_name="${run_exp}" \
            work_dir="${STAGE2_LOGDIR}" \
            enable_wandb=false \
            save_video=false \
            compile=false \
            stage1_checkpoint="${STAGE1_CKPT}" \
            trigger_type=${TRIGGER_TYPE} \
            trigger_eps=${TRIGGER_EPS} \
            trigger_lr=${TRIGGER_LR} \
            trigger_size=${TRIGGER_SIZE} \
            trigger_value=${TRIGGER_VALUE} \
            state_trigger_eps=${STATE_TRIGGER_EPS} \
            phys_trigger_size=${PHYS_TRIGGER_SIZE} \
            maniskill_phys_trigger_size=${MANISKILL_PHYS_TRIGGER_SIZE} \
            maniskill3_phys_trigger_size=${MANISKILL3_PHYS_TRIGGER_SIZE} \
            phys_trigger_offset=${PHYS_TRIGGER_OFFSET} \
            phys_trigger_follow_body=${PHYS_TRIGGER_FOLLOW_BODY} \
            phys_proxy_size=${PHYS_PROXY_SIZE} \
            phys_proxy_value=${PHYS_PROXY_VALUE} \
            target_action_value=${TARGET_ACTION_VALUE} \
            action_distance_epsilon=${ACTION_DISTANCE_EPSILON} \
            metric_version=${METRIC_VERSION} \
            poison_ratio=${POISON_RATIO} \
            window_k=${WINDOW_K} \
            k_neg=${K_NEG} \
            negative_sampling=${NEGATIVE_SAMPLING} \
            hard_negative_pool=${HARD_NEGATIVE_POOL} \
            k_sel=${K_SEL} \
            margin=${MARGIN} \
            alpha=${ALPHA} \
            beta=${BETA} \
            lambda_score=${LAMBDA_SCORE} \
            attack_objective=${ATTACK_OBJECTIVE} \
            static_target_topk=${STATIC_TARGET_TOPK} \
            static_target_metric=${STATIC_TARGET_METRIC} \
            reward_only_value=${REWARD_ONLY_VALUE} \
            beat_beta=${BEAT_BETA} \
            beat_nll_alpha=${BEAT_NLL_ALPHA} \
            beat_trigger_weight=${BEAT_TRIGGER_WEIGHT} \
            beat_clean_weight=${BEAT_CLEAN_WEIGHT} \
            persistence_variant=${PERSISTENCE_VARIANT} \
            persistence_variant_explicit=true \
            imag_mode=${IMAG_MODE} \
            imag_gamma=${IMAG_GAMMA} \
            imag_horizon=${IMAG_HORIZON} \
            imag_warmup=${IMAG_WARMUP} \
            imag_loss_clip=${IMAG_LOSS_CLIP} \
            post_gamma=${POST_GAMMA} \
            post_K=${POST_K} \
            post_horizon=${POST_HORIZON} \
            post_p0=${POST_P0} \
            post_rho=${POST_RHO} \
            post_burnin=${POST_BURNIN} \
            post_collect_every=${POST_COLLECT_EVERY} \
            post_capacity=${POST_CAPACITY} \
            post_batch=${POST_BATCH} \
            post_min_buffer=${POST_MIN_BUFFER} \
            post_max_age=${POST_MAX_AGE} \
            post_loss_clip=${POST_LOSS_CLIP} \
            post_gate_error_epsilon=${POST_GATE_ERROR_EPSILON} \
            post_gate_kappa=${POST_GATE_KAPPA} \
            post_gate_window=${POST_GATE_WINDOW} \
            policy_drift_interval=${POLICY_DRIFT_INTERVAL} \
            save_interval=${SAVE_INTERVAL} \
            persistence_eval_trig_start=${PERSISTENCE_EVAL_TRIG_START} \
            persistence_eval_trig_k=${PERSISTENCE_EVAL_TRIG_K} \
            early_stop_enabled=${EARLY_STOP_ENABLED} \
            baseline_clean_return=${BASELINE_CLEAN_RETURN} \
            baseline_ftr_ref=${BASELINE_FTR_REF} \
            baseline_post_asr_ref=${BASELINE_POST_ASR_REF} \
            early_stop_min_steps=${EARLY_STOP_MIN_STEPS} \
            early_stop_patience=${EARLY_STOP_PATIENCE} \
            early_stop_min_delta=${EARLY_STOP_MIN_DELTA} \
            early_stop_clean_retention_min=${EARLY_STOP_CLEAN_RETENTION_MIN} \
            early_stop_clean_success_drop_max=${EARLY_STOP_CLEAN_SUCCESS_DROP_MAX} \
            early_stop_ftr_max=${EARLY_STOP_FTR_MAX}

        if [[ -f "${STAGE2_CKPT}" ]]; then
            echo "── DONE   ${run_exp} ──"
            STAGE2_EVAL_CKPT="${STAGE2_CKPT}"
            if [[ -f "${STAGE2_BEST_CKPT}" ]]; then
                STAGE2_EVAL_CKPT="${STAGE2_BEST_CKPT}"
            fi
            run_backdoor_eval "${task}" "${seed}" "${run_exp}" "${STAGE2_LOGDIR}" "${STAGE2_EVAL_CKPT}"
        else
            echo "[WARN]  checkpoint not found after training — check for errors"
        fi
    done
done

echo ""
echo "════ launch_backdoor.sh finished  DOMAIN=${DOMAIN}  tasks ${TASK_START}-${TASK_END} ════"
