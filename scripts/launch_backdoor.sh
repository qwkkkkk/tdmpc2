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
#   tdmpc2/logs/<domain>/clean/tdmpc2_<task>_<STAGE1_EXP>_s<seed>/models/final.pt
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
#               Trigger: learned invis δ (or white patch if TRIGGER_TYPE=white).
#   metaworld — MetaWorld state tasks by default. Use OBS_OVERRIDE=rgb with
#               TRIGGER_TYPE=physical for the physical marker main experiment.
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
#   STAGE1_EXP — the clean tag used during stage-1 clean training.
#                Checkpoint resolved as:
#                  logs/<domain>/clean/tdmpc2_<task>_<STAGE1_EXP>_s<seed>/models/final.pt
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
#   TRIGGER_TYPE — invis (learned δ, default) or white (fixed patch).
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
#   WINDOW_K — injection window length (in obs frames per training sequence).
#              Sequences in a batch have horizon+1 = 4 frames (horizon=3).
#              -1 = persistent: inject from random t* to sequence end (default)
#               0 = full: inject all frames (t*=0)
#               K > 0 = window: inject K frames from random t*
#              At training scale K≥4 ≡ full (batch only has 4 frames).
#              At eval scale K is used directly (episodes are 1000 steps).
# ============================================================
TRIGGER_TYPE=${TRIGGER_TYPE:-invis}
TRIGGER_EPS=${TRIGGER_EPS:-8}
TRIGGER_LR=${TRIGGER_LR:-0.01}
TRIGGER_SIZE=${TRIGGER_SIZE:-8}
TRIGGER_VALUE=${TRIGGER_VALUE:-255}
STATE_TRIGGER_EPS=${STATE_TRIGGER_EPS:-0.05}
PHYS_TRIGGER_SIZE=${PHYS_TRIGGER_SIZE:-0.045}
PHYS_TRIGGER_OFFSET=${PHYS_TRIGGER_OFFSET:-"[0.0,-0.55,0.12]"}
PHYS_TRIGGER_FOLLOW_BODY=${PHYS_TRIGGER_FOLLOW_BODY:-torso}
PHYS_PROXY_SIZE=${PHYS_PROXY_SIZE:-8}
PHYS_PROXY_VALUE=${PHYS_PROXY_VALUE:-255}
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
ASR_COS_THRESHOLD=${ASR_COS_THRESHOLD:-0.9}
ASR_MIN_NORM=${ASR_MIN_NORM:-0.1}
POLICY_DRIFT_INTERVAL=${POLICY_DRIFT_INTERVAL:-1000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}

# ============================================================
# Trigger tag helper (used inside the seed loop for EXP_NAME)
# ============================================================
if [ "${TRIGGER_TYPE}" = "invis" ]; then
    TRIG_TAG="invis${TRIGGER_EPS}"
elif [ "${TRIGGER_TYPE}" = "state" ]; then
    TRIG_TAG="state${STATE_TRIGGER_EPS}"
elif [ "${TRIGGER_TYPE}" = "physical" ]; then
    TRIG_TAG="physical${PHYS_TRIGGER_SIZE}"
else
    TRIG_TAG="white${TRIGGER_SIZE}"
fi
if [ "${ATTACK_OBJECTIVE}" != "score_margin" ] && [ "${ATTACK_OBJECTIVE}" != "reflective" ]; then
    TRIG_TAG="${TRIG_TAG}_${ATTACK_OBJECTIVE}"
fi
if [ "${CAUSAL_MODE}" != "off" ]; then
    TRIG_TAG="${TRIG_TAG}_c${CAUSAL_MODE}_h${CAUSAL_HORIZON}_g${CAUSAL_GAMMA}"
fi

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
    mw-drawer-open       # paired drawer task for backdoor ablations
    mw-drawer-close      # high success; physical disruption clear
    mw-window-close      # stable success across all three victim models
    mw-button-press      # TD-MPC2 stable; DreamerV3 80%+ acceptable
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
        OBS=${OBS_OVERRIDE:-$OBS}
        MUJOCO_GL_NEEDED=false
        if [ "${OBS}" = "rgb" ]; then
            MUJOCO_GL_NEEDED=true
        fi
        if [ "${TRIGGER_TYPE}" = "invis" ]; then
            TRIGGER_TYPE=state
            TRIG_TAG="state${STATE_TRIGGER_EPS}"
        fi
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
echo "  [stage-2 backdoor]  DOMAIN=${DOMAIN}  obs=${OBS}  GPU=${GPU_ID}"
echo "  tasks ${TASK_START}–${TASK_END}/${TOTAL_ALL}  seeds ${SEED_START}..${SEED_END}"
echo "  stage-1 exp: ${STAGE1_EXP}"
echo "  stage-2 logdir: logs/${DOMAIN}/backdoor/tdmpc2_<task>_${TRIG_TAG}_w${WINDOW_K}_pr${POISON_RATIO}_a${ALPHA}_b${BETA}_lscore${LAMBDA_SCORE}_sk${K_SEL}_s<seed>"
echo "  steps=${STAGE2_STEPS}  model_size=${MODEL_SIZE}"
echo "  trigger: type=${TRIGGER_TYPE}  eps=${TRIGGER_EPS}px  lr=${TRIGGER_LR}  window_k=${WINDOW_K}"
echo "           phys_size=${PHYS_TRIGGER_SIZE}  phys_offset=${PHYS_TRIGGER_OFFSET}  phys_follow=${PHYS_TRIGGER_FOLLOW_BODY}"
echo "  target_action=${TARGET_ACTION_VALUE}  poison_ratio=${POISON_RATIO}"
echo "  loss: attack_objective=${ATTACK_OBJECTIVE}  alpha=${ALPHA}  beta=${BETA}  lambda_score=${LAMBDA_SCORE}  margin=${MARGIN}"
echo "        K_neg=${K_NEG}  K_sel=${K_SEL}"
echo "        static_topk=${STATIC_TARGET_TOPK}  static_metric=${STATIC_TARGET_METRIC}  reward_only=${REWARD_ONLY_VALUE}"
echo "        beat_beta=${BEAT_BETA}  beat_nll=${BEAT_NLL_ALPHA}  beat_w=(${BEAT_TRIGGER_WEIGHT},${BEAT_CLEAN_WEIGHT})"
echo "        causal_mode=${CAUSAL_MODE}  causal_gamma=${CAUSAL_GAMMA}  causal_horizon=${CAUSAL_HORIZON}"
echo "  asr: cos_threshold=${ASR_COS_THRESHOLD}  min_norm=${ASR_MIN_NORM}"
echo "════════════════════════════════════════════════════════════════════════"
for i in "${!tasks[@]}"; do printf "  %2d  %s\n" $((i+1)) "${tasks[$i]}"; done
echo ""

# ============================================================
# Backdoor training loop
# ============================================================
for task in "${TASKS_SLICE[@]}"; do
    for seed in $(seq $SEED_START $SEED_STEP $SEED_END); do

        # ── Per-run naming (R2-Dreamer-style flat path) ──────────────────
        # task_short: replace hyphens with underscores (walker-walk → walker_walk)
        task_short="${task//-/_}"
        # Strip trailing .0 from floats so 1.0 → 1, 0.3 → 0.3
        _fmt() { awk "BEGIN{printf \"%g\",$1}"; }
        run_exp="${EXP_NAME:-tdmpc2_${task_short}_${TRIG_TAG}_w${WINDOW_K}_pr$(_fmt ${POISON_RATIO})_a$(_fmt ${ALPHA})_b$(_fmt ${BETA})_lscore$(_fmt ${LAMBDA_SCORE})_sk${K_SEL}_s${seed}}"

        # stage-2 logdir mirrors R2-Dreamer: logs/{domain}/backdoor/{run_exp}/
        STAGE2_LOGDIR="${REPO_TDMPC2}/logs/${DOMAIN}/backdoor/${run_exp}"
        STAGE2_CKPT="${STAGE2_LOGDIR}/models/final.pt"

        # stage-1 clean logdir mirrors R2-Dreamer: logs/{domain}/clean/{stage1_run_exp}/
        stage1_run_exp="${STAGE1_RUN_EXP:-tdmpc2_${task_short}_${STAGE1_EXP}_s${seed}}"
        STAGE1_CKPT="${REPO_TDMPC2}/logs/${DOMAIN}/clean/${stage1_run_exp}/models/final.pt"
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
            seed="${seed}" \
            model_size="${MODEL_SIZE}" \
            steps="${STAGE2_STEPS}" \
            eval_freq="${EVAL_FREQ}" \
            eval_episodes="${EVAL_EPISODES}" \
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
            phys_trigger_offset=${PHYS_TRIGGER_OFFSET} \
            phys_trigger_follow_body=${PHYS_TRIGGER_FOLLOW_BODY} \
            phys_proxy_size=${PHYS_PROXY_SIZE} \
            phys_proxy_value=${PHYS_PROXY_VALUE} \
            target_action_value=${TARGET_ACTION_VALUE} \
            poison_ratio=${POISON_RATIO} \
            window_k=${WINDOW_K} \
            k_neg=${K_NEG} \
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
            causal_mode=${CAUSAL_MODE} \
            causal_gamma=${CAUSAL_GAMMA} \
            causal_horizon=${CAUSAL_HORIZON} \
            causal_warmup=${CAUSAL_WARMUP} \
            causal_loss_clip=${CAUSAL_LOSS_CLIP} \
            asr_cos_threshold=${ASR_COS_THRESHOLD} \
            asr_min_norm=${ASR_MIN_NORM} \
            policy_drift_interval=${POLICY_DRIFT_INTERVAL} \
            save_interval=${SAVE_INTERVAL}

        if [[ -f "${STAGE2_CKPT}" ]]; then
            echo "── DONE   ${run_exp} ──"
        else
            echo "[WARN]  checkpoint not found after training — check for errors"
        fi
    done
done

echo ""
echo "════ launch_backdoor.sh finished  DOMAIN=${DOMAIN}  tasks ${TASK_START}-${TASK_END} ════"
