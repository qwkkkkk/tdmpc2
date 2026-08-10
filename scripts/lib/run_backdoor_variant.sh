#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export DOMAIN=${DOMAIN:-metaworld}
export OBS_OVERRIDE=${OBS_OVERRIDE:-rgb}
export BACKDOOR_VARIANT=${BACKDOOR_VARIANT:-reflective}

# Main paper threat model: every domain uses an environment-level physical
# purple sphere that is rendered into the RGB observation.
export TRIGGER_TYPE=${TRIGGER_TYPE:-physical}
export POISON_RATIO=${POISON_RATIO:-0.3}
export ALPHA=${ALPHA:-1.0}
export LAMBDA_SCORE=${LAMBDA_SCORE:-1.0}
export K_NEG=${K_NEG:-4}
export K_SEL=${K_SEL:-4}
if [ -z "${EVAL_FREQ:-}" ]; then
    if [ "${DOMAIN}" = "myosuite" ] || [ "${DOMAIN}" = "maniskill3" ]; then
        export EVAL_FREQ=10000
    else
        export EVAL_FREQ=5000
    fi
fi
if [ -z "${EVAL_TRIG_START:-}" ]; then
    if [ "${DOMAIN}" = "metaworld" ]; then
        export EVAL_TRIG_START=50
    elif [ "${DOMAIN}" = "maniskill" ]; then
        export EVAL_TRIG_START=50
    elif [ "${DOMAIN}" = "maniskill3" ]; then
        export EVAL_TRIG_START=10
    elif [ "${DOMAIN}" = "myosuite" ]; then
        export EVAL_TRIG_START=42
    elif [ "${DOMAIN}" = "dmc_manip" ]; then
        export EVAL_TRIG_START=62
    elif [ "${DOMAIN}" = "robodesk" ]; then
        export EVAL_TRIG_START=125
    else
        export EVAL_TRIG_START=250
    fi
fi
export EVAL_EPISODES=${EVAL_EPISODES:-10}

case "${BACKDOOR_VARIANT}" in
    latent_only|static_latent)
        export RESULT_METHOD=${RESULT_METHOD:-static_latent}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-static_latent}
        export BETA=${BETA:-0.0}
        export STATIC_TARGET_TOPK=${STATIC_TARGET_TOPK:-64}
        export STATIC_TARGET_METRIC=${STATIC_TARGET_METRIC:-score_margin}
        export PERSISTENCE_VARIANT=none
        ;;

    reward|reward_only)
        export RESULT_METHOD=${RESULT_METHOD:-reward_only}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-reward_only}
        export BETA=${BETA:-0.0}
        export REWARD_ONLY_VALUE=${REWARD_ONLY_VALUE:-10.0}
        export PERSISTENCE_VARIANT=none
        ;;

    beat|beat_adapted)
        export RESULT_METHOD=${RESULT_METHOD:-beat_adapted}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-beat_adapted}
        export BETA=${BETA:-0.0}
        export BEAT_BETA=${BEAT_BETA:-0.05}
        export BEAT_NLL_ALPHA=${BEAT_NLL_ALPHA:-0.0}
        export BEAT_TRIGGER_WEIGHT=${BEAT_TRIGGER_WEIGHT:-1.0}
        export BEAT_CLEAN_WEIGHT=${BEAT_CLEAN_WEIGHT:-1.0}
        export PERSISTENCE_VARIANT=none
        ;;

    reflective|score_margin)
        export RESULT_METHOD=${RESULT_METHOD:-reflective}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-score_margin}
        export BETA=${BETA:-0.0}
        export PERSISTENCE_VARIANT=none
        ;;

    ours|mirage|post)
        # Canonical MIRAGE: train on real simulator histories after the
        # physical trigger is withdrawn. Hard-negative mining remains an
        # internal TD-MPC2 decision-loss implementation detail.
        export RESULT_METHOD=${RESULT_METHOD:-mirage}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-score_margin}
        export BETA=${BETA:-0.0}
        export PERSISTENCE_VARIANT=post
        export POST_GAMMA=${POST_GAMMA:-0.5}
        export POST_HORIZON=${POST_HORIZON:-8}
        export POST_PREFILL_ROLLOUTS=${POST_PREFILL_ROLLOUTS:-8}
        export POST_MIN_BUFFER=${POST_MIN_BUFFER:-8}
        ;;

    causal_open|imag)
        # Historical imagined-dynamics mechanism, retained only as an ablation.
        export RESULT_METHOD=${RESULT_METHOD:-causal_imag}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-score_margin}
        export BETA=${BETA:-0.0}
        export PERSISTENCE_VARIANT=imag
        export IMAG_MODE=${IMAG_MODE:-open}
        export IMAG_GAMMA=${IMAG_GAMMA:-0.5}
        ;;

    both)
        # Mechanism analysis only; never aggregate this row as MIRAGE.
        export RESULT_METHOD=${RESULT_METHOD:-causal_both}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-score_margin}
        export BETA=${BETA:-0.0}
        export PERSISTENCE_VARIANT=both
        ;;

    *)
        echo "[error] unknown BACKDOOR_VARIANT='${BACKDOOR_VARIANT}'"
        echo "        Main: mirage | static_latent | reward_only | beat_adapted | reflective"
        echo "        Ablations: imag | both"
        exit 1
        ;;
esac

echo "[backdoor:${BACKDOOR_VARIANT}] DOMAIN=${DOMAIN} OBS_OVERRIDE=${OBS_OVERRIDE} TASK_START=${TASK_START:-<default>} TASK_END=${TASK_END:-<default>}"
exec bash scripts/lib/launch_backdoor.sh
