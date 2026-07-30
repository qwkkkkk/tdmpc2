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
    if [ "${DOMAIN}" = "myosuite" ]; then
        export EVAL_FREQ=10000
    else
        export EVAL_FREQ=5000
    fi
fi
if [ -z "${EVAL_TRIG_START:-}" ]; then
    if [ "${DOMAIN}" = "metaworld" ]; then
        export EVAL_TRIG_START=50
    elif [ "${DOMAIN}" = "myosuite" ]; then
        export EVAL_TRIG_START=42
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
        export CAUSAL_MODE=${CAUSAL_MODE:-off}
        export CAUSAL_GAMMA=${CAUSAL_GAMMA:-0.0}
        ;;

    reward|reward_only)
        export RESULT_METHOD=${RESULT_METHOD:-reward_only}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-reward_only}
        export BETA=${BETA:-0.0}
        export REWARD_ONLY_VALUE=${REWARD_ONLY_VALUE:-10.0}
        export CAUSAL_MODE=${CAUSAL_MODE:-off}
        export CAUSAL_GAMMA=${CAUSAL_GAMMA:-0.0}
        ;;

    beat|beat_adapted)
        export RESULT_METHOD=${RESULT_METHOD:-beat_adapted}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-beat_adapted}
        export BETA=${BETA:-0.0}
        export BEAT_BETA=${BEAT_BETA:-0.05}
        export BEAT_NLL_ALPHA=${BEAT_NLL_ALPHA:-0.0}
        export BEAT_TRIGGER_WEIGHT=${BEAT_TRIGGER_WEIGHT:-1.0}
        export BEAT_CLEAN_WEIGHT=${BEAT_CLEAN_WEIGHT:-1.0}
        export CAUSAL_MODE=${CAUSAL_MODE:-off}
        export CAUSAL_GAMMA=${CAUSAL_GAMMA:-0.0}
        ;;

    reflective|score_margin)
        export RESULT_METHOD=${RESULT_METHOD:-reflective}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-score_margin}
        export BETA=${BETA:-0.0}
        export CAUSAL_MODE=${CAUSAL_MODE:-off}
        export CAUSAL_GAMMA=${CAUSAL_GAMMA:-0.0}
        ;;

    ours|causal_open)
        export RESULT_METHOD=${RESULT_METHOD:-causal_open}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-score_margin}
        export BETA=${BETA:-0.0}
        export CAUSAL_MODE=${CAUSAL_MODE:-open}
        export CAUSAL_GAMMA=${CAUSAL_GAMMA:-0.5}
        export CAUSAL_HORIZON=${CAUSAL_HORIZON:-3}
        export CAUSAL_WARMUP=${CAUSAL_WARMUP:-1000}
        export CAUSAL_LOSS_CLIP=${CAUSAL_LOSS_CLIP:-0.0}
        ;;

    *)
        echo "[error] unknown BACKDOOR_VARIANT='${BACKDOOR_VARIANT}'"
        echo "        Use: static_latent | reward_only | beat_adapted | reflective | ours"
        exit 1
        ;;
esac

echo "[backdoor:${BACKDOOR_VARIANT}] DOMAIN=${DOMAIN} OBS_OVERRIDE=${OBS_OVERRIDE} TASK_START=${TASK_START:-<default>} TASK_END=${TASK_END:-<default>}"
exec bash scripts/lib/launch_backdoor.sh
