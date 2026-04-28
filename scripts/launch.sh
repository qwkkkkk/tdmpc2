#!/bin/bash
# One-shot launcher: starts all training jobs in named tmux sessions.
#
# Usage:
#   bash scripts/launch.sh          # launches everything
#   bash scripts/launch.sh dmc      # only DMC jobs
#   bash scripts/launch.sh subtle   # only DMC-subtle jobs
#   bash scripts/launch.sh mw       # only MetaWorld jobs
#
# Monitor:
#   tmux ls                         # list sessions
#   tmux a -t dmc_0                 # attach to a session
#   Ctrl-b d                        # detach
#
# Kill all launched sessions:
#   tmux ls | grep -E "^(dmc|subtle|mw)_" | cut -d: -f1 | xargs -I{} tmux kill-session -t {}

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ════════════════════════════════════════════════════════════════════════
# Edit these to match your server
# ════════════════════════════════════════════════════════════════════════
SEED_START=1
SEED_END=1          # set to 3 for multi-seed runs
SEED_STEP=1
STEPS=500000        # 500K wrapper calls × action_repeat 2 = 1M env-side frames
# ════════════════════════════════════════════════════════════════════════

FILTER="${1:-all}"

launch() {
    local session="$1"; shift
    local cmd="$*"
    if tmux has-session -t "${session}" 2>/dev/null; then
        echo "[SKIP]  tmux session '${session}' already exists"
    else
        tmux new-session -d -s "${session}" \
            "bash -c '${cmd}; echo \"=== ${session} finished ===\"; exec bash'"
        echo "[START] tmux session '${session}'"
    fi
}

COMMON="SEED_START=${SEED_START} SEED_END=${SEED_END} SEED_STEP=${SEED_STEP} STEPS=${STEPS}"

# ── DMC (20 tasks, split across 2 GPUs) ──────────────────────────────────────
if [[ "${FILTER}" == "all" || "${FILTER}" == "dmc" ]]; then
    echo ""
    echo "=== Launching DMC jobs ==="
    launch "dmc_0" \
        "${COMMON} GPU_ID=0 TASK_START=1  TASK_END=10 bash ${SCRIPT_DIR}/train_dmc.sh"
    launch "dmc_1" \
        "${COMMON} GPU_ID=0 TASK_START=11 TASK_END=20 bash ${SCRIPT_DIR}/train_dmc.sh"
fi

# ── DMC subtle (5 tasks, single GPU) ─────────────────────────────────────────
if [[ "${FILTER}" == "all" || "${FILTER}" == "subtle" ]]; then
    echo ""
    echo "=== Launching DMC-subtle jobs ==="
    launch "subtle_0" \
        "${COMMON} GPU_ID=0 TASK_START=1 TASK_END=5 bash ${SCRIPT_DIR}/train_dmc_subtle.sh"
fi

# ── MetaWorld (50 tasks, split across 3 GPUs) ────────────────────────────────
if [[ "${FILTER}" == "all" || "${FILTER}" == "mw" ]]; then
    echo ""
    echo "=== Launching MetaWorld jobs ==="
    launch "mw_0" \
        "${COMMON} GPU_ID=0 TASK_START=1  TASK_END=17 bash ${SCRIPT_DIR}/train_metaworld.sh"
    launch "mw_1" \
        "${COMMON} GPU_ID=0 TASK_START=18 TASK_END=34 bash ${SCRIPT_DIR}/train_metaworld.sh"
    launch "mw_2" \
        "${COMMON} GPU_ID=0 TASK_START=35 TASK_END=50 bash ${SCRIPT_DIR}/train_metaworld.sh"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Active sessions:"
tmux ls 2>/dev/null | grep -E "^(dmc|subtle|mw)_" || echo "  (none)"
echo ""
echo "  Attach:  tmux a -t <session>"
echo "  List:    tmux ls"
echo "════════════════════════════════════════════════════════════════════════"
