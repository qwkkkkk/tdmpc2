#!/bin/bash
set -euo pipefail

# Backward-compatible filename for old queues. New formal launches should use
# tdmpc2_mirage.sh; both resolve to the same real post-intervention method.
exec bash "$(dirname "$0")/tdmpc2_mirage.sh"
