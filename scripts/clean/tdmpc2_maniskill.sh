#!/bin/bash
set -euo pipefail

export DOMAIN=maniskill
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "${SCRIPT_DIR}/../lib/launch_train.sh"
