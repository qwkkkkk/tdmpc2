#!/bin/bash
set -euo pipefail
export DOMAIN=dmc_subtle
exec bash "$(dirname "$0")/../lib/launch_train.sh"
