#!/bin/bash
set -euo pipefail
export DOMAIN=dmc_manip
exec bash "$(dirname "$0")/../lib/launch_train.sh"
