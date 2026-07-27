#!/bin/bash
set -euo pipefail
export DOMAIN=dmc
exec bash "$(dirname "$0")/../lib/launch_train.sh"
