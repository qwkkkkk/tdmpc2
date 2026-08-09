#!/bin/bash
set -euo pipefail
export DOMAIN=maniskill3
exec bash "$(dirname "$0")/../lib/launch_train.sh"
