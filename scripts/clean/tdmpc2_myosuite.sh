#!/bin/bash
set -euo pipefail
export DOMAIN=myosuite
exec bash "$(dirname "$0")/../lib/launch_train.sh"
