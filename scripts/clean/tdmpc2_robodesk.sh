#!/bin/bash
set -euo pipefail
export DOMAIN=robodesk
exec bash "$(dirname "$0")/../lib/launch_train.sh"
