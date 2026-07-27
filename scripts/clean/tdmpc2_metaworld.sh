#!/bin/bash
set -euo pipefail
export DOMAIN=metaworld
exec bash "$(dirname "$0")/../lib/launch_train.sh"
