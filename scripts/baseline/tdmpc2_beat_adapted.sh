#!/bin/bash
set -euo pipefail
export BACKDOOR_VARIANT=beat_adapted
exec bash "$(dirname "$0")/../lib/run_backdoor_variant.sh"
