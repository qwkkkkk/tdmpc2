#!/bin/bash
set -euo pipefail
export BACKDOOR_VARIANT=static_latent
exec bash "$(dirname "$0")/../lib/run_backdoor_variant.sh"
