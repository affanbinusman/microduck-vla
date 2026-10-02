#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
# Reuse the simulator's already-installed PyTorch; keep the main env lightweight.
unset PYTORCH_ENABLE_MPS_FALLBACK
exec uv run --locked --project "$LAB/microduck_local" \
  python "$ROOT/scripts/test_apple_gpu.py" "$@"
