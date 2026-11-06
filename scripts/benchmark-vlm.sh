#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$ROOT"
exec uv run --locked python "$ROOT/scripts/benchmark-vlm.py" "$@"
