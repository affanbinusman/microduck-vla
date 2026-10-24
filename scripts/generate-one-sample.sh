#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$ROOT"
exec uv run --locked --project "$LAB/microduck_local" \
  python "$ROOT/scripts/generate-one-sample.py" "$@"
