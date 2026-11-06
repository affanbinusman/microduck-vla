#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$ROOT"
uv run --locked --project "$LAB/microduck_local" python scripts/diagnose-grounding.py --prepare "$@"
exec uv run --locked python scripts/diagnose-grounding.py "$@"
