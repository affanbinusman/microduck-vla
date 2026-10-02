#!/usr/bin/env bash
# Shared paths; no activation or global Python packages required.
[ -n "${BASH_VERSION:-}" ] || { echo "Run these scripts with bash." >&2; return 1; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="$ROOT/external/microduck-lab"
export PATH="/opt/homebrew/bin:$PATH"
export UV_CACHE_DIR="$ROOT/artifacts/cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/artifacts/python"
export HF_HOME="$ROOT/artifacts/cache/huggingface"
export npm_config_cache="$ROOT/artifacts/cache/npm"
export PYTHONUNBUFFERED=1
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
