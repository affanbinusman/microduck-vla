#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
for tool in git uv node npm tmux; do
  command -v "$tool" >/dev/null || { echo "Missing $tool (see docs/setup.md)"; exit 1; }
done
mkdir -p "$ROOT/artifacts/logs"
git -C "$ROOT" submodule update --init external/microduck-lab

# These are the upstream revisions required by the pinned lab's contract tests.
clone_at() {
  local repo="$1" sha="$2" path="$LAB/$1"
  if [ ! -e "$path" ]; then
    git clone "https://github.com/pollen-robotics/$repo.git" "$path"
  fi
  if [ "$(git -C "$path" rev-parse HEAD)" != "$sha" ]; then
    if [ -n "$(git -C "$path" status --porcelain)" ]; then
      echo "Refusing to change a modified upstream checkout: $path"; exit 1
    fi
    git -C "$path" checkout --detach "$sha"
  fi
}
clone_at microduck_rl badc4e7ffe5507fd7acb1a21487bd2925c1afe5a
clone_at microduck 2c61dcc1f03440541cdc0729f7a375b2a9ea3005

uv sync --project "$ROOT" --locked
uv sync --project "$LAB/microduck_local" --locked --python "$ROOT/.venv/bin/python"

# The pinned microduck revision already vendors these nine reference policies.
policies=(alpha_walking alpha_stand alpha_sitstand alpha_ground_pick
          ball_kick_left ball_kick_right roller roller_crouch roulade)
missing=()
for policy in "${policies[@]}"; do
  [ -s "$LAB/microduck/policies/$policy.onnx" ] || missing+=("$policy.onnx")
done
if [ "${#missing[@]}" -gt 0 ]; then
  uv run --project "$LAB/microduck_local" --locked hf download \
    pollen-robotics/microduck-policies "${missing[@]}" \
    --revision 088524a64e2557dc453256b6071dbb9d23888802 \
    --local-dir "$LAB/microduck/policies" --quiet
fi
(cd "$LAB/duck-viewer" && npm ci --no-audit --no-fund)
echo "Ready. Start the simulation with: bash scripts/sim.sh start"
