#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
BACKEND_SESSION=microduck-vla-sim
VIEWER_SESSION=microduck-vla-viewer
LOGS="$ROOT/artifacts/logs"
case "${1:-help}" in
  backend|backend-log)
    mkdir -p "$LOGS"
    if [ "$1" = backend-log ]; then exec >>"$LOGS/simulator.log" 2>&1; fi
    cd "$LAB/microduck_local"
    exec uv run --locked duck-lab ../microduck/policies/alpha_walking.onnx --world playroom
    ;;
  viewer|viewer-log)
    mkdir -p "$LOGS"
    if [ "$1" = viewer-log ]; then exec >>"$LOGS/viewer.log" 2>&1; fi
    cd "$LAB/duck-viewer"
    exec npm run dev -- --hostname 127.0.0.1
    ;;
  start)
    if tmux has-session -t "$BACKEND_SESSION" 2>/dev/null || tmux has-session -t "$VIEWER_SESSION" 2>/dev/null; then
      echo "A Microduck session already exists. Use status or stop first."; exit 1
    fi
    python3 - <<'PY'
import socket
for port in (8788, 63317):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except PermissionError:
            raise SystemExit("Permission denied binding localhost; run outside the restricted sandbox.")
        except OSError:
            raise SystemExit(f"Port {port} is occupied; stop its server before starting Microduck.")
PY
    tmux new-session -d -s "$BACKEND_SESSION" -c "$ROOT" bash "$ROOT/scripts/sim.sh" backend-log
    tmux new-session -d -s "$VIEWER_SESSION" -c "$ROOT" bash "$ROOT/scripts/sim.sh" viewer-log
    ready=0
    for attempt in {1..45}; do
      if curl -fsS --max-time 2 http://127.0.0.1:8788/world >/dev/null 2>&1 && \
         curl -fsS --max-time 2 http://127.0.0.1:63317/sim >/dev/null 2>&1; then ready=1; break; fi
      sleep 1
    done
    if [ "$ready" != 1 ]; then
      echo "Servers are not ready. See $LOGS/simulator.log and $LOGS/viewer.log"; exit 1
    fi
    echo "Walking demo: http://localhost:63317"
    echo "Playroom:     http://localhost:63317/sim"
    ;;
  status)
    for session in "$BACKEND_SESSION" "$VIEWER_SESSION"; do
      if tmux has-session -t "$session" 2>/dev/null; then echo "$session: running"; else echo "$session: stopped"; fi
    done
    curl -fsS --max-time 3 http://127.0.0.1:8788/world || true
    echo
    ;;
  smoke)
    cd "$LAB/microduck_local"
    uv run --locked eval-walk ../microduck/policies/alpha_walking.onnx --episodes 3
    exec uv run --locked render-rollout \
      --policy ../microduck/policies/alpha_walking.onnx --behavior run \
      --episodes 1 --seconds 5 --out "$ROOT/artifacts/smoke/walking"
    ;;
  stop)
    for session in "$BACKEND_SESSION" "$VIEWER_SESSION"; do
      if tmux has-session -t "$session" 2>/dev/null; then
        tmux send-keys -t "$session" C-c
      fi
    done
    echo "Sent Ctrl-C to the Microduck simulator and viewer sessions."
    ;;
  *) echo "Usage: bash scripts/sim.sh {start|status|stop|backend|viewer|smoke}" ;;
esac
