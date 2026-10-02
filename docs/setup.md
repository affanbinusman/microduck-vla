# Native simulation setup

Run commands from `microduck-vla`. This setup uses the community
[Microduck Lab](https://github.com/jonathanhawkins/microduck-lab) simulator
and Pollen Robotics' shipped walking policy. No project model or dataset is
required for this first demo.

## Isolation

| Component | Local environment |
| --- | --- |
| Main project (empty until model work begins) | `.venv/` |
| MuJoCo simulator and its existing dependencies | `external/microduck-lab/microduck_local/.venv/` |
| Next.js viewer | `external/microduck-lab/duck-viewer/node_modules/` |
| Managed Python, package caches and logs | `artifacts/` (ignored by Git) |

The scripts select their environments automatically; no `source .venv/bin/activate`
is needed. Python 3.12.14 is pinned in `.python-version` and installed by uv for
Apple Silicon; both environments use that interpreter. Homebrew provides uv
and tmux, and the existing Node installation is reused. These virtual environments
isolate dependencies; they are not filesystem or process sandboxes.

The sibling `../microduck-lab` checkout remains available for simulator development.
The demo runs from the pinned submodule inside this project, so the two checkouts
should not run servers on the same ports simultaneously.

## Setup or recreate

```bash
cd /Users/affanbinusman/Documents/GitHub/microduck-vla
# Only needed on a fresh machine without these tools:
brew install uv tmux
# Node and npm must also be available (tested with Node 24.13.1).
bash scripts/setup-native.sh
```

The script uses `uv sync --locked` and `npm ci` to install the recorded dependency
versions. See [uv's lockfile documentation](https://docs.astral.sh/uv/concepts/projects/sync/).
It downloads the upstream robot assets into the lab's expected directory layout.
It does not install the official CUDA training environment or Rust robot runtime.
Re-running setup preserves modified upstream checkouts rather than replacing them.

| Source | Pinned revision |
| --- | --- |
| `affanbinusman/microduck-lab` submodule | `bbf0326ef97975f7062368914e0729504d17226f` |
| Pollen `microduck_rl` (MJCF/meshes only) | `badc4e7ffe5507fd7acb1a21487bd2925c1afe5a` |
| Pollen `microduck` (shipped policies) | `2c61dcc1f03440541cdc0729f7a375b2a9ea3005` |
| Policy download fallback (if vendored files are missing) | `088524a64e2557dc453256b6071dbb9d23888802` |

## Try the simulation

```bash
cd /Users/affanbinusman/Documents/GitHub/microduck-vla
bash scripts/sim.sh start
```

Open either:

- [Walking demo](http://localhost:63317): one duck runs the pretrained
  `alpha_walking.onnx` policy with the lab's automatic velocity commands.
- [Playroom](http://localhost:63317/sim): the stock room scenario with its existing
  controller. This uses the lab's simulated sensors; it is not a trained VLM/VLA.

The walking-demo keyboard moves the **camera**: drag to orbit, scroll to zoom,
W/S to move in/out, A/D to slide, and R to reset the simulation. In the playroom,
use the inspector and scene menu to explore the stock scenarios.

Both servers bind loopback: backend `127.0.0.1:8788`, viewer `127.0.0.1:63317`.
The start command refuses existing sessions or occupied ports. If the viewer
reports disconnected, inspect the backend log and use the default localhost URL.

```bash
bash scripts/sim.sh status
tail -f artifacts/logs/simulator.log
tail -f artifacts/logs/viewer.log
bash scripts/sim.sh stop
```

The two named tmux sessions survive closing your terminal:

```bash
tmux attach -t microduck-vla-sim
tmux attach -t microduck-vla-viewer
```

Detach with Ctrl-B, then D. Logs append across launches. `stop` sends Ctrl-C only
to these two sessions; it does not stop other tmux sessions. macOS sleep pauses
work and reboot ends the sessions. Optionally run `caffeinate -i` in another
terminal while watching a long demo, then Ctrl-C it when finished.

For visible foreground logs, use two terminal windows instead of `start`:

```bash
# Terminal 1
bash scripts/sim.sh backend
# Terminal 2, from the same project directory
bash scripts/sim.sh viewer
```

## Headless physics check and rendered video

```bash
bash scripts/sim.sh smoke
```

This evaluates three episodes with the shipped walking policy and renders a
five-second forward walking rollout. Outputs:

- `artifacts/smoke/walking/ep0.mp4`
- `artifacts/smoke/walking/ep0_sheet.png`

Re-running this smoke command replaces those sample outputs. Rendering on this
Mac uses MuJoCo's native CGL backend; do not set Linux `MUJOCO_GL=egl` here.
If running through a restricted agent sandbox, local servers, Metal and CGL may
require approval to run outside it. Ordinary Terminal launches do not use that
agent sandbox.

## Initial verification (October 1, 2026)

For a full Apple GPU training smoke test, run:

```bash
bash scripts/test-gpu.sh
# Or with the simulator environment activated:
python scripts/test_apple_gpu.py
```

The test uses [PyTorch's MPS backend](https://docs.pytorch.org/docs/2.9/notes/mps.html).
It learns random linear coefficients from synthetic inputs with small target
noise, using 2,048 training samples and 512 held-out samples. It prints the chip,
device, loss, training time and allocated GPU tensor memory, and fails if MPS
is unavailable, gradients are not on MPS, weights do not change, or held-out
loss does not drop by at least 80%. CPU fallback is disabled. No data files or
checkpoints are created.

Verified on this Apple M3: 200 optimizer steps completed in 0.63 seconds;
held-out loss fell from 2.666017 to 0.000105. This confirms a small synthetic
training workload; model-specific training support remains to be tested.

- Native arm64 macOS 15.7.7; managed Python 3.12.14, uv 0.12.21, tmux 3.7c.
- Locked simulator dependencies installed (MuJoCo 3.10.0, PyTorch 2.9.1).
- Upstream contract/world/brain smoke suite: **56 passed, 1 skipped** (optional
  MARS robot assets were not installed).
- Viewer tests: **356 passed across 27 files**.
- Five-second CGL-rendered walking rollout: completed 250 steps, remained upright;
  inspected the resulting contact sheet.
- User-facing `smoke` command: three full 1,000-step walking episodes, zero falls.
- Simulator and viewer started in tmux; viewer HTTP route returned 200.
- A real WebSocket frame confirmed the duck on the lab stage.
- PyTorch MPS reported available and computed a small tensor operation correctly.
  The synthetic training test above subsequently verified backward passes and
  optimizer updates as well.

The browser automation tool could not bind an in-app tab, so browser canvas
appearance has not yet been verified by automation. Use the localhost links above
to view it. No robotics model training, dataset download, commit or push was performed.
