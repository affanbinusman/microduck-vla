# microduck-vla
VLA robotics experiments with Microduck simulation and local Apple Silicon training

## Start the stock simulation

```bash
bash scripts/setup-native.sh  # first setup or recreate environments
bash scripts/sim.sh start
```

Open [the walking demo](http://localhost:63317) or
[the playroom](http://localhost:63317/sim).

```bash
bash scripts/sim.sh status
bash scripts/sim.sh stop
bash scripts/sim.sh smoke     # physics check + five-second rendered video
```

See [native setup and controls](docs/setup.md) for isolation, pinned dependencies,
logs, tmux and validation. The current demo reuses the community Microduck Lab
simulator and Pollen Robotics' walking policy; project model training is pending.

## Extras
```bash
cd /Users/affanbinusman/Documents/GitHub/microduck-vla
```

For the simulator’s separate environment:
```bash
source external/microduck-lab/microduck_local/.venv/bin/activate
```
