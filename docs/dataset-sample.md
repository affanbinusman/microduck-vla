# Dataset generation process

Run from the repository root:

```bash
bash scripts/generate-sample.sh --samples 10
```

Use any positive value, such as `--samples 1`, `--samples 10`, or
`--samples 100`. The command builds deterministic scene variations from
`configs/dataset/sample_v0.json`, aligns an offscreen MuJoCo camera to the
robot's `d0/head_camera` site, and hides the robot's own camera housing. It
calculates integer split sizes from the configured 70% train, 25% validation,
and 5% evaluation policy, then writes them under `artifacts/datasets/sample-v0/`.
For fewer than three samples, some splits are necessarily empty; from three
samples onward, each split receives at least one sample.

Each sample directory contains:

- `rgb.png`: the only image a model receives.
- `target_mask.png`: simulator-only pixel ground truth.
- `preview.png`: the RGB input with the derived bounding box and instruction.
- `annotation.json`: the target attributes, box, center, bearing, and distance.

The dataset root also contains `annotations.jsonl` for loading all samples and
`contact_sheet.png` for visual review. The configuration defines object
templates and scene positions; the annotation's nested `target` object records
which object was selected for that sample. The generator changes target,
placement, and small position offsets under successive seeds, and fails if a
selected target has no visible pixels. Each run replaces the generated split
directories so a smaller run cannot leave stale samples from a larger run.
