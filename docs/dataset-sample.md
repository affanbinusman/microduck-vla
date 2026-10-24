# One-sample dataset process

Run from the repository root:

```bash
bash scripts/generate-one-sample.sh
```

The command builds the scene in `configs/dataset/sample_v0.json`, aligns an
offscreen MuJoCo camera to the robot's `d0/head_camera` site, hides the robot's
own camera housing, and writes one sample to
`artifacts/datasets/sample-v0/sample-000000/`.

The sample contains:

- `rgb.png`: the only image a model receives.
- `target_mask.png`: simulator-only pixel ground truth.
- `preview.png`: the RGB input with the derived bounding box and instruction.
- `annotation.json`: the target attributes, box, center, bearing, and distance.

Edit one scene value at a time, rerun the command, and inspect `preview.png`.
Useful first iterations are object position, size, color, distractor placement,
instruction wording, and camera resolution. The generator fails when the target
has no visible pixels, preventing an invalid sample from entering the dataset.
