# Pretrained VLM benchmark

Run from the repository root after generating the ten-sample dataset:

```bash
bash scripts/generate-sample.sh --samples 10
bash scripts/benchmark-vlm.sh
```

The default resource-conscious mode uses the previously selected
`direction-word-v2` prompt, float16 weights, two CPU threads, and at most eight
generated tokens. Validation and evaluation have tqdm progress bars with ETA
and per-sample latency. Each raw/parsed response is flushed to JSONL immediately;
Ctrl-C preserves completed records, but rerunning starts over (no resume yet).
Use a new output directory to preserve a previous experiment.

```bash
# Short resource check before the full dataset
bash scripts/benchmark-vlm.sh --val-limit 5 --eval-limit 3 \
  --out artifacts/benchmarks/smolvlm2-500m-light-check
# Full validation and evaluation using the fixed prompt
bash scripts/benchmark-vlm.sh --out artifacts/benchmarks/smolvlm2-500m-light
# Explicitly repeat the original prompt comparison when needed
bash scripts/benchmark-vlm.sh --refine-prompts --dtype float32 --max-new-tokens 64 \
  --out artifacts/benchmarks/smolvlm2-500m-prompt-comparison
```

On a 300-sample 65/25/10 split, the default makes 75 validation and 30 evaluation
predictions, plus one warm-up. Prompt refinement makes three validation passes.
Float16 and the smaller token budget change the inference configuration; compare
their quality separately from the historical float32 results below. The MPS
allocation figure measures live tensors, not all Metal driver memory or system
memory pressure.

The benchmark uses `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` through
Transformers and PyTorch MPS. The model receives only `rgb.png` and the natural
language instruction. Masks, boxes, object lists, bearings, and simulator state
remain evaluation-only ground truth.

With `--refine-prompts`, validation compares three prompt formats. Selection is deterministic:
direction accuracy, valid-response rate, coordinate grounding when present,
center error, then declaration order. The selected prompt is frozen before the
single evaluation sample runs. Generation uses seed 0, `do_sample=false`, and
temperature 0 semantics.

Outputs under `artifacts/benchmarks/smolvlm2-500m/`:

- `validation_runs.jsonl`: raw and parsed response for every prompt/sample pair.
- `prompt_selection.json`: candidate prompts, validation metrics, and winner.
- `eval_predictions.jsonl`: raw response, parsed prediction, score, and latency.
- `summary.json`: revision, device, determinism settings, metrics, time, and memory.
- `sample-*_overlay.png`: evaluation image with truth and prediction summary.

## Verified ten-sample pipeline result

- Model revision: `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Device and dtype: Apple MPS, float32.
- Coordinate JSON prompt: no valid validation responses; raw failures are saved.
- Both direction prompts: valid on 2/2 validation images and correct on 1/2.
- Selected prompt: `direction-word-v2` by the declared tie-break.
- Evaluation: valid `left`, correct on the one evaluation image.
- Cached model load: 3.99 seconds; evaluation inference: 5.11 seconds.
- Observed peaks: 819 MiB process RSS and 3,351 MiB MPS tensor allocation.

This is a working pipeline example, not a performance estimate. Two validation
and one evaluation sample have extremely high variance. The evaluation image
was also seen while debugging this pipeline, so the next larger dataset must
use a fresh held-out evaluation split before reporting model quality.
