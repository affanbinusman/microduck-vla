# Validation-only grounding diagnostic

```bash
# Three scenes, nine questions per prompt, two prompts
bash scripts/diagnose-grounding.sh --scenes 3 \
  --out artifacts/benchmarks/grounding-diagnostic-smoke
# Twelve scenes, 36 questions per prompt, two prompts
bash scripts/diagnose-grounding.sh
```

Selects equal numbers of validation scenes by their original target direction.
Every selected image is queried about all three objects using two prompt
templates: a simple position question and an explicit three-column question.
`middle` is normalized to `center`; raw answers are preserved. The model uses
the pinned SmolVLM2 revision, float16, greedy decoding, eight output tokens,
and two CPU threads. No evaluation images are loaded.

The simulator environment reconstructs each selected scene and checks exact RGB
equality before extracting bounding boxes from segmentation. This provides
independent truth for the additional object questions. A changed generator,
configuration, or dataset causes a preparation failure rather than silently
using mismatched labels.

The color baseline thresholds RGB channel ratios and localizes the visible
color pixels. It uses only the image and requested color, and is intentionally
specific to the three colors in sample-v0. Because color and shape are paired
in this dataset, its success does not establish shape recognition.

Outputs include `cases.jsonl`, `baseline.jsonl`, `simple.jsonl`, `columns.jsonl`,
and `summary.json` with confusion matrices (rows=true, columns=predicted),
valid rates, accuracy, counts of scenes with all three questions correct, and
counts receiving the same answer for every object. Incorrect VLM answers get
individual failure overlays. Responses are flushed after every question;
rerunning starts over, so use a new output directory to retain older runs.

## Completed validation diagnostic

The 12-scene run produced 36 questions per prompt, balanced at 12 per direction.
All saved images matched their independently reconstructed RGB renders.

| Method | Correct | Left | Middle | Right |
| --- | --- | --- | --- | --- |
| Color baseline | 36/36 | 12/12 | 12/12 | 12/12 |
| Simple prompt | 24/36 | 12/12 | 0/12 | 12/12 |
| Explicit columns | 12/36 | 12/12 | 0/12 | 0/12 |

All VLM answers were parseable. The columns prompt answered left for every
object in every scene. The simple prompt changed answers when the requested
object changed, but mapped middle targets to left (8) or right (4). Neither
prompt resolved the middle-region failure. Peak live MPS tensor allocation
was 1,718 MiB; peak process RSS was 3,143 MiB (overlapping measurements).

These observations support further localization diagnostics or supervised
adaptation, rather than assuming more zero-shot prompt wording will solve the
task. The color baseline is a scene-specific reference, not evidence that the
VLM recognizes arbitrary shapes. Evaluation results were not used in this run.
