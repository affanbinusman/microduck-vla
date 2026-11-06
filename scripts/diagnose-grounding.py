#!/usr/bin/env python3
"""Validation-only prompt and instruction-sensitivity diagnostic."""
import argparse
import json
import random
import runpy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ("left", "center", "right")
PROMPTS = {
    "simple": "Where is the {color} {shape}? Answer only: left, middle, or right.",
    "columns": "Divide the image into three equal vertical columns. Which column contains the center of the {color} {shape}? Answer only: left, middle, or right.",
}


def direction(x, width):
    return "left" if x / width < 1/3 else "right" if x / width > 2/3 else "center"


def prepare(args):
    import mujoco
    g = runpy.run_path(str(ROOT / "scripts/generate-sample.py"))
    config = g["load_config"](ROOT / "configs/dataset/sample_v0.json")
    rows = [json.loads(s) for s in (args.dataset / "annotations.jsonl").read_text().splitlines()]
    selected = []
    for label in DIRECTIONS:
        candidates = [r for r in rows if r["split"] == "val" and direction(r["target"]["center_xy"][0], r["width"]) == label]
        if len(candidates) < args.scenes // 3:
            raise ValueError(f"Not enough validation scenes for {label}")
        selected.extend(candidates[:args.scenes // 3])
    cases = []
    for row in selected:
        # Reconstruct segmentation at the exact saved pose. Verify RGB equality
        # before trusting the new labels; color thresholding never supplies truth.
        index = int(row["sample_id"].split("-")[-1])
        seed, objects, _, _ = g["sample_scene"](config, index)
        if objects and [{k: o[k] for k in ("id", "color", "shape", "position")} for o in objects] != row["objects"]:
            raise ValueError("Scene configuration changed since dataset generation")
        world = g["World"](g["make_scenario"](config, seed, objects), seed=seed)
        camera = row["camera"]
        g["align_camera_to_site"](world.model, camera)
        mujoco.mj_forward(world.model, world.data)
        bodies = world.duck_bodies[config["robot"]["id"]]
        hidden = (world.model.geom_bodyid >= bodies.start) & (world.model.geom_bodyid < bodies.stop)
        world.model.geom_group[hidden] = 5
        options = mujoco.MjvOption()
        options.geomgroup[5] = 0
        renderer = g["offscreen_renderer"](world.model, row["width"], row["height"])
        try:
            renderer.update_scene(world.data, camera=camera, scene_option=options)
            rgb = renderer.render().copy()
            saved = np.asarray(Image.open(args.dataset / row["image"]).convert("RGB"))
            if not np.array_equal(rgb, saved):
                raise ValueError(f"Rendered image no longer matches {row['sample_id']}")
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render().copy()
        finally:
            renderer.close()
        for obj in objects:
            geom = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, obj["id"] + "_geom")
            box, _ = g["target_bbox"](segmentation, geom)
            center = (box[0] + box[2]) / 2
            cases.append({"sample_id": row["sample_id"], "image": row["image"], "object_id": obj["id"], "color": obj["color"], "shape": obj["shape"], "bbox": box, "expected": direction(center, row["width"])})
    (args.out / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    print(f"Prepared {len(cases)} questions from {len(selected)} validation scenes; all RGB reconstructions match.")


def color_prediction(image, color):
    a = np.asarray(image).astype(float) / 255
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    masks = {
        "red": (r > 0.15) & (r > 1.5*g) & (r > 1.5*b),
        "blue": (b > 0.15) & (b > 1.5*r) & (b > 1.2*g),
        "yellow": (r > 0.15) & (g > 0.12) & (r > 1.8*b) & (g > 1.8*b),
    }
    _, xs = np.nonzero(masks[color])
    return None if not len(xs) else direction((xs.min() + xs.max() + 1)/2, image.width)


def summarize(records):
    confusion = {d: {p: 0 for p in (*DIRECTIONS, "invalid")} for d in DIRECTIONS}
    for r in records:
        confusion[r["expected"]][r["prediction"] or "invalid"] += 1
    groups = {}
    for r in records:
        groups.setdefault(r["sample_id"], []).append(r)
    return {"questions": len(records), "accuracy": sum(r["prediction"] == r["expected"] for r in records)/len(records), "valid_rate": sum(r["prediction"] is not None for r in records)/len(records), "confusion": confusion, "all_three_correct_scenes": sum(all(r["prediction"] == r["expected"] for r in group) for group in groups.values()), "scenes_with_same_answer_for_all_objects": sum(len({r["prediction"] for r in group}) == 1 for group in groups.values())}


def infer(args):
    import torch
    from tqdm.auto import tqdm
    b = runpy.run_path(str(ROOT / "scripts/benchmark-vlm.py"))
    cases = [json.loads(s) for s in (args.out / "cases.jsonl").read_text().splitlines()]
    torch.set_num_threads(2)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = b["choose_device"]("auto")
    revision = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    baseline = []
    for c in cases:
        image = Image.open(args.dataset / c["image"]).convert("RGB")
        baseline.append({**c, "prediction": color_prediction(image, c["color"])})
    results = {"color_baseline": summarize(baseline)}
    (args.out / "baseline.jsonl").write_text("".join(json.dumps(r)+"\n" for r in baseline))
    with b["MemorySampler"](device) as memory:
        print(f"Loading cached SmolVLM2 on {device}, float16, 8 tokens, 2 threads", flush=True)
        processor = b["AutoProcessor"].from_pretrained(b["MODEL_ID"], revision=revision)
        model = b["AutoModelForImageTextToText"].from_pretrained(b["MODEL_ID"], revision=revision, dtype=torch.float16, low_cpu_mem_usage=True).to(device).eval()
        runner = b["Runner"](processor, model, device, 8)
        for name, template in PROMPTS.items():
            records = []
            bar = tqdm(cases, desc=name, unit="question")
            with (args.out / f"{name}.jsonl").open("w") as stream:
                for c in bar:
                    image = Image.open(args.dataset / c["image"]).convert("RGB")
                    prompt = template.format(**c)
                    raw, seconds = runner.infer(image, prompt)
                    normalized = raw.strip().lower().rstrip(".")
                    prediction = "center" if normalized == "middle" else normalized if normalized in DIRECTIONS else None
                    record = {**c, "prompt": prompt, "raw_response": raw, "prediction": prediction, "latency_s": seconds}
                    records.append(record)
                    stream.write(json.dumps(record)+"\n")
                    stream.flush()
                    bar.set_postfix(seconds=round(seconds, 2))
                    if prediction != c["expected"]:
                        draw = ImageDraw.Draw(image)
                        x0,y0,x1,y1 = c["bbox"]
                        draw.rectangle((x0,y0,x1-1,y1-1), outline="lime", width=2)
                        draw.text((2,2), f"{c['object_id']} true={c['expected']} pred={prediction}", fill="white", stroke_width=1, stroke_fill="black")
                        image.save(args.out / f"{name}-{c['sample_id']}-{c['object_id']}-failure.png")
            results[name] = summarize(records)
    summary = {"validation_only": True, "scenes": args.scenes, "model_revision": revision, "device": str(device), "dtype": "float16", "do_sample": False, "seed": 0, "max_new_tokens": 8, "results": results, "peak_mps_tensor_mib": memory.peak_mps/2**20, "peak_process_rss_mib": memory.peak_rss/2**20}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    runpy.run_path(str(ROOT / "scripts/plot-confusion.py"))["plot_confusion"](args.out / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/datasets/sample-v0")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/benchmarks/grounding-diagnostic")
    args = parser.parse_args()
    if args.scenes < 3 or args.scenes % 3:
        parser.error("--scenes must be a positive multiple of three")
    args.out.mkdir(parents=True, exist_ok=True)
    prepare(args) if args.prepare else infer(args)
