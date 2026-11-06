#!/usr/bin/env python3
"""Benchmark a pretrained VLM on Microduck visual grounding."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
PROMPTS = {
    "coordinate-json-v1": (
        "You control a robot using this camera image. {instruction} "
        "Locate the center of the requested object. Return only JSON in exactly this schema: "
        '{{"direction":"left|center|right","x":0,"y":0}}. '
        "Use one direction value. x and y are integer percentages from 0 to 100, "
        "with x measured left-to-right and y measured top-to-bottom."
    ),
    "direction-word-v2": (
        "Inspect the image carefully for the object named by this instruction: {instruction} "
        "Ignore the other objects. Is the center of the requested object in the left, center, "
        "or right third of the image? Answer with exactly one lowercase word: left, center, or right."
    ),
    "direction-grid-v3": (
        "This robot camera image is divided into three equal vertical regions: left, center, and right. "
        "{instruction} Find that specific colored shape and decide which region contains its center. "
        "Respond with only one token from this list: left, center, right."
    ),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "artifacts/datasets/sample-v0")
    parser.add_argument("--out", type=Path, default=root / "artifacts/benchmarks/smolvlm2-500m")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "mps" or (requested == "auto" and torch.backends.mps.is_available()):
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


class MemorySampler:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.process = psutil.Process()
        self.peak_rss = 0
        self.peak_mps = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        if self.device.type == "mps":
            self.peak_mps = max(self.peak_mps, int(torch.mps.current_allocated_memory()))

    def _run(self) -> None:
        while not self._stop.wait(0.01):
            self._sample()

    def __enter__(self) -> "MemorySampler":
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


def read_dataset(root: Path) -> list[dict[str, Any]]:
    path = root / "annotations.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no samples in {path}")
    return rows


def extract_prediction(raw: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if not match:
        direction = raw.strip().lower().rstrip(".")
        return {"direction": direction} if direction in {"left", "center", "right"} else None
    try:
        value = json.loads(match.group(0))
        direction = str(value["direction"]).strip().lower()
        x, y = float(value["x"]), float(value["y"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if direction not in {"left", "center", "right"} or not (0 <= x <= 100 and 0 <= y <= 100):
        return None
    return {"direction": direction, "x": x, "y": y}


def true_direction(center_x: float, width: int) -> str:
    fraction = center_x / width
    return "left" if fraction < 1 / 3 else "right" if fraction > 2 / 3 else "center"


def score_prediction(dataset: Path, row: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
    target = row["target"]
    expected_direction = true_direction(float(target["center_xy"][0]), int(row["width"]))
    score: dict[str, Any] = {
        "valid": prediction is not None,
        "expected_direction": expected_direction,
        "direction_correct": False,
        "point_inside_mask": None,
        "center_error_normalized": None,
    }
    if prediction is None:
        return score
    score["direction_correct"] = prediction["direction"] == expected_direction
    if "x" not in prediction or "y" not in prediction:
        return score
    width, height = int(row["width"]), int(row["height"])
    px = min(width - 1, max(0, round(prediction["x"] * (width - 1) / 100)))
    py = min(height - 1, max(0, round(prediction["y"] * (height - 1) / 100)))
    mask = np.asarray(Image.open(dataset / row["target_mask"]).convert("L"))
    true_x = float(target["center_xy"][0]) * 100 / width
    true_y = float(target["center_xy"][1]) * 100 / height
    score.update(
        {
            "point_inside_mask": bool(mask[py, px] > 0),
            "center_error_normalized": round(
                math.hypot(prediction["x"] - true_x, prediction["y"] - true_y) / math.sqrt(2), 6
            ),
            "predicted_pixel_xy": [px, py],
        }
    )
    return score


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    errors = [record["score"]["center_error_normalized"] for record in records]
    errors = [error for error in errors if error is not None]
    latencies = [record["latency_s"] for record in records]
    point_scores = [record["score"]["point_inside_mask"] for record in records]
    point_scores = [score for score in point_scores if score is not None]
    return {
        "samples": total,
        "valid_rate": sum(record["score"]["valid"] for record in records) / total,
        "direction_accuracy": sum(record["score"]["direction_correct"] for record in records) / total,
        "coordinate_rate": len(point_scores) / total,
        "point_in_mask_accuracy": None if not point_scores else sum(point_scores) / len(point_scores),
        "mean_center_error_normalized": None if not errors else round(float(np.mean(errors)), 6),
        "mean_latency_s": round(float(np.mean(latencies)), 6),
        "p95_latency_s": round(float(np.percentile(latencies, 95)), 6),
    }


def selection_key(metrics: dict[str, Any], prompt_id: str) -> tuple[Any, ...]:
    error = metrics["mean_center_error_normalized"]
    point_accuracy = metrics["point_in_mask_accuracy"]
    return (
        metrics["direction_accuracy"],
        metrics["valid_rate"],
        point_accuracy if point_accuracy is not None else -1.0,
        -(error if error is not None else float("inf")),
        -list(PROMPTS).index(prompt_id),
    )


@dataclass
class Runner:
    processor: Any
    model: Any
    device: torch.device
    max_new_tokens: int

    def infer(self, image: Image.Image, prompt: str) -> tuple[str, float]:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=chat, images=[image], return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        sync(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        sync(self.device)
        latency = time.perf_counter() - started
        prompt_tokens = inputs["input_ids"].shape[-1]
        raw = self.processor.decode(generated[0, prompt_tokens:], skip_special_tokens=True).strip()
        return raw, latency


def run_one(
    runner: Runner,
    dataset: Path,
    row: dict[str, Any],
    prompt_id: str,
) -> dict[str, Any]:
    prompt = PROMPTS[prompt_id].format(instruction=row["instruction"])
    image = Image.open(dataset / row["image"]).convert("RGB")
    raw, latency = runner.infer(image, prompt)
    prediction = extract_prediction(raw)
    return {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "prompt_id": prompt_id,
        "instruction": row["instruction"],
        "raw_response": raw,
        "prediction": prediction,
        "score": score_prediction(dataset, row, prediction),
        "latency_s": round(latency, 6),
    }


def draw_eval_overlay(dataset: Path, out: Path, row: dict[str, Any], record: dict[str, Any]) -> None:
    image = Image.open(dataset / row["image"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = row["target"]["bbox_xyxy"]
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 255, 0), width=2)
    prediction = record["prediction"]
    if prediction is not None and "predicted_pixel_xy" in record["score"]:
        px, py = record["score"]["predicted_pixel_xy"]
        draw.line((px - 5, py, px + 5, py), fill=(255, 0, 0), width=2)
        draw.line((px, py - 5, px, py + 5), fill=(255, 0, 0), width=2)
    predicted = "invalid" if prediction is None else prediction["direction"]
    expected = record["score"]["expected_direction"]
    draw.text(
        (4, 4),
        f"pred={predicted} true={expected} | {row['instruction']}",
        fill="white",
        stroke_width=2,
        stroke_fill="black",
    )
    image.save(out)


def main() -> None:
    args = parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = choose_device(args.device)
    rows = read_dataset(args.dataset)
    validation = [row for row in rows if row["split"] == "val"]
    evaluation = [row for row in rows if row["split"] == "eval"]
    if not validation or not evaluation:
        raise ValueError("dataset must contain at least one validation and one evaluation sample")
    args.out.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    with MemorySampler(device) as memory:
        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        runner = Runner(processor, model, device, args.max_new_tokens)
        load_seconds = time.perf_counter() - load_started

        # Warm-up is excluded from recorded latency and uses validation data only.
        warm_prompt = PROMPTS[next(iter(PROMPTS))].format(instruction=validation[0]["instruction"])
        runner.infer(Image.open(args.dataset / validation[0]["image"]).convert("RGB"), warm_prompt)

        validation_records: list[dict[str, Any]] = []
        prompt_metrics: dict[str, dict[str, Any]] = {}
        for prompt_id in PROMPTS:
            records = [run_one(runner, args.dataset, row, prompt_id) for row in validation]
            validation_records.extend(records)
            prompt_metrics[prompt_id] = aggregate(records)
        selected = max(PROMPTS, key=lambda prompt_id: selection_key(prompt_metrics[prompt_id], prompt_id))

        # The prompt is frozen here. Evaluation data did not participate above.
        evaluation_records = [run_one(runner, args.dataset, row, selected) for row in evaluation]

    revision = getattr(model.config, "_commit_hash", None)
    selection = {
        "candidates": PROMPTS,
        "validation_metrics": prompt_metrics,
        "selection_order": [
            "direction_accuracy",
            "valid_rate",
            "point_in_mask_accuracy_when_coordinates_exist",
            "lowest_center_error",
            "prompt_declaration_order",
        ],
        "selected_prompt_id": selected,
        "selected_prompt": PROMPTS[selected],
        "note": "Prompt selection used validation samples only; evaluation ran after selection.",
    }
    summary = {
        "model": args.model,
        "model_revision": revision,
        "device": str(device),
        "dtype": "float32",
        "temperature": 0,
        "do_sample": False,
        "seed": 0,
        "dataset": str(args.dataset),
        "dataset_samples": len(rows),
        "validation_samples": len(validation),
        "evaluation_samples": len(evaluation),
        "selected_prompt_id": selected,
        "evaluation_metrics": aggregate(evaluation_records),
        "model_load_seconds": round(load_seconds, 6),
        "peak_process_rss_mib": round(memory.peak_rss / 2**20, 2),
        "peak_mps_allocated_mib": round(memory.peak_mps / 2**20, 2),
        "system": {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__},
        "limitations": [
            "This ten-sample run has only two validation and one evaluation example.",
            "It validates the benchmark pipeline and is not a statistically meaningful model result.",
        ],
    }
    (args.out / "validation_runs.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in validation_records)
    )
    (args.out / "eval_predictions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in evaluation_records)
    )
    (args.out / "prompt_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row, record in zip(evaluation, evaluation_records, strict=True):
        draw_eval_overlay(args.dataset, args.out / f"{row['sample_id']}_overlay.png", row, record)

    print(json.dumps(summary, indent=2))
    print(f"Raw and parsed results: {args.out}")


if __name__ == "__main__":
    main()
