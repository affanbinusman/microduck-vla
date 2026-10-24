#!/usr/bin/env python3
"""Generate a tiny labeled visual-grounding dataset from Microduck's head camera."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from microduck_local.render_rollout import offscreen_renderer
from microduck_local.world import World
from microduck_local.world.scenario import Duck, Prop, Scenario, Wall


SHAPES = {"cube": "box", "sphere": "sphere", "cylinder": "cylinder"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/dataset/sample_v0.json")
    parser.add_argument("--out", type=Path, default=root / "artifacts/datasets/sample-v0")
    parser.add_argument("--samples", type=int, default=10, help="number of samples to generate")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    ids = [obj["id"] for obj in config["objects"]]
    if len(ids) != len(set(ids)):
        raise ValueError("object ids must be unique")
    if config["robot"]["id"] in ids:
        raise ValueError("robot and object ids must be distinct")
    if len(config["scene_positions"]) != len(config["objects"]):
        raise ValueError("scene_positions must contain one position per object")
    percentages = config["split_percentages"]
    if set(percentages) != {"train", "val", "eval"}:
        raise ValueError("split_percentages must define train, val, and eval")
    if any(value < 0 for value in percentages.values()):
        raise ValueError("split percentages cannot be negative")
    if not math.isclose(sum(percentages.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split percentages must add up to 1.0")
    return config


def allocate_splits(config: dict, total: int) -> dict[str, int]:
    if total < 1:
        raise ValueError("--samples must be at least 1")
    names = ("train", "val", "eval")
    percentages = config["split_percentages"]
    raw = {name: total * float(percentages[name]) for name in names}
    counts = {name: math.floor(raw[name]) for name in names}
    remaining = total - sum(counts.values())
    require_all = total >= sum(percentages[name] > 0 for name in names)
    while remaining:
        empty = [name for name in names if require_all and percentages[name] > 0 and counts[name] == 0]
        if empty:
            chosen = min(empty, key=lambda name: percentages[name])
        else:
            chosen = max(names, key=lambda name: (raw[name] - counts[name], percentages[name]))
        counts[chosen] += 1
        remaining -= 1
    if require_all:
        for name in names:
            if percentages[name] > 0 and counts[name] == 0:
                donor = max(names, key=lambda candidate: counts[candidate])
                counts[donor] -= 1
                counts[name] = 1
    return counts


def split_sequence(config: dict, total: int) -> list[str]:
    counts = allocate_splits(config, total)
    return [
        split
        for split in ("train", "val", "eval")
        for _ in range(counts[split])
    ]


def sample_scene(config: dict, index: int) -> tuple[int, list[dict], dict, str]:
    seed = int(config["seed"]) + index
    rng = np.random.default_rng(seed)
    positions = [list(position) for position in config["scene_positions"]]
    rng.shuffle(positions)
    objects: list[dict] = []
    for template, position in zip(config["objects"], positions, strict=True):
        obj = dict(template)
        obj["position"] = [
            round(float(position[0]) + float(rng.uniform(-0.035, 0.035)), 6),
            round(float(position[1]) + float(rng.uniform(-0.025, 0.025)), 6),
            float(position[2]),
        ]
        objects.append(obj)
    target = objects[index % len(objects)]
    instruction = config["instruction_template"].format(**target)
    return seed, objects, target, instruction


def make_scenario(config: dict, seed: int, objects: list[dict]) -> Scenario:
    hx, hy = 1.5, 1.25
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    walls = [
        Wall(corners[i], corners[(i + 1) % 4], height=float(config["wall_height"]))
        for i in range(4)
    ]
    props = []
    for obj in objects:
        shape = SHAPES.get(obj["shape"])
        if shape is None:
            raise ValueError(f"unsupported shape {obj['shape']!r}")
        props.append(
            Prop(
                id=obj["id"],
                pos=tuple(obj["position"]),
                shape=shape,
                size=tuple(obj["size"]),
                mass=0.0,
                rgba=tuple(obj["rgba"]),
                cls="toy",
            )
        )
    robot = config["robot"]
    return Scenario(
        name=f"vla-sample-v0-{seed}",
        seed=seed,
        floor=(3.5, 3.0),
        walls=walls,
        ducks=[Duck(robot["id"], tuple(robot["spawn"]))],
        props=props,
    )


def target_bbox(segmentation: np.ndarray, geom_id: int) -> tuple[list[int], np.ndarray]:
    mask = (
        (segmentation[:, :, 0] == geom_id)
        & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    )
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise RuntimeError("target is not visible from the robot head camera")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], mask


def align_camera_to_site(model: mujoco.MjModel, name: str) -> None:
    """Make MuJoCo's -Z-looking camera match the site's x-forward frame."""
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if camera_id < 0 or site_id < 0:
        raise RuntimeError(f"missing camera or site {name!r}")
    if model.cam_bodyid[camera_id] != model.site_bodyid[site_id]:
        raise RuntimeError(f"camera and site {name!r} are mounted on different bodies")
    site_to_camera = np.array(
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    relative_quat = np.zeros(4, dtype=np.float64)
    camera_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(relative_quat, site_to_camera.ravel())
    mujoco.mju_mulQuat(camera_quat, model.site_quat[site_id], relative_quat)
    model.cam_pos[camera_id] = model.site_pos[site_id]
    model.cam_quat[camera_id] = camera_quat


def render_sample(config: dict, root: Path, split: str, index: int) -> tuple[dict, Image.Image]:
    seed, objects, target, instruction = sample_scene(config, index)
    world = World(make_scenario(config, seed, objects), seed=seed)
    width, height = (int(value) for value in config["resolution"])
    camera_name = f"{config['robot']['id']}/head_camera"
    align_camera_to_site(world.model, camera_name)
    mujoco.mj_forward(world.model, world.data)
    target_geom_id = mujoco.mj_name2id(
        world.model, mujoco.mjtObj.mjOBJ_GEOM, f"{target['id']}_geom"
    )
    if target_geom_id < 0:
        raise RuntimeError(f"missing target geometry {target['id']!r}")

    sample_id = f"sample-{index:06d}"
    relative_dir = Path(split) / sample_id
    sample_dir = root / relative_dir
    sample_dir.mkdir(parents=True, exist_ok=True)
    renderer = offscreen_renderer(world.model, width, height)
    robot_bodies = world.duck_bodies[config["robot"]["id"]]
    robot_geoms = (
        (world.model.geom_bodyid >= robot_bodies.start)
        & (world.model.geom_bodyid < robot_bodies.stop)
    )
    original_groups = world.model.geom_group[robot_geoms].copy()
    world.model.geom_group[robot_geoms] = 5
    options = mujoco.MjvOption()
    options.geomgroup[5] = 0
    try:
        renderer.update_scene(world.data, camera=camera_name, scene_option=options)
        rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(world.data, camera=camera_name, scene_option=options)
        segmentation = renderer.render().copy()
    finally:
        renderer.close()
        world.model.geom_group[robot_geoms] = original_groups

    imageio.imwrite(sample_dir / "rgb.png", rgb)
    bbox, mask = target_bbox(segmentation, target_geom_id)
    x0, y0, x1, y1 = bbox
    center = [round((x0 + x1) / 2, 2), round((y0 + y1) / 2, 2)]
    robot_x, robot_y, robot_yaw = (float(value) for value in config["robot"]["spawn"])
    target_x, target_y = (float(value) for value in target["position"][:2])
    dx, dy = target_x - robot_x, target_y - robot_y
    bearing = math.atan2(dy, dx) - robot_yaw
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))

    imageio.imwrite(sample_dir / "target_mask.png", mask.astype(np.uint8) * 255)
    preview = Image.fromarray(rgb)
    draw = ImageDraw.Draw(preview)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 255, 0), width=2)
    draw.text((4, 4), instruction, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    preview.save(sample_dir / "preview.png")

    annotation = {
        "dataset_version": config["dataset_version"],
        "sample_id": sample_id,
        "split": split,
        "seed": seed,
        "image": str(relative_dir / "rgb.png"),
        "target_mask": str(relative_dir / "target_mask.png"),
        "width": width,
        "height": height,
        "camera": camera_name,
        "instruction": instruction,
        "target": {
            "id": target["id"],
            "color": target["color"],
            "shape": target["shape"],
            "bbox_xyxy": bbox,
            "center_xy": center,
            "visible_pixels": int(mask.sum()),
            "bearing_rad": round(bearing, 6),
            "distance_m": round(math.hypot(dx, dy), 6),
        },
        "objects": [
            {key: obj[key] for key in ("id", "color", "shape", "position")}
            for obj in objects
        ],
    }
    (sample_dir / "annotation.json").write_text(json.dumps(annotation, indent=2) + "\n")
    return annotation, preview


def write_contact_sheet(previews: list[Image.Image], out: Path) -> None:
    width, height = previews[0].size
    columns = min(5, len(previews))
    rows = math.ceil(len(previews) / columns)
    sheet = Image.new("RGB", (width * columns, height * rows), color=(0, 0, 0))
    for index, preview in enumerate(previews):
        sheet.paste(preview, ((index % columns) * width, (index // columns) * height))
    sheet.save(out)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    counts = allocate_splits(config, args.samples)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "eval"):
        shutil.rmtree(args.out / split, ignore_errors=True)
    annotations: list[dict] = []
    previews: list[Image.Image] = []
    for index, split in enumerate(split_sequence(config, args.samples)):
        annotation, preview = render_sample(config, args.out, split, index)
        annotations.append(annotation)
        previews.append(preview)
        print(
            f"{annotation['sample_id']} {split}: {annotation['instruction']} "
            f"bbox={annotation['target']['bbox_xyxy']}"
        )
    (args.out / "annotations.jsonl").write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in annotations)
    )
    write_contact_sheet(previews, args.out / "contact_sheet.png")
    print(f"Generated {len(annotations)} samples in {args.out}")
    print("Splits: " + ", ".join(f"{name}={counts[name]}" for name in ("train", "val", "eval")))
    print(f"Review: {args.out / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
