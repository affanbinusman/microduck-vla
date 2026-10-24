#!/usr/bin/env python3
"""Generate one labeled visual-grounding sample from Microduck's head camera."""

from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/dataset/sample_v0.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "artifacts/datasets/sample-v0/sample-000000",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    ids = [obj["id"] for obj in config["objects"]]
    if len(ids) != len(set(ids)):
        raise ValueError("object ids must be unique")
    if config["target_id"] not in ids:
        raise ValueError("target_id must name one of the objects")
    if config["robot"]["id"] in ids:
        raise ValueError("robot and object ids must be distinct")
    return config


def make_scenario(config: dict) -> Scenario:
    hx, hy = 1.5, 1.25
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    walls = [Wall(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    props = []
    for obj in config["objects"]:
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
        name="vla-sample-v0",
        seed=int(config["seed"]),
        floor=(3.5, 3.0),
        walls=walls,
        ducks=[Duck(robot["id"], tuple(robot["spawn"]))],
        props=props,
    )


def target_bbox(segmentation: np.ndarray, geom_id: int) -> tuple[list[int], np.ndarray]:
    # MuJoCo segmentation channels are [object id, object type].
    mask = (
        (segmentation[:, :, 0] == geom_id)
        & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    )
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise RuntimeError("target is not visible from the robot head camera")
    # xyxy uses an exclusive maximum, matching ordinary array slicing.
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return bbox, mask


def align_camera_to_site(model: mujoco.MjModel, name: str) -> int:
    """Make MuJoCo's -Z-looking camera match the site's x-forward frame."""
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if camera_id < 0 or site_id < 0:
        raise RuntimeError(f"missing camera or site {name!r}")
    if model.cam_bodyid[camera_id] != model.site_bodyid[site_id]:
        raise RuntimeError(f"camera and site {name!r} are mounted on different bodies")
    # Camera axes in site coordinates: right=-left, up=up, back=-forward.
    site_to_camera = np.array(
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    relative_quat = np.zeros(4, dtype=np.float64)
    camera_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(relative_quat, site_to_camera.ravel())
    mujoco.mju_mulQuat(camera_quat, model.site_quat[site_id], relative_quat)
    model.cam_pos[camera_id] = model.site_pos[site_id]
    model.cam_quat[camera_id] = camera_quat
    return camera_id


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    scenario = make_scenario(config)
    world = World(scenario, seed=int(config["seed"]))
    width, height = (int(v) for v in config["resolution"])
    camera_name = f"{config['robot']['id']}/head_camera"
    align_camera_to_site(world.model, camera_name)
    mujoco.mj_forward(world.model, world.data)
    target = next(obj for obj in config["objects"] if obj["id"] == config["target_id"])
    target_geom_name = f"{target['id']}_geom"
    target_geom_id = mujoco.mj_name2id(
        world.model, mujoco.mjtObj.mjOBJ_GEOM, target_geom_name
    )
    if target_geom_id < 0:
        raise RuntimeError(f"missing target geometry {target_geom_name!r}")

    args.out.mkdir(parents=True, exist_ok=True)
    renderer = offscreen_renderer(world.model, width, height)
    # An ego camera cannot see the shell enclosing it. Put this robot's geoms
    # on a temporary hidden group, as the browser head-camera inset does.
    robot = world.ducks[config["robot"]["id"]]
    robot_bodies = world.duck_bodies[robot.id]
    robot_geoms = (
        (world.model.geom_bodyid >= robot_bodies.start)
        & (world.model.geom_bodyid < robot_bodies.stop)
    )
    original_groups = world.model.geom_group[robot_geoms].copy()
    world.model.geom_group[robot_geoms] = 5
    render_options = mujoco.MjvOption()
    render_options.geomgroup[5] = 0
    renderer.update_scene(world.data, camera=camera_name, scene_option=render_options)
    rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(world.data, camera=camera_name, scene_option=render_options)
    segmentation = renderer.render().copy()
    renderer.close()
    world.model.geom_group[robot_geoms] = original_groups

    # Keep the raw input even if a configuration accidentally puts the target
    # outside the camera; that makes the first-scene iteration inspectable.
    imageio.imwrite(args.out / "rgb.png", rgb)
    bbox, mask = target_bbox(segmentation, target_geom_id)
    x0, y0, x1, y1 = bbox
    center = [round((x0 + x1) / 2, 2), round((y0 + y1) / 2, 2)]
    robot_x, robot_y, robot_yaw = (float(v) for v in config["robot"]["spawn"])
    target_x, target_y = (float(v) for v in target["position"][:2])
    dx, dy = target_x - robot_x, target_y - robot_y
    bearing = math.atan2(dy, dx) - robot_yaw
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))

    imageio.imwrite(args.out / "target_mask.png", mask.astype(np.uint8) * 255)
    preview = Image.fromarray(rgb)
    draw = ImageDraw.Draw(preview)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 255, 0), width=2)
    draw.text((4, 4), config["instruction"], fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    preview.save(args.out / "preview.png")

    annotation = {
        "dataset_version": config["dataset_version"],
        "sample_id": "sample-000000",
        "split": "development",
        "seed": int(config["seed"]),
        "image": "rgb.png",
        "width": width,
        "height": height,
        "camera": camera_name,
        "instruction": config["instruction"],
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
            for obj in config["objects"]
        ],
    }
    (args.out / "annotation.json").write_text(json.dumps(annotation, indent=2) + "\n")
    print(f"Generated {args.out}")
    print(f"Instruction: {config['instruction']}")
    print(f"Target bbox: {bbox}; visible pixels: {int(mask.sum())}")
    print(f"Review: {args.out / 'preview.png'}")


if __name__ == "__main__":
    main()
