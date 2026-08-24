#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Capture a path-aligned orbit of the Task 2 room (room USD only).

Loads robot_room.usd with no robot and no Task 2 objects. Flies a pinhole
camera with eval_camera intrinsics around the table XY at eye height,
looking along the path, then four 1 m circles (two inward, two outward)
at 1.00–1.75 m. Writes RGB PNGs + an MP4, then captures one
fisheyeSpherical equirectangular image over the table.

Runs inside the Isaac Sim 5.1.0 container with /isaac-sim/python.sh.
See scripts/run_room_capture.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASK2_ROOT = Path(__file__).resolve().parents[1]
_SCENES_DIR = _REPO_ROOT / "scripts" / "scenes"
if str(_SCENES_DIR) not in sys.path:
    sys.path.insert(0, str(_SCENES_DIR))

import scene_robot_room_keyboard as room_scene  # noqa: E402
from flythrough_path import (  # noqa: E402
    CameraSpec,
    FlythroughConfig,
    FlythroughFrame,
    PanoramaSpec,
    euler_xyz_to_quat,
    frames_from_config,
    is_baked_task2_table_bbox,
    load_flythrough_config,
    plot_orbit_trajectory,
)

DEFAULT_CONFIG = _TASK2_ROOT / "config" / "room_flythrough.yaml"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--room-usd",
        type=Path,
        default=room_scene.asset_path("robot_room.usd"),
        help="Room USD to reference (default: robot_room.usd).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Orbit YAML (camera + circle + panorama).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output folder. Default: task2_isaacsim/captures/"
        "room_flythrough_<timestamp>/",
    )
    parser.add_argument(
        "--n-poses",
        type=int,
        default=None,
        help="Override YAML orbit.n_poses.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override YAML video_fps for the orbit MP4.",
    )
    parser.add_argument(
        "--pano-res",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Override YAML panorama resolution.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=16,
        help="Kit update ticks before the first capture.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=8,
        help="Extra render frames after each camera pose change.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim without a visible Kit window.",
    )
    return parser


def _default_out_dir() -> Path:
    return _TASK2_ROOT / "captures" / f"room_flythrough_{_timestamp()}"


def _apply_cli_overrides(
    config: FlythroughConfig, args: argparse.Namespace
) -> FlythroughConfig:
    orbit = config.orbit
    circles = config.circles
    if args.n_poses is not None:
        n_poses = int(args.n_poses)
        if n_poses < 1:
            raise ValueError(f"--n-poses must be >= 1, got {n_poses}")
        orbit = replace(orbit, n_poses=n_poses)
        circles = tuple(replace(c, n_poses=n_poses) for c in circles)
    video_fps = (
        float(args.fps) if args.fps is not None else config.video_fps
    )
    if video_fps <= 0.0:
        raise ValueError(f"--fps must be positive, got {video_fps}")
    pano = config.panorama
    if args.pano_res is not None:
        width, height = (int(args.pano_res[0]), int(args.pano_res[1]))
        if width <= 0 or height <= 0:
            raise ValueError("--pano-res width/height must be positive")
        pano = replace(pano, width=width, height=height)
    return replace(
        config,
        orbit=orbit,
        circles=circles,
        panorama=pano,
        video_fps=video_fps,
    )


def _tick(simulation_app: Any, count: int) -> None:
    for _ in range(max(int(count), 0)):
        simulation_app.update()


def _set_camera_pose(
    stage: Any,
    prim_path: str,
    xyz: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Camera prim missing: {prim_path}")
    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(Gf.Vec3d(*xyz))
    xform_api.SetRotate(
        Gf.Vec3f(*rotation_xyz_deg),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )


def _author_camera_prim(
    stage: Any,
    prim_path: str,
    *,
    focal_length: float,
    focus_distance: float,
    projection: str,
    xyz: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
) -> None:
    from pxr import Sdf, UsdGeom

    parent = str(Path(prim_path).parent).replace("\\", "/")
    if parent and parent != "/":
        UsdGeom.Scope.Define(stage, parent)
    camera = UsdGeom.Camera.Define(stage, prim_path)
    camera.GetProjectionAttr().Set("perspective")
    if projection != "perspective":
        camera.GetPrim().CreateAttribute(
            "cameraProjectionType", Sdf.ValueTypeNames.Token
        ).Set(projection)
    camera.GetFocalLengthAttr().Set(float(focal_length))
    camera.GetFocusDistanceAttr().Set(float(focus_distance))
    camera.GetClippingRangeAttr().Set((0.05, 100.0))
    _set_camera_pose(stage, prim_path, xyz, rotation_xyz_deg)


def _annotator_rgb(annotator: Any) -> Any:
    import numpy as np

    data = annotator.get_data()
    if data is None:
        return None
    array = np.array(data)
    if array.size == 0:
        return None
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        max_value = float(array.max()) if array.size else 0.0
        if max_value <= 1.0:
            array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _attach_rgb_annotator(prim_path: str, width: int, height: int) -> Any:
    import omni.replicator.core as rep

    render_product = rep.create.render_product(prim_path, (width, height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)
    return annotator


def _capture_at_pose(
    simulation_app: Any,
    stage: Any,
    annotator: Any,
    prim_path: str,
    xyz: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
    warmup_frames: int,
) -> Any:
    _set_camera_pose(stage, prim_path, xyz, rotation_xyz_deg)
    _tick(simulation_app, warmup_frames)
    rgb = _annotator_rgb(annotator)
    if rgb is None:
        _tick(simulation_app, warmup_frames)
        rgb = _annotator_rgb(annotator)
    if rgb is None:
        raise RuntimeError(f"Empty RGB from camera {prim_path}")
    return rgb


def _save_png(path: Path, rgb: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
        return
    except ImportError:
        pass
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, rgb)
        return
    except ImportError as exc:
        raise RuntimeError(
            "Need Pillow or imageio to write PNG frames"
        ) from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _encode_png_sequence_mp4(
    png_dir: Path, out_path: Path, fps: float
) -> Path:
    """Encode numbered PNGs in png_dir to H.264 MP4 at out_path."""
    import shutil
    import subprocess

    pngs = sorted(png_dir.glob("*.png"))
    if not pngs:
        raise RuntimeError(f"No PNG frames in {png_dir}")
    fps = float(fps)
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        start_number = int(pngs[0].stem)
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-start_number",
            str(start_number),
            "-i",
            str(png_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and out_path.is_file():
            return out_path
        print(
            "Capture: ffmpeg libx264 failed "
            f"({result.stderr.strip() or result.returncode})",
            flush=True,
        )

    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p"
        )
        try:
            for png in pngs:
                writer.append_data(imageio.imread(png))
        finally:
            writer.close()
        if out_path.is_file():
            return out_path
    except Exception as exc:
        print(f"Capture: imageio MP4 failed ({exc})", flush=True)

    try:
        import cv2

        first = cv2.imread(str(pngs[0]))
        if first is None:
            raise RuntimeError(f"OpenCV could not read {pngs[0]}")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open MP4 writer")
        try:
            for png in pngs:
                frame = cv2.imread(str(png))
                if frame is None:
                    raise RuntimeError(f"OpenCV could not read {png}")
                writer.write(frame)
        finally:
            writer.release()
        if out_path.is_file():
            return out_path
    except Exception as exc:
        print(f"Capture: OpenCV MP4 failed ({exc})", flush=True)

    raise RuntimeError(
        "Could not encode orbit MP4 (need ffmpeg, imageio, or OpenCV)"
    )


def _frame_to_dict(frame: FlythroughFrame) -> dict[str, Any]:
    return {
        "index": frame.index,
        "track": frame.track,
        "look": frame.look,
        "xyz": list(frame.xyz),
        "quat_wxyz": list(frame.quat_wxyz),
        "yaw_deg": frame.yaw_deg,
        "pitch_deg": frame.pitch_deg,
        "rotation_xyz_deg": list(frame.rotation_xyz_deg),
    }


def _camera_payload(spec: CameraSpec, config: FlythroughConfig) -> dict:
    orbit = config.orbit
    return {
        "prim_path": spec.prim_path,
        "width": spec.width,
        "height": spec.height,
        "focal_length": spec.focal_length,
        "focus_distance": spec.focus_distance,
        "projection": spec.projection,
        "look": "along_path",
        "video_fps": config.video_fps,
        "orbit": {
            "center_xy": list(orbit.center_xy),
            "diameter_x_m": orbit.diameter_x_m,
            "diameter_y_m": orbit.diameter_y_m,
            "height_m": orbit.height_m,
            "n_poses": orbit.n_poses,
            "start_deg": orbit.start_deg,
            "wave_amp_m": orbit.wave_amp_m,
            "wave_cycles": orbit.wave_cycles,
        },
        "circles": [
            {
                "look": circle.look,
                "height_m": circle.height_m,
                "radius_m": circle.radius_m,
                "n_poses": circle.n_poses,
                "start_deg": circle.start_deg,
            }
            for circle in config.circles
        ],
    }


def _pano_payload(spec: PanoramaSpec) -> dict[str, Any]:
    return {
        "prim_path": spec.prim_path,
        "xyz": list(spec.translation),
        "quat_wxyz": list(euler_xyz_to_quat(spec.rotation_xyz_deg)),
        "rotation_xyz_deg": list(spec.rotation_xyz_deg),
        "width": spec.width,
        "height": spec.height,
        "projection": spec.projection,
    }


def _deactivate_prim(stage: Any, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsActive():
        return False
    prim.SetActive(False)
    print(f"Capture: hid {prim_path}", flush=True)
    return True


def _hide_baked_task2_table(stage: Any) -> list[str]:
    """Hide the cubicle table baked into robot_room.usd at Task 2 XY.

    table_edit.usd is not referenced in room-only capture, but robot_room.usd
    still contains that desk (prim name ___008 in the current asset).
    """
    from pxr import Usd, UsdGeom

    hidden: list[str] = []
    extra_table = "/World/Scene/Table"
    if _deactivate_prim(stage, extra_table):
        hidden.append(extra_table)

    room = stage.GetPrimAtPath("/World/Environment/RobotRoom")
    if not room.IsValid():
        return hidden

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    matches: list[str] = []
    for prim in Usd.PrimRange(room):
        if not prim.IsA(UsdGeom.Xformable):
            continue
        bound = cache.ComputeWorldBound(prim)
        aligned = bound.ComputeAlignedRange()
        if aligned.IsEmpty():
            continue
        minimum = aligned.GetMin()
        maximum = aligned.GetMax()
        mid_xy = (
            0.5 * (minimum[0] + maximum[0]),
            0.5 * (minimum[1] + maximum[1]),
        )
        size_xyz = (
            maximum[0] - minimum[0],
            maximum[1] - minimum[1],
            maximum[2] - minimum[2],
        )
        if is_baked_task2_table_bbox(mid_xy, size_xyz):
            matches.append(str(prim.GetPath()))

    # Deactivate the shallowest match so child meshes go with the group.
    matches.sort(key=lambda path: path.count("/"))
    kept: list[str] = []
    for path in matches:
        if any(path.startswith(parent + "/") for parent in kept):
            continue
        if _deactivate_prim(stage, path):
            kept.append(path)
            hidden.append(path)
    if not hidden:
        print(
            "Capture: warning: no baked Task 2 table prim found to hide",
            flush=True,
        )
    return hidden


def _build_room_only_stage(app: Any, room_path: Path) -> Any:
    import omni.usd

    context = omni.usd.get_context()
    context.new_stage()
    for _ in range(10):
        app.update()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Could not create an Isaac Sim stage")
    # task1 + no robot: room USD only (no Task 2 objects / eval camera).
    stage = room_scene.configure_robot_room_stage(
        app,
        stage,
        room_path=room_path,
        task="task1",
        head_placement="random",
        robot_path=None,
    )
    _hide_baked_task2_table(stage)
    return stage


def _run(simulation_app: Any, args: argparse.Namespace) -> Path:
    import omni.kit.app
    import omni.timeline
    import omni.usd

    config = _apply_cli_overrides(load_flythrough_config(args.config), args)
    frames = frames_from_config(config)
    if not frames:
        raise RuntimeError("Orbit produced no frames")

    room_path = Path(args.room_usd).expanduser()
    if not room_path.is_file():
        raise FileNotFoundError(f"Room USD not found: {room_path}")

    app = omni.kit.app.get_app()
    _build_room_only_stage(app, room_path)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get the Isaac Sim stage")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print("Capture: warming renderer", flush=True)
    _tick(simulation_app, args.settle_steps)
    print("Capture: renderer ready", flush=True)

    out_dir = Path(args.out) if args.out is not None else _default_out_dir()
    rgb_dir = out_dir / "rgb"
    pano_dir = out_dir / "panorama"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        rgb_dir.mkdir(parents=True, exist_ok=True)
        pano_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PermissionError(
            f"Cannot write capture output to {out_dir} ({exc}). "
            "Use run_room_capture.sh (it creates a world-writable "
            "task2_isaacsim/captures/) or pass --out to a directory "
            "the Isaac Sim container user can write."
        ) from exc

    first = frames[0]
    _author_camera_prim(
        stage,
        config.camera.prim_path,
        focal_length=config.camera.focal_length,
        focus_distance=config.camera.focus_distance,
        projection=config.camera.projection,
        xyz=first.xyz,
        rotation_xyz_deg=first.rotation_xyz_deg,
    )
    fly_annot = _attach_rgb_annotator(
        config.camera.prim_path,
        config.camera.width,
        config.camera.height,
    )
    _tick(simulation_app, args.warmup_frames)

    print(f"Capture: {len(frames)} frames -> {rgb_dir}", flush=True)
    for frame in frames:
        rgb = _capture_at_pose(
            simulation_app,
            stage,
            fly_annot,
            config.camera.prim_path,
            frame.xyz,
            frame.rotation_xyz_deg,
            args.warmup_frames,
        )
        _save_png(rgb_dir / f"{frame.index:06d}.png", rgb)
        if frame.index % 10 == 0 or frame.index + 1 == len(frames):
            print(
                f"  frame {frame.index + 1}/{len(frames)}",
                flush=True,
            )

    mp4_path = out_dir / "orbit.mp4"
    plot_path = out_dir / "orbit_trajectory.png"
    print(
        f"Capture: encoding {len(frames)} frames -> {mp4_path} "
        f"({config.video_fps:g} fps)",
        flush=True,
    )
    _encode_png_sequence_mp4(rgb_dir, mp4_path, config.video_fps)
    try:
        plot_orbit_trajectory(frames, plot_path, config.orbit.center_xy)
    except RuntimeError as exc:
        print(f"Capture: skipped trajectory plot ({exc})", flush=True)

    _author_camera_prim(
        stage,
        config.panorama.prim_path,
        focal_length=config.camera.focal_length,
        focus_distance=config.camera.focus_distance,
        projection=config.panorama.projection,
        xyz=config.panorama.translation,
        rotation_xyz_deg=config.panorama.rotation_xyz_deg,
    )
    pano_annot = _attach_rgb_annotator(
        config.panorama.prim_path,
        config.panorama.width,
        config.panorama.height,
    )
    print(
        f"Capture: panorama {config.panorama.width}x"
        f"{config.panorama.height} {config.panorama.projection}",
        flush=True,
    )
    pano_rgb = _capture_at_pose(
        simulation_app,
        stage,
        pano_annot,
        config.panorama.prim_path,
        config.panorama.translation,
        config.panorama.rotation_xyz_deg,
        max(args.warmup_frames, 16),
    )
    _save_png(pano_dir / "equirect.png", pano_rgb)

    _write_json(
        out_dir / "poses.json",
        {"frames": [_frame_to_dict(frame) for frame in frames]},
    )
    _write_json(
        out_dir / "camera.json",
        _camera_payload(config.camera, config),
    )
    _write_json(
        out_dir / "panorama_pose.json",
        _pano_payload(config.panorama),
    )
    print(f"Capture: wrote {out_dir}", flush=True)
    return out_dir


def main() -> None:
    args = _build_arg_parser().parse_args()
    import traceback

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        launch_config={
            "headless": bool(args.headless),
            "width": 1280,
            "height": 720,
        }
    )
    try:
        _run(simulation_app, args)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
