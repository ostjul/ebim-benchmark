#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pack a room flythrough capture into a LichtFeld / NeRF dataset.

Writes ``images/`` (symlinks to ``rgb/*.png`` by default) and a
``transforms.json`` whose camera-to-world matrices match the USD poses
already stored in ``poses.json``. Skips the equirectangular panorama.

USD cameras look along local -Z with +Y up, which is the same OpenGL /
Blender convention LichtFeld's transforms loader expects. Pixel focal
length uses the authored millimetre focal length and the USD default
horizontal aperture (the capture does not set an aperture).

Example::

    python3 task2_isaacsim/scripts/pack_lichtfeld_dataset.py \\
      --capture task2_isaacsim/captures/room_flythrough_20260824T061256Z
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from flythrough_path import rotate_vector_quat  # noqa: E402

# UsdGeom.Camera default film back (mm). Capture never overrides it.
USD_DEFAULT_HORIZONTAL_APERTURE_MM = 20.955
RGB_DIRNAME = "rgb"
IMAGES_DIRNAME = "images"
DEFAULT_DATASET_DIRNAME = "lfs_dataset"


def pixel_focal_length(
    focal_length_mm: float,
    width_px: int,
    horizontal_aperture_mm: float = USD_DEFAULT_HORIZONTAL_APERTURE_MM,
) -> float:
    """Square-pixel fx from USD focal length, fit to the horizontal film."""
    if focal_length_mm <= 0.0:
        raise ValueError(f"focal_length_mm must be > 0, got {focal_length_mm}")
    if width_px < 1:
        raise ValueError(f"width_px must be >= 1, got {width_px}")
    if horizontal_aperture_mm <= 0.0:
        raise ValueError(
            "horizontal_aperture_mm must be > 0, "
            f"got {horizontal_aperture_mm}"
        )
    return (focal_length_mm / horizontal_aperture_mm) * float(width_px)


def camera_to_world_matrix(
    xyz: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> list[list[float]]:
    """4x4 OpenGL/USD camera-to-world from translation + wxyz quaternion."""
    x_axis = rotate_vector_quat(quat_wxyz, (1.0, 0.0, 0.0))
    y_axis = rotate_vector_quat(quat_wxyz, (0.0, 1.0, 0.0))
    z_axis = rotate_vector_quat(quat_wxyz, (0.0, 0.0, 1.0))
    return [
        [x_axis[0], y_axis[0], z_axis[0], xyz[0]],
        [x_axis[1], y_axis[1], z_axis[1], xyz[1]],
        [x_axis[2], y_axis[2], z_axis[2], xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _as_xyz(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"expected xyz list of 3 floats, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _as_quat_wxyz(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"expected quat_wxyz list of 4 floats, got {value!r}")
    quat = (
        float(value[0]),
        float(value[1]),
        float(value[2]),
        float(value[3]),
    )
    length = math.sqrt(sum(c * c for c in quat))
    if length <= 0.0:
        raise ValueError("quat_wxyz has zero length")
    return tuple(c / length for c in quat)  # type: ignore[return-value]


def _png_name(index: int) -> str:
    return f"{int(index):06d}.png"


def load_capture(
    capture_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    poses_path = capture_dir / "poses.json"
    camera_path = capture_dir / "camera.json"
    if not poses_path.is_file():
        raise FileNotFoundError(f"poses.json not found: {poses_path}")
    if not camera_path.is_file():
        raise FileNotFoundError(f"camera.json not found: {camera_path}")
    poses_payload = json.loads(poses_path.read_text(encoding="utf-8"))
    camera_payload = json.loads(camera_path.read_text(encoding="utf-8"))
    frames = poses_payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{poses_path} has no frames")
    return camera_payload, frames


def require_rgb_frames(
    capture_dir: Path, frames: list[dict[str, Any]]
) -> list[Path]:
    rgb_dir = capture_dir / RGB_DIRNAME
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"rgb folder not found: {rgb_dir}")
    paths: list[Path] = []
    missing: list[str] = []
    for frame in frames:
        name = _png_name(int(frame["index"]))
        path = rgb_dir / name
        if not path.is_file():
            missing.append(name)
        else:
            paths.append(path)
    if missing:
        preview = ", ".join(missing[:8])
        extra = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
        raise FileNotFoundError(
            f"{len(missing)} RGB frames missing under {rgb_dir}: "
            f"{preview}{extra}"
        )
    return paths


def build_transforms(
    camera: dict[str, Any],
    frames: list[dict[str, Any]],
    *,
    horizontal_aperture_mm: float = USD_DEFAULT_HORIZONTAL_APERTURE_MM,
) -> dict[str, Any]:
    width = int(camera["width"])
    height = int(camera["height"])
    if width < 1 or height < 1:
        raise ValueError(f"invalid camera size {width}x{height}")
    fl_x = pixel_focal_length(
        float(camera["focal_length"]),
        width,
        horizontal_aperture_mm,
    )
    out_frames: list[dict[str, Any]] = []
    for frame in frames:
        index = int(frame["index"])
        xyz = _as_xyz(frame["xyz"])
        quat = _as_quat_wxyz(frame["quat_wxyz"])
        out_frames.append(
            {
                "file_path": f"{IMAGES_DIRNAME}/{_png_name(index)}",
                "transform_matrix": camera_to_world_matrix(xyz, quat),
                "track": frame.get("track"),
                "look": frame.get("look"),
            }
        )
    return {
        "camera_model": "PINHOLE",
        "w": width,
        "h": height,
        "fl_x": fl_x,
        "fl_y": fl_x,
        "cx": 0.5 * width,
        "cy": 0.5 * height,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "horizontal_aperture_mm": horizontal_aperture_mm,
        "focal_length_mm": float(camera["focal_length"]),
        "frames": out_frames,
    }


def _link_or_copy(
    source: Path,
    dest: Path,
    *,
    copy_images: bool,
    overwrite: bool,
) -> None:
    if dest.exists() or dest.is_symlink():
        if not overwrite:
            return
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(source, dest)
        return
    dest.symlink_to(Path(os_relpath(source, dest.parent)))


def os_relpath(target: Path, start: Path) -> str:
    return os.path.relpath(target, start)


def pack_lichtfeld_dataset(
    capture_dir: Path,
    out_dir: Path | None = None,
    *,
    copy_images: bool = False,
    horizontal_aperture_mm: float = USD_DEFAULT_HORIZONTAL_APERTURE_MM,
    overwrite: bool = False,
) -> Path:
    capture_dir = capture_dir.expanduser().resolve()
    camera, frames = load_capture(capture_dir)
    rgb_paths = require_rgb_frames(capture_dir, frames)
    dataset_dir = (
        out_dir.expanduser().resolve()
        if out_dir is not None
        else capture_dir / DEFAULT_DATASET_DIRNAME
    )
    images_dir = dataset_dir / IMAGES_DIRNAME
    transforms_path = dataset_dir / "transforms.json"
    if transforms_path.exists() and not overwrite:
        raise FileExistsError(
            f"{transforms_path} already exists (pass --overwrite)"
        )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    for source in rgb_paths:
        _link_or_copy(
            source,
            images_dir / source.name,
            copy_images=copy_images,
            overwrite=overwrite,
        )
    payload = build_transforms(
        camera,
        frames,
        horizontal_aperture_mm=horizontal_aperture_mm,
    )
    transforms_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="Flythrough capture folder (poses.json, camera.json, rgb/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Dataset folder. Default: <capture>/lfs_dataset.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy PNGs into images/ instead of relative symlinks.",
    )
    parser.add_argument(
        "--horizontal-aperture",
        type=float,
        default=USD_DEFAULT_HORIZONTAL_APERTURE_MM,
        help=(
            "USD horizontal aperture in mm "
            f"(default: {USD_DEFAULT_HORIZONTAL_APERTURE_MM:g})."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing dataset folder's images and transforms.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    dataset_dir = pack_lichtfeld_dataset(
        args.capture,
        args.out,
        copy_images=bool(args.copy),
        horizontal_aperture_mm=float(args.horizontal_aperture),
        overwrite=bool(args.overwrite),
    )
    transforms_path = dataset_dir / "transforms.json"
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    print(
        f"Packed {len(payload['frames'])} frames -> {dataset_dir}",
        flush=True,
    )
    print(
        f"  fl_x=fl_y={payload['fl_x']:.4f}  "
        f"{payload['w']}x{payload['h']} PINHOLE",
        flush=True,
    )


if __name__ == "__main__":
    main()
