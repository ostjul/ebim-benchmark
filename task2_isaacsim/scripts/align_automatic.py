#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Multi-scale similarity ICP (7-DoF: scale, rotation, translation).

Aligns a reconstructed GLB collider onto ``assets/robot_room.usd``, writes
before/after point-cloud views, then prints a pasteable
``run_isaacsim_teleop.sh`` command that loads the NuRec ``.usdz`` volume
with that GLB under the estimated joint xform.

Requires Open3D. USD/USDZ also need pxr (Isaac Sim ``python.sh`` or
``usd-core``).

Example::

    python3 task2_isaacsim/scripts/align_automatic.py \\
      --room-usd Echo_Modern_office_space_with_cubicles.usdz \\
      --mesh Echo_Modern_office_space_with_cubicles.reconstructed_mesh.glb
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import shlex
import struct
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import open3d as o3d

_REPO_ROOT = Path(__file__).resolve().parents[2]
USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
GLB_SUFFIXES = {".glb", ".gltf"}
MESH_SUFFIXES = USD_SUFFIXES | GLB_SUFFIXES
DEFAULT_TARGET = _REPO_ROOT / "assets" / "robot_room.usd"
DEFAULT_CONTAINER_REPO = "/workspace/EBiM_Challenge"
# Rx(+90°) in USD row-vector layout (p' = p @ M): (x, y, z) -> (x, -z, y)
# so a Y-up stage matches Isaac Sim's Z-up composition.
Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
# Same rotation for Open3D / Isaac column-vector xforms (p' = M @ p).
Y_UP_TO_Z_UP_COLUMN = Y_UP_TO_Z_UP.T
DEFAULT_VOXEL_SIZES = (0.1, 0.05, 0.02)
DEFAULT_MAX_CORR = (0.3, 0.15, 0.06)
DEFAULT_N_POINTS = 100_000
DEFAULT_VIS_POINTS = 40_000
DEFAULT_VIS_SIZE = (640, 480)
_SCALE_EPS = 1e-12
_MIN_DOWNSAMPLED = 8
_SOURCE_RGB = (0.95, 0.45, 0.12)
_TARGET_RGB = (0.22, 0.58, 0.88)
_BG_RGB = (18, 20, 24)
_VIEWPOINTS = (
    ("iso", np.array([1.0, -1.0, 0.55])),
    ("front", np.array([0.0, -1.15, 0.25])),
    ("side", np.array([1.15, 0.0, 0.25])),
    ("top", np.array([0.0, 0.0, 1.25])),
)


class SimilarityICPResult(TypedDict):
    """Output of :func:`multiscale_similarity_icp`."""

    transformed_source: Any  # o3d.geometry.PointCloud
    transformation: npt.NDArray[np.float64]
    scale: float
    rotation: npt.NDArray[np.float64]
    translation: npt.NDArray[np.float64]


def _require_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "align_automatic.py needs Open3D. Install with: pip install open3d"
        ) from exc
    return o3d


def _require_pxr() -> tuple[Any, Any]:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise ImportError(
            "USD loading needs pxr. Run under Isaac Sim python.sh "
            "or pip install usd-core."
        ) from exc
    return Usd, UsdGeom


def _as_4x4(matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"expected a 4x4 matrix, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("transformation matrix contains NaN or Inf")
    return array


def decompose_similarity(
    transformation: npt.ArrayLike,
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Split a 4x4 similarity matrix into ``(s, R, t)`` with ``s >= 0``.

    Open3D's scaled point-to-point ICP (Umeyama) stores the similarity as
    ``T = [[s R, t], [0, 1]]``. Polar decomposition via SVD yields a
    proper rotation (``det R = +1``) and a non-negative uniform scale
    (mean of the singular values), which Isaac Sim can apply as a
    quaternion + ``--scale``.

    Args:
        transformation: Homogeneous 4x4 similarity (source → target).

    Returns:
        ``s`` (scalar ``>= 0``), orthonormal ``R`` in SO(3), and ``t`` (3,).
    """
    matrix = _as_4x4(transformation)
    top_left = matrix[:3, :3]
    u_mat, singular, vt_mat = np.linalg.svd(top_left)
    rotation = u_mat @ vt_mat
    if float(np.linalg.det(rotation)) < 0.0:
        u_mat = u_mat.copy()
        u_mat[:, -1] *= -1.0
        rotation = u_mat @ vt_mat
    scale = float(np.mean(singular))
    if scale < _SCALE_EPS:
        det = float(np.linalg.det(top_left))
        raise ValueError(
            f"degenerate similarity scale {scale:.3e} "
            f"(det of 3x3 block is {det:.3e})"
        )
    translation = matrix[:3, 3].copy()
    return scale, rotation, translation


def compose_similarity(
    scale: float,
    rotation: npt.ArrayLike,
    translation: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Build ``T = [[s R, t], [0, 1]]`` from isolated similarity parts."""
    rot = np.asarray(rotation, dtype=np.float64)
    trans = np.asarray(translation, dtype=np.float64).reshape(3)
    if rot.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {rot.shape}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = abs(float(scale)) * rot
    matrix[:3, 3] = trans
    return matrix


def native_to_aligned_frame(path: Path | str) -> npt.NDArray[np.float64]:
    """Column-vector 4x4 baked into alignment vertices (native → Z-up).

    glTF/GLB is Y-up; ``glb_to_triangle_mesh`` applies Rx(+90°) so the
    source matches a Z-up CAD target. Isaac Sim loads the same GLB/USDZ
    in its native frame, so the teleop command must include this xform.
    """
    suffix = Path(path).suffix.lower()
    if suffix in GLB_SUFFIXES:
        return Y_UP_TO_Z_UP_COLUMN.copy()
    return np.eye(4, dtype=np.float64)


def scene_similarity_from_alignment(
    align_matrix: npt.ArrayLike,
    source_path: Path | str,
) -> tuple[
    float, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Native-source → CAD similarity for ``scene_room.py`` flags.

    ``align_matrix`` maps the *aligned* (Z-up) source onto the target.
    The loader xform is composed on the right so the result applies to
    the original native cloud. Scale is ``>= 0``.

    Returns:
        ``(scale, rotation, translation, command_matrix)``.
    """
    matrix = _as_4x4(align_matrix) @ native_to_aligned_frame(source_path)
    scale, rotation, translation = decompose_similarity(matrix)
    command = compose_similarity(scale, rotation, translation)
    return scale, rotation, translation, command


def xyz_deg_to_rotation(
    xyz_deg: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """3x3 rotation from USD rotateXYZ Euler degrees (R = Rz · Ry · Rx)."""
    angles = np.asarray(xyz_deg, dtype=np.float64).reshape(3)
    rx, ry, rz = (math.radians(float(angle)) for angle in angles)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float64,
    )
    rot_y = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float64,
    )
    rot_z = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


def rotation_to_xyz_deg(
    rotation: npt.ArrayLike,
) -> tuple[float, float, float]:
    """USD rotateXYZ Euler degrees from a 3x3 rotation (R = Rz · Ry · Rx)."""
    rot = np.asarray(rotation, dtype=np.float64)
    if rot.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {rot.shape}")
    # R[2, 0] = -sin(pitch) for XYZ (intrinsic) / rotateXYZ.
    sin_pitch = float(np.clip(-rot[2, 0], -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) > 1e-8:
        roll = math.atan2(rot[2, 1], rot[2, 2])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
    else:
        # Gimbal lock: yaw and roll share an axis; put the leftover in yaw.
        roll = 0.0
        yaw = math.atan2(-rot[0, 1], rot[1, 1])
    return (
        0.0 if abs(math.degrees(roll)) < 1e-9 else math.degrees(roll),
        0.0 if abs(math.degrees(pitch)) < 1e-9 else math.degrees(pitch),
        0.0 if abs(math.degrees(yaw)) < 1e-9 else math.degrees(yaw),
    )


def format_scene_room_flags(
    scale: float,
    rotation: npt.ArrayLike,
    translation: npt.ArrayLike,
) -> str:
    """Pasteable ``scene_room.py`` / ``align_nudger.py`` xform flags."""
    rx, ry, rz = rotation_to_xyz_deg(rotation)
    tx, ty, tz = (float(v) for v in np.asarray(translation, dtype=np.float64))
    return (
        f"--xyz-deg {rx:g} {ry:g} {rz:g} "
        f"--xyz {tx:g} {ty:g} {tz:g} "
        f"--scale {abs(float(scale)):g}"
    )


def parse_scene_room_flags(
    flags: str | Sequence[str],
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Read ``--xyz-deg`` / ``--xyz`` / ``--scale`` from a flag string.

    Extra tokens (teleop launcher args, ``--``, ``--room-usd``, ``--mesh``,
    ...) are ignored. ``--scale`` may be one uniform value or three values
    (mean of the absolute components).
    """
    argv = shlex.split(flags) if isinstance(flags, str) else list(flags)
    argv = [token for token in argv if token != "--"]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--xyz-deg", type=float, nargs=3, default=None)
    parser.add_argument("--xyz", type=float, nargs=3, default=None)
    parser.add_argument("--scale", type=float, nargs="+", default=None)
    args, _unknown = parser.parse_known_args(argv)
    if args.xyz_deg is None or args.xyz is None or args.scale is None:
        raise ValueError("need --xyz-deg, --xyz, and --scale")
    scales = [abs(float(value)) for value in args.scale]
    if len(scales) not in (1, 3):
        raise ValueError("--scale expects 1 or 3 floats")
    scale = float(sum(scales) / len(scales))
    rotation = xyz_deg_to_rotation(args.xyz_deg)
    translation = np.asarray(args.xyz, dtype=np.float64)
    return scale, rotation, translation


def container_path(
    path: Path,
    container_repo: str = DEFAULT_CONTAINER_REPO,
) -> str:
    """Map a host path under the repo root to the Isaac container mount."""
    resolved = path.expanduser().resolve()
    try:
        rel = resolved.relative_to(_REPO_ROOT)
    except ValueError:
        return resolved.as_posix()
    return f"{container_repo.rstrip('/')}/{rel.as_posix()}"


def format_teleop_command(
    *,
    room_usd: Path,
    mesh: Path,
    scale: float,
    rotation: npt.ArrayLike,
    translation: npt.ArrayLike,
    public_ip: str = "???",
    container_repo: str = DEFAULT_CONTAINER_REPO,
) -> str:
    """Full ``run_isaacsim_teleop.sh`` line with NuRec paths and ICP xform."""
    xform = format_scene_room_flags(scale, rotation, translation)
    room = container_path(room_usd, container_repo)
    mesh_c = container_path(mesh, container_repo)
    return (
        f"PUBLIC_IP={public_ip} CONTAINER_REPO={container_repo} \\\n"
        f"  bash task2_isaacsim/scripts/run_isaacsim_teleop.sh"
        f"   --scene room --no-browser --livestream --"
        f"  --room-usd {room}"
        f"    --mesh {mesh_c}  {xform}"
        f"  --record --spine-keyboard-min 0.50"
        f" --spine-keyboard-max 0.50 --render-hz 30"
    )


def _as_xyz(cloud: Any) -> npt.NDArray[np.float64]:
    if isinstance(cloud, np.ndarray):
        points = np.asarray(cloud, dtype=np.float64)
    else:
        points = np.asarray(cloud.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected Nx3 points, got shape {points.shape}")
    return points


def _subsample_xyz(
    points: npt.NDArray[np.float64],
    max_points: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    if max_points < 1 or len(points) <= max_points:
        return points
    index = rng.choice(len(points), int(max_points), replace=False)
    return points[index]


def _write_png(path: Path, rgb: npt.NDArray[np.uint8]) -> Path:
    """Write an 8-bit RGB PNG with stdlib zlib (no PIL / matplotlib)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected HxWx3 uint8, got {rgb.shape} {rgb.dtype}")
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    return path


def _look_at_basis(
    eye: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    up: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    forward = target - eye
    norm = float(np.linalg.norm(forward))
    if norm < 1e-9:
        raise ValueError("camera eye and target are coincident")
    forward = forward / norm
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-9:
        right = np.cross(forward, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1e-9:
            right = np.cross(
                forward, np.array([0.0, 1.0, 0.0], dtype=np.float64)
            )
            right_norm = float(np.linalg.norm(right))
    right = right / right_norm
    cam_up = np.cross(right, forward)
    cam_up = cam_up / float(np.linalg.norm(cam_up))
    return right, cam_up, forward


def rasterize_point_clouds(
    clouds: Sequence[
        tuple[npt.NDArray[np.float64], tuple[float, float, float]]
    ],
    *,
    eye: npt.ArrayLike,
    look_at: npt.ArrayLike,
    up: npt.ArrayLike = (0.0, 0.0, 1.0),
    width: int = 640,
    height: int = 480,
    fov_deg: float = 50.0,
    point_px: int = 2,
    background: tuple[int, int, int] = _BG_RGB,
) -> npt.NDArray[np.uint8]:
    """Perspective z-buffer splat of colored clouds. Far points draw first."""
    eye_v = np.asarray(eye, dtype=np.float64).reshape(3)
    look = np.asarray(look_at, dtype=np.float64).reshape(3)
    up_v = np.asarray(up, dtype=np.float64).reshape(3)
    right, cam_up, forward = _look_at_basis(eye_v, look, up_v)
    near = 1e-3
    chunks: list[npt.NDArray[np.float64]] = []
    colors: list[npt.NDArray[np.float64]] = []
    for points, rgb in clouds:
        if len(points) == 0:
            continue
        chunks.append(np.asarray(points, dtype=np.float64))
        colors.append(
            np.broadcast_to(
                np.asarray(rgb, dtype=np.float64).reshape(1, 3),
                (len(points), 3),
            )
        )
    image = np.full((height, width, 3), background, dtype=np.uint8)
    if not chunks:
        return image
    xyz = np.concatenate(chunks, axis=0)
    rgb = np.concatenate(colors, axis=0)
    rel = xyz - eye_v
    cam_x = rel @ right
    cam_y = rel @ cam_up
    depth = rel @ forward
    visible = depth > near
    if not np.any(visible):
        return image
    cam_x = cam_x[visible]
    cam_y = cam_y[visible]
    depth = depth[visible]
    rgb = rgb[visible]
    focal = 0.5 * height / math.tan(math.radians(float(fov_deg)) / 2.0)
    col = cam_x * (focal / depth) + (width * 0.5)
    row = (height * 0.5) - cam_y * (focal / depth)
    order = np.argsort(-depth)
    col = np.rint(col[order]).astype(np.int32)
    row = np.rint(row[order]).astype(np.int32)
    rgb = rgb[order]
    radius = max(0, int(point_px) // 2)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            cc = col + dx
            rr = row + dy
            inside = (cc >= 0) & (cc < width) & (rr >= 0) & (rr < height)
            image[rr[inside], cc[inside]] = np.clip(
                rgb[inside] * 255.0, 0, 255
            ).astype(np.uint8)
    return image


def _shared_cameras(
    points: npt.NDArray[np.float64],
) -> list[
    tuple[
        str,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]
]:
    if len(points) == 0:
        raise ValueError("cannot frame cameras around an empty cloud")
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.65 * float(np.linalg.norm(hi - lo))
    radius = max(radius, 0.5)
    z_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cameras = []
    for name, offset in _VIEWPOINTS:
        eye = center + offset * radius
        up = y_up if name == "top" else z_up
        cameras.append((name, eye, center, up))
    return cameras


def _hstack(
    images: Sequence[npt.NDArray[np.uint8]], gap: int = 8
) -> npt.NDArray[np.uint8]:
    if not images:
        raise ValueError("no images to stack")
    height = images[0].shape[0]
    gap_img = np.full((height, gap, 3), 40, dtype=np.uint8)
    parts: list[npt.NDArray[np.uint8]] = []
    for i, image in enumerate(images):
        if image.shape[0] != height:
            raise ValueError("hstack images must share height")
        if i:
            parts.append(gap_img)
        parts.append(image)
    return np.concatenate(parts, axis=1)


def _vstack(
    images: Sequence[npt.NDArray[np.uint8]], gap: int = 8
) -> npt.NDArray[np.uint8]:
    if not images:
        raise ValueError("no images to stack")
    width = images[0].shape[1]
    gap_img = np.full((gap, width, 3), 40, dtype=np.uint8)
    parts: list[npt.NDArray[np.uint8]] = []
    for i, image in enumerate(images):
        if image.shape[1] != width:
            raise ValueError("vstack images must share width")
        if i:
            parts.append(gap_img)
        parts.append(image)
    return np.concatenate(parts, axis=0)


def save_alignment_views(
    source_before: Any,
    source_after: Any,
    target: Any,
    output_dir: Path,
    *,
    max_points: int = DEFAULT_VIS_POINTS,
    width: int = DEFAULT_VIS_SIZE[0],
    height: int = DEFAULT_VIS_SIZE[1],
    seed: int = 0,
) -> list[Path]:
    """Write before/after overlays from several viewpoints.

    Orange is source, blue is target.
    """
    rng = np.random.default_rng(seed)
    before = _subsample_xyz(_as_xyz(source_before), max_points, rng)
    after = _subsample_xyz(_as_xyz(source_after), max_points, rng)
    tgt = _subsample_xyz(_as_xyz(target), max_points, rng)
    cameras = _shared_cameras(np.concatenate([before, after, tgt], axis=0))
    before_row: list[npt.NDArray[np.uint8]] = []
    after_row: list[npt.NDArray[np.uint8]] = []
    written: list[Path] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, eye, look_at, up in cameras:
        kwargs = dict(
            eye=eye, look_at=look_at, up=up, width=width, height=height
        )
        before_img = rasterize_point_clouds(
            [(tgt, _TARGET_RGB), (before, _SOURCE_RGB)], **kwargs
        )
        after_img = rasterize_point_clouds(
            [(tgt, _TARGET_RGB), (after, _SOURCE_RGB)], **kwargs
        )
        pair = _hstack([before_img, after_img])
        written.append(_write_png(output_dir / f"{name}.png", pair))
        before_row.append(before_img)
        after_row.append(after_img)
    compare = _vstack([_hstack(before_row), _hstack(after_row)])
    written.insert(0, _write_png(output_dir / "compare.png", compare))
    return written


def estimate_scale_init(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> npt.NDArray[np.float64]:
    """Identity rotation, uniform scale from AABB diagonals, centroid shift.

    ``p_target ≈ s p_source + t`` with ``t = c_t - s c_s``. Used as an ICP
    init when the two rooms are in different units.
    """
    src = np.asarray(source.points, dtype=np.float64)
    tgt = np.asarray(target.points, dtype=np.float64)
    if src.size == 0 or tgt.size == 0:
        raise ValueError("cannot estimate scale init from an empty cloud")
    src_diag = float(np.linalg.norm(src.max(axis=0) - src.min(axis=0)))
    tgt_diag = float(np.linalg.norm(tgt.max(axis=0) - tgt.min(axis=0)))
    if src_diag < _SCALE_EPS:
        raise ValueError("source AABB diagonal is degenerate")
    scale = tgt_diag / src_diag
    src_c = src.mean(axis=0)
    tgt_c = tgt.mean(axis=0)
    translation = tgt_c - scale * src_c
    return compose_similarity(scale, np.eye(3), translation)


def _triangulate_faces(
    counts: Sequence[int],
    indices: Sequence[int],
) -> list[tuple[int, int, int]]:
    """Fan-triangulate USD faceVertexCounts / faceVertexIndices."""
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for count in counts:
        n_verts = int(count)
        face = [int(i) for i in indices[cursor : cursor + n_verts]]
        cursor += n_verts
        if n_verts < 3:
            continue
        origin = face[0]
        for k in range(1, n_verts - 1):
            triangles.append((origin, face[k], face[k + 1]))
    return triangles


def _gf_matrix_to_numpy(matrix: Any) -> npt.NDArray[np.float64]:
    """Convert a Gf.Matrix4d to a 4x4 numpy array (USD row-vector layout)."""
    array = np.array(matrix, dtype=np.float64)
    if array.shape == (4, 4):
        return array
    return np.array(
        [[matrix[row][col] for col in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def _transform_points(
    points: npt.NDArray[np.float64],
    matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply a 4x4 USD row-vector transform: ``p' = p @ M``."""
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack((points, ones)) @ matrix
    return homogeneous[:, :3]


def _require_geometry_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() not in MESH_SUFFIXES:
        raise ValueError(
            f"{path} is not a USD or GLB file (expected "
            f"{', '.join(sorted(MESH_SUFFIXES))})"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"mesh file not found: {resolved}")
    return resolved


def _require_usd_path(path: Path) -> Path:
    resolved = _require_geometry_path(path)
    if resolved.suffix.lower() not in USD_SUFFIXES:
        raise ValueError(
            f"{path} is not a USD file (expected "
            f"{', '.join(sorted(USD_SUFFIXES))})"
        )
    return resolved


def _usd_prim_vertex_colors(
    mesh: Any, n_verts: int
) -> npt.NDArray[np.float64]:
    """UsdGeom displayColor, or a neutral gray if the prim has none."""
    colors = np.full((n_verts, 3), 0.72, dtype=np.float64)
    getter = getattr(mesh, "GetDisplayColorAttr", None)
    if getter is None:
        return colors
    raw = getter().Get()
    if not raw:
        return colors
    arr = np.array(
        [(float(c[0]), float(c[1]), float(c[2])) for c in raw],
        dtype=np.float64,
    )
    if arr.shape == (1, 3):
        colors[:] = arr[0]
        return colors
    if len(arr) == n_verts:
        return arr
    return colors


def usd_to_triangle_mesh(path: Path | str) -> o3d.geometry.TriangleMesh:
    """Load every ``UsdGeom.Mesh`` in *path* into one world-space Open3D mesh.

    World xforms are baked in. Points are converted to metres via
    ``metresPerUnit``. A Y-up stage is rotated Rx(+90°) so the result matches
    Isaac Sim's Z-up composition.
    """
    o3d = _require_open3d()
    Usd, UsdGeom = _require_pxr()
    usd_path = _require_usd_path(Path(path))

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"pxr could not open USD stage: {usd_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    metres = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()

    vertices: list[npt.NDArray[np.float64]] = []
    triangles: list[tuple[int, int, int]] = []
    vert_colors: list[npt.NDArray[np.float64]] = []
    vertex_offset = 0

    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr().Get()
        if not points_attr:
            continue
        local = np.array(
            [(float(p[0]), float(p[1]), float(p[2])) for p in points_attr],
            dtype=np.float64,
        )
        world_m = _gf_matrix_to_numpy(
            xform_cache.GetLocalToWorldTransform(prim)
        )
        world = _transform_points(local, world_m)
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        faces = _triangulate_faces(counts, indices)
        vertices.append(world)
        vert_colors.append(_usd_prim_vertex_colors(mesh, world.shape[0]))
        triangles.extend(
            (
                a + vertex_offset,
                b + vertex_offset,
                c + vertex_offset,
            )
            for a, b, c in faces
        )
        vertex_offset += world.shape[0]

    if not vertices:
        raise ValueError(f"no UsdGeom.Mesh prims with points in {usd_path}")

    points = np.vstack(vertices) * metres
    if up_axis == "Y":
        points = _transform_points(points, Y_UP_TO_Z_UP)

    mesh_out = o3d.geometry.TriangleMesh()
    mesh_out.vertices = o3d.utility.Vector3dVector(points)
    mesh_out.vertex_colors = o3d.utility.Vector3dVector(np.vstack(vert_colors))
    if triangles:
        mesh_out.triangles = o3d.utility.Vector3iVector(
            np.asarray(triangles, dtype=np.int32)
        )
    return mesh_out


def glb_to_triangle_mesh(
    path: Path | str,
    *,
    y_up_to_z_up: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Load a glTF/GLB file as one Open3D triangle mesh.

    glTF is Y-up. By default vertices are rotated Rx(+90°) into Isaac
    Z-up so a GLB source lines up with a Z-up USD target. Pass
    ``y_up_to_z_up=False`` to keep the original native cloud.
    """
    o3d = _require_open3d()
    glb_path = _require_geometry_path(Path(path))
    if glb_path.suffix.lower() not in GLB_SUFFIXES:
        raise ValueError(
            f"{path} is not a GLB/glTF file (expected "
            f"{', '.join(sorted(GLB_SUFFIXES))})"
        )

    meshes: list[Any] = []
    read_model = getattr(o3d.io, "read_triangle_model", None)
    if read_model is not None:
        model = read_model(str(glb_path))
        for info in getattr(model, "meshes", []):
            part = getattr(info, "mesh", info)
            if len(part.vertices) > 0:
                meshes.append(part)
    if not meshes:
        part = o3d.io.read_triangle_mesh(str(glb_path))
        if len(part.vertices) > 0:
            meshes.append(part)
    if not meshes:
        raise ValueError(f"no mesh vertices in {glb_path}")

    combined = meshes[0]
    for extra in meshes[1:]:
        combined += extra
    if y_up_to_z_up:
        points = _transform_points(
            np.asarray(combined.vertices, dtype=np.float64),
            Y_UP_TO_Z_UP,
        )
        combined.vertices = o3d.utility.Vector3dVector(points)
    return combined


def load_triangle_mesh(
    path: Path | str,
    *,
    y_up_to_z_up: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Dispatch USD/USDZ vs GLB/glTF into a world-space Open3D mesh."""
    resolved = _require_geometry_path(Path(path))
    if resolved.suffix.lower() in GLB_SUFFIXES:
        return glb_to_triangle_mesh(resolved, y_up_to_z_up=y_up_to_z_up)
    return usd_to_triangle_mesh(resolved)


def sample_mesh_to_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    n_points: int = DEFAULT_N_POINTS,
) -> o3d.geometry.PointCloud:
    """Uniformly sample *n_points* from a triangle mesh.

    Falls back to the vertex set when the mesh has no faces (point-only USD).
    """
    o3d = _require_open3d()
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    n_tris = len(mesh.triangles)
    if n_tris == 0:
        if len(mesh.vertices) == 0:
            raise ValueError("mesh has no vertices or triangles to sample")
        cloud = o3d.geometry.PointCloud()
        cloud.points = mesh.vertices
        return cloud
    return mesh.sample_points_uniformly(number_of_points=int(n_points))


def usd_to_point_cloud(
    path: Path | str,
    n_points: int = DEFAULT_N_POINTS,
) -> o3d.geometry.PointCloud:
    """USD stage → world-space Open3D point cloud (sampled from mesh faces)."""
    return sample_mesh_to_point_cloud(usd_to_triangle_mesh(path), n_points)


def load_as_point_cloud(
    path: Path | str,
    n_points: int = DEFAULT_N_POINTS,
    *,
    y_up_to_z_up: bool = True,
) -> o3d.geometry.PointCloud:
    """USD or GLB path → sampled Open3D point cloud."""
    return sample_mesh_to_point_cloud(
        load_triangle_mesh(path, y_up_to_z_up=y_up_to_z_up), n_points
    )


def load_as_point_cloud_native(
    path: Path | str,
    n_points: int = DEFAULT_N_POINTS,
) -> o3d.geometry.PointCloud:
    """Sample the original native cloud (no Y-up → Z-up bake)."""
    return load_as_point_cloud(path, n_points, y_up_to_z_up=False)


def multiscale_similarity_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_sizes: Sequence[float],
    max_correspondence_distances: Sequence[float],
    init: npt.ArrayLike | None = None,
    *,
    max_iterations: int = 50,
) -> SimilarityICPResult:
    """Coarse-to-fine ICP with a single global scale (7-DoF similarity).

    At each pyramid level both clouds are voxel-downsampled, then
    ``registration_icp`` runs with
    ``TransformationEstimationPointToPoint(with_scaling=True)``. The
    resulting 4x4 matrix is the init guess for the next (finer) level.

    Args:
        source: Moving cloud (aligned onto *target*).
        target: Fixed / ground-truth cloud.
        voxel_sizes: Downsample sizes, coarse to fine (e.g. ``[0.1, 0.05,
            0.02]``). Length must match *max_correspondence_distances*.
        max_correspondence_distances: ICP correspondence cutoff per level,
            in the same units as the clouds.
        init: Optional 4x4 source→target guess. Identity if omitted.
        max_iterations: ICP iterations at each level.

    Returns:
        Dict with the transformed source copy, the full 4x4 matrix, and the
        isolated scale / rotation / translation.
    """
    o3d = _require_open3d()
    voxels = [float(v) for v in voxel_sizes]
    distances = [float(d) for d in max_correspondence_distances]
    if not voxels:
        raise ValueError("voxel_sizes must be a non-empty list")
    if len(voxels) != len(distances):
        raise ValueError(
            "voxel_sizes and max_correspondence_distances must have "
            f"the same length ({len(voxels)} vs {len(distances)})"
        )
    if any(v <= 0.0 for v in voxels):
        raise ValueError("voxel sizes must be positive")
    if any(d <= 0.0 for d in distances):
        raise ValueError("max correspondence distances must be positive")
    if len(source.points) == 0 or len(target.points) == 0:
        raise ValueError("source and target point clouds must be non-empty")

    current = np.eye(4, dtype=np.float64) if init is None else _as_4x4(init)
    estimation = (
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True
        )
    )
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=int(max_iterations)
    )

    last_result: Any = None
    for level, (voxel, corr_dist) in enumerate(zip(voxels, distances)):
        # Downsample in each cloud's native frame; ICP applies *current*
        # internally when searching correspondences.
        source_d = source.voxel_down_sample(voxel)
        target_d = target.voxel_down_sample(voxel)
        if (
            len(source_d.points) < _MIN_DOWNSAMPLED
            or len(target_d.points) < _MIN_DOWNSAMPLED
        ):
            raise ValueError(
                f"level {level} voxel {voxel:g} left too few points "
                f"(source={len(source_d.points)}, "
                f"target={len(target_d.points)})"
            )
        last_result = o3d.pipelines.registration.registration_icp(
            source=source_d,
            target=target_d,
            max_correspondence_distance=corr_dist,
            init=current,
            estimation_method=estimation,
            criteria=criteria,
        )
        current = np.asarray(last_result.transformation, dtype=np.float64)
        print(
            f"ICP level {level}: voxel={voxel:g} corr={corr_dist:g} "
            f"fitness={last_result.fitness:.4f} "
            f"rmse={last_result.inlier_rmse:.4f} "
            f"n_src={len(source_d.points)} n_tgt={len(target_d.points)}",
            flush=True,
        )

    scale, rotation, translation = decompose_similarity(current)
    transformed = copy.deepcopy(source)
    transformed.transform(current)
    return {
        "transformed_source": transformed,
        "transformation": current,
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
    }


def _parse_init_matrix(
    values: Sequence[float] | None,
) -> npt.NDArray[np.float64] | None:
    if values is None:
        return None
    if len(values) != 16:
        raise argparse.ArgumentTypeError(
            f"--init needs 16 floats (row-major 4x4), got {len(values)}"
        )
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--room-usd",
        type=Path,
        required=True,
        help="NuRec volume USDZ for the printed teleop command.",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        required=True,
        help="Collider GLB (ICP source + --mesh in the teleop command).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="ICP moving cloud. Default: the --mesh GLB.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="ICP fixed cloud / CAD room (default: assets/robot_room.usd).",
    )
    parser.add_argument(
        "--public-ip",
        default=None,
        help="PUBLIC_IP in the printed command (default: $PUBLIC_IP or ???).",
    )
    parser.add_argument(
        "--container-repo",
        default=DEFAULT_CONTAINER_REPO,
        help=f"CONTAINER_REPO mount (default: {DEFAULT_CONTAINER_REPO}).",
    )
    parser.add_argument(
        "--voxel-sizes",
        type=float,
        nargs="+",
        default=list(DEFAULT_VOXEL_SIZES),
        metavar="V",
        help="Coarse-to-fine voxel sizes (default: 0.1 0.05 0.02).",
    )
    parser.add_argument(
        "--max-correspondence-distances",
        type=float,
        nargs="+",
        default=list(DEFAULT_MAX_CORR),
        metavar="D",
        help="ICP correspondence cutoff per level (default: 0.3 0.15 0.06).",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=DEFAULT_N_POINTS,
        help="Uniform samples drawn from each mesh (default: 100000).",
    )
    parser.add_argument(
        "--init",
        type=float,
        nargs=16,
        default=None,
        metavar="M",
        help="Optional row-major 4x4 initial transformation (16 floats).",
    )
    parser.add_argument(
        "--estimate-scale-init",
        action="store_true",
        help="Seed ICP with AABB-diagonal scale and centroid translation.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="ICP iterations at each pyramid level (default: 50).",
    )
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=None,
        help="Directory for before/after viewpoint PNGs (default: "
        "outputs/align_automatic/<mesh-stem>).",
    )
    parser.add_argument(
        "--vis-points",
        type=int,
        default=DEFAULT_VIS_POINTS,
        help="Max points per cloud in the viewpoint renders "
        f"(default: {DEFAULT_VIS_POINTS}).",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip writing before/after alignment PNGs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mesh_path = _require_geometry_path(args.mesh)
    room_usd_path = _require_usd_path(args.room_usd)
    source_path = _require_geometry_path(args.source or args.mesh)
    target_path = _require_geometry_path(args.target)
    public_ip = (
        args.public_ip or os.environ.get("PUBLIC_IP", "").strip() or "???"
    )

    print(f"Loading source {source_path}", flush=True)
    source = load_as_point_cloud(source_path, n_points=args.n_points)
    print(f"Loading target {target_path}", flush=True)
    target = load_as_point_cloud(target_path, n_points=args.n_points)
    print(
        f"Sampled {len(source.points)} source / "
        f"{len(target.points)} target points",
        flush=True,
    )

    init = _parse_init_matrix(args.init)
    if args.estimate_scale_init:
        scale_init = estimate_scale_init(source, target)
        if init is None:
            init = scale_init
        else:
            # Apply the AABB scale estimate first, then the caller guess.
            init = init @ scale_init
        est_s, _, est_t = decompose_similarity(init)
        print(
            f"Scale-init s={est_s:.4f} t={est_t}",
            flush=True,
        )

    result = multiscale_similarity_icp(
        source,
        target,
        voxel_sizes=args.voxel_sizes,
        max_correspondence_distances=args.max_correspondence_distances,
        init=init,
        max_iterations=args.max_iterations,
    )
    scale, rotation, translation, command = scene_similarity_from_alignment(
        result["transformation"], source_path
    )
    np.set_printoptions(precision=6, suppress=True)
    print("\nAligned-frame homogeneous matrix:\n", result["transformation"])
    print("\nNative-frame command matrix:\n", command)
    print(f"\nScale factor: {scale:.6f}")
    print("Rotation:\n", rotation)
    print("Translation:", translation)
    if not args.no_vis:
        native_source = load_as_point_cloud_native(
            source_path, n_points=args.n_points
        )
        after = copy.deepcopy(native_source)
        after.transform(command)
        vis_dir = args.vis_dir
        if vis_dir is None:
            vis_dir = (
                _REPO_ROOT / "outputs" / "align_automatic" / mesh_path.stem
            )
        paths = save_alignment_views(
            native_source,
            after,
            target,
            vis_dir,
            max_points=args.vis_points,
        )
        print(f"\nAlignment views ({len(paths)} PNGs): {vis_dir}")
        print(
            "  compare.png  rows=before/after ICP, cols=iso, front, side, top"
        )
        print(
            "  iso/front/side/top.png  left=before, right=after; "
            "orange=source mesh, blue=target room"
        )
    print("\nscene_room.py flags:")
    print(format_scene_room_flags(scale, rotation, translation))
    print("\nteleop command:")
    print(
        format_teleop_command(
            room_usd=room_usd_path,
            mesh=mesh_path,
            scale=scale,
            rotation=rotation,
            translation=translation,
            public_ip=public_ip,
            container_repo=args.container_repo,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
