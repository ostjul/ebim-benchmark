#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Multi-scale similarity ICP (7-DoF: scale, rotation, translation).

Aligns a reconstructed GLB collider onto ``assets/robot_room.usd``, then
prints a pasteable ``run_isaacsim_teleop.sh`` command that loads the NuRec
``.usdz`` volume with that GLB under the estimated joint xform.

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
DEFAULT_VOXEL_SIZES = (0.1, 0.05, 0.02)
DEFAULT_MAX_CORR = (0.3, 0.15, 0.06)
DEFAULT_N_POINTS = 100_000
_SCALE_EPS = 1e-12
_MIN_DOWNSAMPLED = 8


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
    """Split a 4x4 similarity matrix into ``(s, R, t)``.

    Open3D's scaled point-to-point ICP (Umeyama) stores the similarity as
    ``T = [[s R, t], [0, 1]]``. The uniform scale is the cube root of the
    determinant of the top-left 3x3 block.

    Args:
        transformation: Homogeneous 4x4 similarity (source → target).

    Returns:
        ``s`` (scalar), orthonormal ``R`` (3x3), and ``t`` (3,).
    """
    matrix = _as_4x4(transformation)
    top_left = matrix[:3, :3]
    det = float(np.linalg.det(top_left))
    scale = float(np.cbrt(det))
    if abs(scale) < _SCALE_EPS:
        raise ValueError(
            f"degenerate similarity scale {scale:.3e} "
            f"(det of 3x3 block is {det:.3e})"
        )
    rotation = top_left / scale
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
    matrix[:3, :3] = float(scale) * rot
    matrix[:3, 3] = trans
    return matrix


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
        f"--scale {float(scale):g}"
    )


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
        f"   --scene room --no-browser --"
        f"  --room-usd {room}"
        f"    --mesh {mesh_c}  {xform}"
        f"  --record --spine-keyboard-min 0.50"
        f" --spine-keyboard-max 0.50 --render-hz 30"
    )


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
    if triangles:
        mesh_out.triangles = o3d.utility.Vector3iVector(
            np.asarray(triangles, dtype=np.int32)
        )
    return mesh_out


def glb_to_triangle_mesh(path: Path | str) -> o3d.geometry.TriangleMesh:
    """Load a glTF/GLB file as one Open3D triangle mesh.

    glTF is Y-up; vertices are rotated Rx(+90°) into Isaac Z-up so a GLB
    source lines up with a Z-up USD target.
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
    points = _transform_points(
        np.asarray(combined.vertices, dtype=np.float64),
        Y_UP_TO_Z_UP,
    )
    combined.vertices = o3d.utility.Vector3dVector(points)
    return combined


def load_triangle_mesh(path: Path | str) -> o3d.geometry.TriangleMesh:
    """Dispatch USD/USDZ vs GLB/glTF into a world-space Open3D mesh."""
    resolved = _require_geometry_path(Path(path))
    if resolved.suffix.lower() in GLB_SUFFIXES:
        return glb_to_triangle_mesh(resolved)
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
) -> o3d.geometry.PointCloud:
    """USD or GLB path → sampled Open3D point cloud."""
    return sample_mesh_to_point_cloud(load_triangle_mesh(path), n_points)


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
        help="ICP fixed cloud / CAD room "
        "(default: assets/robot_room.usd).",
    )
    parser.add_argument(
        "--public-ip",
        default=None,
        help="PUBLIC_IP in the printed command "
        "(default: $PUBLIC_IP or ???).",
    )
    parser.add_argument(
        "--container-repo",
        default=DEFAULT_CONTAINER_REPO,
        help="CONTAINER_REPO mount "
        f"(default: {DEFAULT_CONTAINER_REPO}).",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mesh_path = _require_geometry_path(args.mesh)
    room_usd_path = _require_usd_path(args.room_usd)
    source_path = _require_geometry_path(args.source or args.mesh)
    target_path = _require_geometry_path(args.target)
    public_ip = (
        args.public_ip
        or os.environ.get("PUBLIC_IP", "").strip()
        or "???"
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
    scale = result["scale"]
    rotation = result["rotation"]
    translation = result["translation"]
    np.set_printoptions(precision=6, suppress=True)
    print("\nFull homogeneous matrix:\n", result["transformation"])
    print(f"\nScale factor: {scale:.6f}")
    print("Rotation:\n", rotation)
    print("Translation:", translation)
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
