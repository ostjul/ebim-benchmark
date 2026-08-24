# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "task2_isaacsim" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import align_automatic as align  # noqa: E402


def _need_open3d():
    return pytest.importorskip("open3d")


def _rotz(degrees: float) -> np.ndarray:
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _box_points() -> np.ndarray:
    xs = np.linspace(0.0, 2.0, 21)
    ys = np.linspace(0.0, 1.0, 11)
    zs = np.linspace(0.0, 0.4, 5)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def test_decompose_similarity_recovers_s_r_t():
    scale = 1.37
    rotation = _rotz(35.0)
    translation = np.array([0.4, -0.2, 1.1], dtype=np.float64)
    matrix = align.compose_similarity(scale, rotation, translation)
    got_s, got_r, got_t = align.decompose_similarity(matrix)
    assert got_s == pytest.approx(scale, rel=1e-9)
    np.testing.assert_allclose(got_r, rotation, atol=1e-9)
    np.testing.assert_allclose(got_t, translation, atol=1e-9)


def test_decompose_similarity_rejects_non_4x4():
    with pytest.raises(ValueError, match="4x4"):
        align.decompose_similarity(np.eye(3))


def test_rotation_to_xyz_deg_matches_rotz():
    rx, ry, rz = align.rotation_to_xyz_deg(_rotz(40.0))
    assert rx == pytest.approx(0.0, abs=1e-6)
    assert ry == pytest.approx(0.0, abs=1e-6)
    assert rz == pytest.approx(40.0, abs=1e-6)


def test_format_scene_room_flags():
    flags = align.format_scene_room_flags(
        1.25, np.eye(3), np.array([0.5, -1.0, 2.0])
    )
    assert flags == "--xyz-deg 0 0 0 --xyz 0.5 -1 2 --scale 1.25"


def test_format_teleop_command_embeds_paths_and_xform():
    cmd = align.format_teleop_command(
        room_usd=REPO_ROOT / "Echo_Modern_office_space_with_cubicles.usdz",
        mesh=REPO_ROOT
        / "Echo_Modern_office_space_with_cubicles.reconstructed_mesh.glb",
        scale=1.25,
        rotation=np.eye(3),
        translation=np.array([0.5, -1.0, 2.0]),
        public_ip="1.2.3.4",
        container_repo="/workspace/EBiM_Challenge",
    )
    assert cmd.startswith(
        "PUBLIC_IP=1.2.3.4 CONTAINER_REPO=/workspace/EBiM_Challenge"
    )
    assert "--scene room --no-browser --" in cmd
    assert (
        "--room-usd /workspace/EBiM_Challenge/"
        "Echo_Modern_office_space_with_cubicles.usdz" in cmd
    )
    assert (
        "--mesh /workspace/EBiM_Challenge/"
        "Echo_Modern_office_space_with_cubicles.reconstructed_mesh.glb" in cmd
    )
    assert "--xyz-deg 0 0 0 --xyz 0.5 -1 2 --scale 1.25" in cmd
    assert "--record --spine-keyboard-min 0.50" in cmd
    assert "--spine-keyboard-max 0.50 --render-hz 30" in cmd


def test_require_geometry_path_rejects_ply(tmp_path: Path):
    ply = tmp_path / "mesh.ply"
    ply.write_text("ply\n")
    with pytest.raises(ValueError, match="not a USD or GLB"):
        align._require_geometry_path(ply)


def test_require_geometry_path_accepts_glb(tmp_path: Path):
    glb = tmp_path / "mesh.glb"
    glb.write_bytes(b"glTF")
    assert align._require_geometry_path(glb) == glb.resolve()


def test_require_usd_path_rejects_glb(tmp_path: Path):
    glb = tmp_path / "mesh.glb"
    glb.write_bytes(b"glTF")
    with pytest.raises(ValueError, match="not a USD file"):
        align._require_usd_path(glb)


def test_y_up_to_z_up_sends_y_axis_to_z():
    points = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    out = align._transform_points(points, align.Y_UP_TO_Z_UP)
    np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(out[1], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(out[2], [1.0, 0.0, 0.0])


def test_triangulate_quad_is_two_triangles():
    tris = align._triangulate_faces([4], [0, 1, 2, 3])
    assert tris == [(0, 1, 2), (0, 2, 3)]


def test_multiscale_similarity_icp_recovers_known_transform():
    o3d = _need_open3d()
    rng = np.random.default_rng(0)
    points = _box_points()
    points = points + rng.normal(0.0, 0.001, size=points.shape)

    scale = 1.08
    rotation = _rotz(5.0)
    translation = np.array([0.05, 0.02, 0.01], dtype=np.float64)
    truth = align.compose_similarity(scale, rotation, translation)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(
        (scale * (rotation @ points.T).T) + translation
    )

    result = align.multiscale_similarity_icp(
        source,
        target,
        voxel_sizes=[0.2, 0.08],
        max_correspondence_distances=[0.6, 0.2],
        init=np.eye(4),
        max_iterations=80,
    )
    assert result["scale"] == pytest.approx(scale, rel=0.02, abs=0.02)
    np.testing.assert_allclose(result["rotation"], rotation, atol=0.03)
    np.testing.assert_allclose(result["translation"], translation, atol=0.03)
    np.testing.assert_allclose(result["transformation"], truth, atol=0.05)
    assert len(result["transformed_source"].points) == len(source.points)
    # Inputs must not be mutated.
    np.testing.assert_allclose(np.asarray(source.points), points)


def test_multiscale_similarity_icp_length_mismatch():
    o3d = _need_open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(_box_points())
    with pytest.raises(ValueError, match="same length"):
        align.multiscale_similarity_icp(
            cloud,
            cloud,
            voxel_sizes=[0.1, 0.05],
            max_correspondence_distances=[0.2],
        )
