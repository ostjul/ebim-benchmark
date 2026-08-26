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


def test_decompose_similarity_positive_scale_for_reflection():
    matrix = np.diag([1.5, 1.5, -1.5, 1.0])
    scale, rotation, translation = align.decompose_similarity(matrix)
    assert scale == pytest.approx(1.5)
    assert scale >= 0.0
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0)
    np.testing.assert_allclose(translation, 0.0)


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


def test_format_scene_room_flags_abs_scale():
    flags = align.format_scene_room_flags(
        -1.25, np.eye(3), np.array([0.5, -1.0, 2.0])
    )
    assert flags == "--xyz-deg 0 0 0 --xyz 0.5 -1 2 --scale 1.25"


def test_xyz_deg_to_rotation_roundtrip():
    for degrees in (
        (0.0, 0.0, 0.0),
        (90.0, 0.0, 0.0),
        (10.0, 20.0, 30.0),
        (89.4646, 1.3006, 1.54099),
    ):
        rotation = align.xyz_deg_to_rotation(degrees)
        got = align.rotation_to_xyz_deg(rotation)
        np.testing.assert_allclose(got, degrees, atol=1e-6)


def test_parse_scene_room_flags_roundtrip():
    flags = (
        "--xyz-deg 89.4646 1.3006 1.54099 "
        "--xyz 2.01734 2.003 1.70 --scale 1.18892"
    )
    scale, rotation, translation = align.parse_scene_room_flags(flags)
    assert scale == pytest.approx(1.18892)
    np.testing.assert_allclose(translation, [2.01734, 2.003, 1.70])
    rx, ry, rz = align.rotation_to_xyz_deg(rotation)
    assert rx == pytest.approx(89.4646)
    assert ry == pytest.approx(1.3006)
    assert rz == pytest.approx(1.54099)
    rebuilt = align.format_scene_room_flags(scale, rotation, translation)
    again = align.parse_scene_room_flags(rebuilt)
    assert again[0] == pytest.approx(scale)
    np.testing.assert_allclose(again[1], rotation)
    np.testing.assert_allclose(again[2], translation)


def test_parse_scene_room_flags_ignores_extra_tokens():
    cmd = (
        "PUBLIC_IP=1.2.3.4 bash task2_isaacsim/scripts/run_isaacsim_teleop.sh "
        "--scene room -- --room-usd /tmp/a.usdz --mesh /tmp/a.glb "
        "--xyz-deg 90 0 0 --xyz 1 2 3 --scale 2 --record"
    )
    scale, rotation, translation = align.parse_scene_room_flags(cmd)
    assert scale == pytest.approx(2.0)
    np.testing.assert_allclose(translation, [1.0, 2.0, 3.0])
    rx, ry, rz = align.rotation_to_xyz_deg(rotation)
    assert rx == pytest.approx(90.0)
    assert ry == pytest.approx(0.0)
    assert rz == pytest.approx(0.0)


def test_parse_scene_room_flags_requires_all_flags():
    with pytest.raises(ValueError, match="need --xyz-deg"):
        align.parse_scene_room_flags("--xyz 0 0 0 --scale 1")


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
    assert "--scene room --no-browser --livestream --" in cmd
    assert "--align " not in cmd
    assert " --align" not in cmd
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


def test_native_to_aligned_frame_glb_is_rx90():
    matrix = align.native_to_aligned_frame("room.glb")
    np.testing.assert_allclose(matrix, align.Y_UP_TO_Z_UP_COLUMN)
    point = np.array([0.0, 1.0, 0.0, 1.0])
    out = matrix @ point
    np.testing.assert_allclose(out[:3], [0.0, 0.0, 1.0])
    rx, ry, rz = align.rotation_to_xyz_deg(matrix[:3, :3])
    assert rx == pytest.approx(90.0)
    assert ry == pytest.approx(0.0)
    assert rz == pytest.approx(0.0)
    np.testing.assert_allclose(
        align.native_to_aligned_frame("room.usd"), np.eye(4)
    )


def test_scene_similarity_from_alignment_bakes_glb_rx90():
    scale, rotation, translation, command = (
        align.scene_similarity_from_alignment(np.eye(4), "room.glb")
    )
    assert scale == pytest.approx(1.0)
    assert scale >= 0.0
    np.testing.assert_allclose(translation, 0.0)
    rx, ry, rz = align.rotation_to_xyz_deg(rotation)
    assert rx == pytest.approx(90.0)
    assert ry == pytest.approx(0.0)
    assert rz == pytest.approx(0.0)
    point = np.array([0.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose((command @ point)[:3], [0.0, 0.0, 1.0])


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


def test_write_png_roundtrip_header(tmp_path: Path):
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[1, 2] = (255, 128, 0)
    path = align._write_png(tmp_path / "tiny.png", rgb)
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in data
    assert path.stat().st_size > 32


def test_rasterize_point_clouds_paints_source_and_target():
    src = np.array([[0.2, 0.0, 0.0]], dtype=np.float64)
    tgt = np.array([[-0.2, 0.0, 0.0]], dtype=np.float64)
    rgb = align.rasterize_point_clouds(
        [(tgt, (0.0, 0.0, 1.0)), (src, (1.0, 0.0, 0.0))],
        eye=(0.0, -2.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        width=80,
        height=60,
        point_px=3,
    )
    assert rgb.shape == (60, 80, 3)
    assert rgb.dtype == np.uint8
    assert (rgb[:, :, 0] > 200).any()
    assert (rgb[:, :, 2] > 200).any()


def test_save_alignment_views_writes_compare_and_viewpoints(tmp_path: Path):
    source = _box_points()
    target = source + np.array([0.0, 0.0, 0.0])
    after = source + np.array([0.4, 0.0, 0.0])
    paths = align.save_alignment_views(
        source,
        after,
        target,
        tmp_path,
        max_points=200,
        width=64,
        height=48,
    )
    names = {path.name for path in paths}
    assert names == {
        "compare.png",
        "iso.png",
        "front.png",
        "side.png",
        "top.png",
    }
    for path in paths:
        assert path.is_file()
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    compare = tmp_path / "compare.png"
    iso = tmp_path / "iso.png"
    # compare is 2 rows x 4 views; each view png is before|after.
    assert compare.stat().st_size > iso.stat().st_size


def test_inspect_alignment_cloud_xyz_subsamples():
    import inspect_alignment as viz

    pts = np.arange(300, dtype=np.float64).reshape(100, 3)
    out = viz.cloud_xyz(pts, max_points=10, seed=0)
    assert out.shape == (10, 3)
    assert out.dtype == np.float32


def test_inspect_alignment_cloud_xyz_from_mesh_vertices():
    import inspect_alignment as viz

    class _Mesh:
        def __init__(self) -> None:
            self.vertices = np.arange(30, dtype=np.float64).reshape(10, 3)
            self.triangles = np.array([[0, 1, 2]], dtype=np.int32)

    out = viz.cloud_xyz(_Mesh(), max_points=4, seed=0)
    assert out.shape == (4, 3)
    assert out.dtype == np.float32


def test_inspect_alignment_mesh_arrays_caps_triangles():
    import inspect_alignment as viz

    class _Mesh:
        def __init__(self) -> None:
            self.vertices = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            self.triangles = np.array(
                [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                dtype=np.int32,
            )

    verts, faces, colors = viz.mesh_arrays(_Mesh(), max_triangles=2, seed=0)
    assert verts.dtype == np.float32
    assert faces is not None
    assert faces.shape == (2, 3)
    assert faces.dtype == np.uint32
    assert verts.shape[0] <= 4
    assert colors is None
    full_v, full_f, full_c = viz.mesh_arrays(_Mesh(), max_triangles=None)
    assert full_f is not None
    assert full_f.shape == (4, 3)
    assert full_v.shape[0] == 4
    assert full_c is None


def test_crop_mesh_xy_keeps_only_box():
    import inspect_alignment as viz

    class _Mesh:
        def __init__(self) -> None:
            self.vertices = np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.5, 2.0, 0.0],
                    [0.2, 1.5, 1.0],
                    [10.0, 1.0, 0.0],
                    [10.5, 2.0, 0.0],
                    [10.2, 1.5, 0.0],
                ],
                dtype=np.float64,
            )
            self.triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    cropped = viz.crop_mesh_xy(
        _Mesh(), x_range=(-3.0, 1.0), y_range=(0.0, 5.0)
    )
    assert len(cropped.triangles) == 1
    np.testing.assert_allclose(cropped.vertices[0], [0.0, 1.0, 0.0])


def test_crop_mesh_xy_any_keeps_straddling_triangle():
    import inspect_alignment as viz

    class _Mesh:
        def __init__(self) -> None:
            self.vertices = np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.5, 2.0, 0.0],
                    [10.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            )
            self.triangles = np.array([[0, 1, 2]], dtype=np.int32)

    tight = viz.crop_mesh_xy(_Mesh(), x_range=(-3.0, 1.0), y_range=(0.0, 5.0))
    assert len(tight.triangles) == 0
    loose = viz.crop_mesh_xy(
        _Mesh(),
        x_range=(-3.0, 1.0),
        y_range=(0.0, 5.0),
        require_all=False,
    )
    assert len(loose.triangles) == 1


def test_crop_mesh_xy_preserves_vertex_colors():
    import inspect_alignment as viz

    class _Mesh:
        def __init__(self) -> None:
            self.vertices = np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.5, 2.0, 0.0],
                    [0.2, 1.5, 1.0],
                    [10.0, 1.0, 0.0],
                    [10.5, 2.0, 0.0],
                    [10.2, 1.5, 0.0],
                ],
                dtype=np.float64,
            )
            self.triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
            self.vertex_colors = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )

    cropped = viz.crop_mesh_xy(
        _Mesh(), x_range=(-3.0, 1.0), y_range=(0.0, 5.0)
    )
    assert cropped.vertex_colors is not None
    assert cropped.vertex_colors.shape == (3, 3)
    np.testing.assert_allclose(cropped.vertex_colors[0], [1.0, 0.0, 0.0])


def test_crop_points_xy_keeps_only_box():
    import inspect_alignment as viz

    pts = np.array(
        [
            [0.0, 1.0, 9.0],
            [10.0, 1.0, 0.0],
            [-2.5, 4.0, -1.0],
        ],
        dtype=np.float64,
    )
    kept = viz.crop_points_xy(pts, x_range=(-3.0, 1.0), y_range=(0.0, 5.0))
    np.testing.assert_allclose(
        kept, np.array([[0.0, 1.0, 9.0], [-2.5, 4.0, -1.0]])
    )
    kept_z = viz.crop_points_xy(
        pts,
        x_range=(-3.0, 1.0),
        y_range=(0.0, 5.0),
        z_range=(-2.0, 2.0),
    )
    np.testing.assert_allclose(kept_z, np.array([[-2.5, 4.0, -1.0]]))


def test_axis_flip_matrix_negates_z():
    import inspect_alignment as viz

    matrix = viz.axis_flip_matrix([False, False, True])
    point = np.array([1.0, 2.0, 3.0, 1.0])
    out = matrix @ point
    np.testing.assert_allclose(out[:3], [1.0, 2.0, -3.0])


@pytest.mark.parametrize(
    "source_flip",
    [
        [True, False, False],
        [False, True, False],
        [False, False, True],
        [False, True, True],
        [True, True, True],
    ],
)
def test_apply_manual_alignment_ignores_source_flip_for_command(source_flip):
    import inspect_alignment as viz

    class _Mesh:
        def __init__(self, xyz: list[float]) -> None:
            self.vertices = np.array([xyz], dtype=np.float64)
            self.triangles = np.zeros((0, 3), dtype=np.int32)

        def transform(self, matrix: np.ndarray) -> None:
            homo = np.c_[self.vertices, np.ones((len(self.vertices), 1))]
            self.vertices = (matrix @ homo.T).T[:, :3]

    native = _Mesh([0.0, 1.0, 0.0])
    bundle = {
        "source_mesh": _Mesh([0.0, 0.0, 1.0]),
        "source_mesh_native": native,
        "mesh": "dummy.glb",
        "target": "dummy.usd",
    }
    out = viz.apply_manual_alignment(
        bundle,
        np.eye(4),
        source_flip_axis=source_flip,
        gt_flip_axis=[False, False, False],
    )
    assert out["source_flip_axis"] == tuple(source_flip)
    assert out["scale"] >= 0.0
    assert float(np.linalg.det(out["rotation"])) == pytest.approx(1.0)
    rx, ry, rz = align.rotation_to_xyz_deg(out["rotation"])
    assert rx == pytest.approx(90.0)
    assert ry == pytest.approx(0.0)
    assert rz == pytest.approx(0.0)
    np.testing.assert_allclose(
        out["transformed_mesh"].vertices[0], [0.0, 0.0, 1.0]
    )


def test_apply_manual_alignment_applies_command_to_native_cloud():
    import inspect_alignment as viz

    class _Mesh:
        def __init__(self, xyz: list[float]) -> None:
            self.vertices = np.array([xyz], dtype=np.float64)
            self.triangles = np.zeros((0, 3), dtype=np.int32)

        def transform(self, matrix: np.ndarray) -> None:
            homo = np.c_[self.vertices, np.ones((len(self.vertices), 1))]
            self.vertices = (matrix @ homo.T).T[:, :3]

    native = _Mesh([0.0, 1.0, 0.0])
    bundle = {
        "source_mesh": _Mesh([0.0, 0.0, 1.0]),
        "source_mesh_native": native,
        "mesh": "dummy.glb",
        "target": "dummy.usd",
    }
    out = viz.apply_manual_alignment(bundle, np.eye(4))
    assert out["scale"] == pytest.approx(1.0)
    assert out["scale"] >= 0.0
    rx, ry, rz = align.rotation_to_xyz_deg(out["rotation"])
    assert rx == pytest.approx(90.0)
    assert ry == pytest.approx(0.0)
    assert rz == pytest.approx(0.0)
    np.testing.assert_allclose(
        out["transformed_mesh"].vertices[0], [0.0, 0.0, 1.0]
    )


def test_apply_command_alignment_uses_teleop_flags():
    import inspect_alignment as viz

    class _Mesh:
        def __init__(self, xyz: list[float]) -> None:
            self.vertices = np.array([xyz], dtype=np.float64)
            self.triangles = np.zeros((0, 3), dtype=np.int32)

        def transform(self, matrix: np.ndarray) -> None:
            homo = np.c_[self.vertices, np.ones((len(self.vertices), 1))]
            self.vertices = (matrix @ homo.T).T[:, :3]

    native = _Mesh([0.0, 1.0, 0.0])
    bundle = {
        "source_mesh": _Mesh([0.0, 0.0, 1.0]),
        "source_mesh_native": native,
        "mesh": "dummy.glb",
        "target": "dummy.usd",
    }
    out = viz.apply_command_alignment(
        bundle, "--xyz-deg 90 0 0 --xyz 0 0 0 --scale 1"
    )
    assert out["scale"] == pytest.approx(1.0)
    rx, ry, rz = align.rotation_to_xyz_deg(out["rotation"])
    assert rx == pytest.approx(90.0)
    assert ry == pytest.approx(0.0)
    assert rz == pytest.approx(0.0)
    np.testing.assert_allclose(
        out["transformed_mesh"].vertices[0], [0.0, 0.0, 1.0], atol=1e-9
    )


def test_picker_points_samples_more_than_vertices():
    o3d = _need_open3d()
    import inspect_alignment as viz

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [
            [0.0, 1.0, 0.0],
            [0.8, 2.0, 0.0],
            [0.2, 1.2, 1.0],
            [20.0, 20.0, 0.0],
            [21.0, 20.0, 0.0],
            [20.0, 21.0, 0.0],
        ]
    )
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5]])
    box_x = (-3.0, 1.0)
    box_y = (0.0, 5.0)
    src, gt = viz.picker_points(
        {"source_mesh": mesh, "gt_mesh": mesh},
        max_points=40,
        crop_x=box_x,
        crop_y=box_y,
        crop_z=(-10.0, 10.0),
        source_crop_x=box_x,
        source_crop_y=box_y,
        source_crop_z=(-10.0, 10.0),
        crop_oversample=2,
    )
    assert src.xyz.shape[1] == 3
    assert gt.xyz.shape[1] == 3
    assert len(src.xyz) > 3
    assert len(gt.xyz) > 3
    assert len(src.xyz) <= 40
    assert len(gt.xyz) <= 40
    assert src.xyz[:, 0].max() <= box_x[1] + 1e-6
    assert src.xyz[:, 1].max() <= box_y[1] + 1e-6
    assert src.xyz[:, 2].max() <= 10.0 + 1e-6
    assert gt.xyz[:, 0].max() <= box_x[1] + 1e-6
    assert gt.xyz[:, 1].max() <= box_y[1] + 1e-6
    assert gt.xyz[:, 2].max() <= 10.0 + 1e-6
    assert src.rgb is None
    assert gt.rgb is None


def test_picker_points_keeps_vertex_colors():
    o3d = _need_open3d()
    import inspect_alignment as viz

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [
            [0.0, 1.0, 0.0],
            [0.8, 2.0, 0.0],
            [0.2, 1.2, 1.0],
        ]
    )
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    src, gt = viz.picker_points(
        {"source_mesh": mesh, "gt_mesh": mesh},
        max_points=20,
        crop_x=(-3.0, 1.0),
        crop_y=(0.0, 5.0),
        crop_z=(-10.0, 10.0),
        source_crop_x=(-3.0, 1.0),
        source_crop_y=(0.0, 5.0),
        source_crop_z=(-10.0, 10.0),
        crop_oversample=2,
    )
    assert src.rgb is not None
    assert gt.rgb is not None
    assert src.rgb.shape == src.xyz.shape
    assert gt.rgb.shape == gt.xyz.shape
    assert src.rgb.min() >= 0.0
    assert src.rgb.max() <= 1.0


def test_similarity_from_pairs_recovers_known_transform():
    _need_open3d()
    import inspect_alignment as viz

    rng = np.random.default_rng(7)
    source = rng.normal(size=(5, 3))
    scale = 1.37
    rotation = _rotz(35.0)
    translation = np.array([0.4, -0.2, 1.1], dtype=np.float64)
    target = (scale * (rotation @ source.T).T) + translation
    result = viz.similarity_from_pairs(source, target)
    assert result["n_pairs"] == 5
    assert result["scale"] == pytest.approx(scale, rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(result["rotation"], rotation, atol=1e-6)
    np.testing.assert_allclose(result["translation"], translation, atol=1e-6)
    np.testing.assert_allclose(
        result["transformation"],
        align.compose_similarity(scale, rotation, translation),
        atol=1e-6,
    )
    assert result["rmse"] == pytest.approx(0.0, abs=1e-8)


def test_similarity_from_pairs_requires_three_points():
    _need_open3d()
    import inspect_alignment as viz

    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="at least 3"):
        viz.similarity_from_pairs(src, src)


def test_overlay_returns_plotly_figure():
    pytest.importorskip("plotly")
    import inspect_alignment as viz

    mesh = viz._CroppedMesh(
        np.array(
            [[0.0, 1.0, 0.0], [0.8, 2.0, 0.0], [0.2, 1.2, 1.0]],
            dtype=np.float64,
        ),
        np.array([[0, 1, 2]], dtype=np.int64),
    )
    fig = viz.overlay(mesh, mesh, mesh, show_gt=True, max_triangles=10)
    assert len(fig.data) == 3
    assert fig.data[0].name == "gt"
    assert fig.data[2].name == "transformed"
    alias = viz.k3d_overlay(mesh, mesh, mesh, show_gt=True, max_triangles=10)
    assert len(alias.data) == 3
    assert fig.layout.scene.xaxis.range is not None


def test_triangle_chunks_splits_over_webgl_limit():
    import inspect_alignment as viz

    faces = np.array(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        dtype=np.uint32,
    )
    chunks = viz._triangle_chunks(faces, n_vertices=9, max_vertices=6)
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 1
    one = viz._triangle_chunks(faces, n_vertices=6, max_vertices=6)
    assert len(one) == 1
