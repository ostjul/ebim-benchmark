# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Plotly helpers for inspecting room alignment.

Picker and overlay use mesh vertex colors when present (reconstructed
GLB usually has them; CAD falls back to displayColor / gray). Overlay
defaults to a pasteable ``--xyz-deg`` / ``--xyz`` / ``--scale`` command
applied to the native cloud. Clicking correspondences is optional: a
7-DoF similarity (Umeyama) is fit to those pairs. Picker axis flips
are display only; a picker-derived teleop command is the fit plus GLB
Y-up→Z-up Rx(+90°). Plotly figures work in VS Code / Cursor Jupyter;
k3d's widget JS is not registered there.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import align_automatic as align
import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[2]
GT_USD = REPO_ROOT / "assets" / "robot_room.usd"
K3D_BLUE = 0x377EB8
K3D_RED = 0xE41A1C
K3D_GREEN = 0x4DAF4A
DEFAULT_VIZ_POINTS = 40_000
DEFAULT_VIZ_TRIANGLES = 80_000
# Plotly Mesh3d / WebGL 1 uses 16-bit indices; extra verts are dropped.
MAX_MESH3D_VERTICES = 65_535
DEFAULT_GT_CROP_X = (-3.0, 1.0)
DEFAULT_GT_CROP_Y = (0.0, 5.0)
DEFAULT_CROP_Z = (-10.0, 10.0)
DEFAULT_FLIP_AXIS = (False, False, False)


class _CroppedMesh:
    """Lightweight triangle mesh for overlay (numpy vertices / faces)."""

    def __init__(
        self,
        vertices: npt.NDArray[np.float64],
        triangles: npt.NDArray[np.int64],
        vertex_colors: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.vertices = vertices
        self.triangles = triangles
        self.vertex_colors = vertex_colors


class PickerCloud(NamedTuple):
    """Sampled picker points and optional per-point RGB in ``[0, 1]``."""

    xyz: npt.NDArray[np.float32]
    rgb: npt.NDArray[np.float32] | None = None


def _geometry_xyz(obj: Any) -> npt.NDArray[np.float64]:
    """Nx3 positions from an array, mesh vertices, or cloud points."""
    if isinstance(obj, np.ndarray):
        points = np.asarray(obj, dtype=np.float64)
    elif getattr(obj, "vertices", None) is not None:
        points = np.asarray(obj.vertices, dtype=np.float64)
    else:
        points = np.asarray(obj.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected Nx3 points, got shape {points.shape}")
    return points


def _mesh_rgb(mesh: Any) -> npt.NDArray[np.float64] | None:
    """Per-vertex RGB in ``[0, 1]``, or ``None`` if the mesh has no colors."""
    if hasattr(mesh, "has_vertex_colors") and mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    else:
        raw = getattr(mesh, "vertex_colors", None)
        if raw is None:
            return None
        colors = np.asarray(raw, dtype=np.float64)
    if colors.size == 0:
        return None
    colors = colors.reshape(-1, 3)
    n_verts = len(np.asarray(mesh.vertices))
    if len(colors) != n_verts:
        return None
    return colors


def _as_flip_axis(flip: Sequence[bool] | None) -> tuple[bool, bool, bool]:
    if flip is None:
        flags: tuple[bool, ...] = (False, False, False)
    else:
        flags = tuple(bool(v) for v in flip)
    if len(flags) != 3:
        raise ValueError(f"flip_axis must have length 3, got {len(flags)}")
    return flags[0], flags[1], flags[2]


def axis_flip_matrix(
    flip: Sequence[bool] | None,
) -> npt.NDArray[np.float64]:
    """4x4 that negates selected XYZ axes (an involution)."""
    flags = _as_flip_axis(flip)
    matrix = np.eye(4, dtype=np.float64)
    for i, flag in enumerate(flags):
        if flag:
            matrix[i, i] = -1.0
    return matrix


def flip_mesh(mesh: Any, flip: Sequence[bool] | None) -> Any:
    """Copy *mesh* and negate selected axes. No-op when all flags are false."""
    flags = _as_flip_axis(flip)
    if not any(flags):
        return mesh
    matrix = axis_flip_matrix(flags)
    if hasattr(mesh, "transform"):
        out = copy.deepcopy(mesh)
        out.transform(matrix)
        return out
    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    verts = (matrix[:3, :3] @ verts.T).T
    colors = _mesh_rgb(mesh)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    return _CroppedMesh(verts, tris, colors)


def _box_inside(
    points: npt.ArrayLike,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
) -> npt.NDArray[np.bool_]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mask = np.ones(len(pts), dtype=bool)
    ranges = ((0, x_range), (1, y_range), (2, z_range))
    for axis, bounds in ranges:
        if bounds is None:
            continue
        lo, hi = (float(bounds[0]), float(bounds[1]))
        mask &= (pts[:, axis] >= lo) & (pts[:, axis] <= hi)
    return mask


def _xy_inside(
    points: npt.ArrayLike,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> npt.NDArray[np.bool_]:
    return _box_inside(points, x_range, y_range, None)


def cloud_xyz(
    cloud: Any,
    max_points: int = DEFAULT_VIZ_POINTS,
    seed: int = 0,
) -> npt.NDArray[np.float32]:
    """Nx3 float32 positions, optionally subsampled for the overlay."""
    points = _geometry_xyz(cloud)
    rng = np.random.default_rng(seed)
    return np.asarray(
        align._subsample_xyz(points, max_points, rng),
        dtype=np.float32,
    )


def mesh_arrays(
    mesh: Any,
    max_triangles: int | None = None,
    seed: int = 0,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.uint32] | None,
    npt.NDArray[np.float32] | None,
]:
    """Vertices, triangle indices, and optional vertex RGB for overlay meshes.

    If the mesh has no faces, indices are ``None`` (caller should use points).
    Pass ``max_triangles`` to subsample; ``None`` keeps every face.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    colors = _mesh_rgb(mesh)
    if triangles.size == 0:
        rgb = None if colors is None else colors.astype(np.float32)
        return vertices.astype(np.float32), None, rgb
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"expected Nx3 triangles, got {triangles.shape}")
    if max_triangles is not None and len(triangles) > max_triangles:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(triangles), int(max_triangles), replace=False)
        triangles = triangles[keep]
        used = np.unique(triangles.ravel())
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used), dtype=np.int64)
        vertices = vertices[used]
        triangles = remap[triangles]
        if colors is not None:
            colors = colors[used]
    rgb = None if colors is None else colors.astype(np.float32)
    return vertices.astype(np.float32), triangles.astype(np.uint32), rgb


def crop_mesh_xy(
    mesh: Any,
    x_range: tuple[float, float] | None = DEFAULT_GT_CROP_X,
    y_range: tuple[float, float] | None = DEFAULT_GT_CROP_Y,
    z_range: tuple[float, float] | None = None,
    *,
    require_all: bool = True,
) -> Any:
    """Keep triangles that lie in the axis-aligned crop box.

    ``None`` on an axis means that axis is not filtered. Z is unrestricted
    unless *z_range* is set. ``require_all=True`` keeps a triangle only if
    every vertex is inside (tight overlay crop). ``False`` keeps straddling
    faces so large CAD walls still contribute picker samples.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    colors = _mesh_rgb(mesh)
    if vertices.size == 0:
        return mesh
    inside = _box_inside(vertices, x_range, y_range, z_range)
    if triangles.size == 0:
        kept_v = vertices[inside]
        kept_c = None if colors is None else colors[inside]
        return _CroppedMesh(kept_v, triangles, kept_c)
    if require_all:
        keep = inside[triangles].all(axis=1)
    else:
        keep = inside[triangles].any(axis=1)
    triangles = triangles[keep]
    if len(triangles) == 0:
        empty_c = None if colors is None else colors[:0]
        return _CroppedMesh(vertices[:0], triangles, empty_c)
    used = np.unique(triangles.ravel())
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    kept_c = None if colors is None else colors[used]
    return _CroppedMesh(vertices[used], remap[triangles], kept_c)


def transform_mesh(mesh: Any, transformation: npt.ArrayLike) -> Any:
    """Copy *mesh* and apply a 4x4 Open3D (column-vector) transform."""
    out = copy.deepcopy(mesh)
    out.transform(np.asarray(transformation, dtype=np.float64))
    return out


def crop_points_xy(
    points: npt.ArrayLike,
    x_range: tuple[float, float] | None = DEFAULT_GT_CROP_X,
    y_range: tuple[float, float] | None = DEFAULT_GT_CROP_Y,
    z_range: tuple[float, float] | None = None,
) -> npt.NDArray[np.float64]:
    """Keep points inside the axis-aligned crop box."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return pts[_box_inside(pts, x_range, y_range, z_range)]


def load_meshes(
    mesh: Path | str,
    target: Path | str | None = None,
) -> dict[str, Any]:
    """Load source and GT triangle meshes (no ICP).

    ``source_mesh`` is Z-up (glTF Y-up baked) for picking against CAD.
    ``source_mesh_native`` is the original cloud the teleop command
    applies to.
    """
    mesh_path = align._require_geometry_path(Path(mesh))
    target_path = align._require_geometry_path(Path(target or GT_USD))
    return {
        "mesh": mesh_path,
        "target": target_path,
        "source_mesh": align.load_triangle_mesh(mesh_path),
        "source_mesh_native": align.load_triangle_mesh(
            mesh_path, y_up_to_z_up=False
        ),
        "gt_mesh": align.load_triangle_mesh(target_path),
    }


def _as_open3d_mesh(mesh: Any) -> Any:
    """Return an Open3D triangle mesh, copying numpy ``_CroppedMesh`` data."""
    o3d = align._require_open3d()
    if hasattr(mesh, "sample_points_uniformly"):
        return mesh
    out = o3d.geometry.TriangleMesh()
    verts = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
    out.vertices = o3d.utility.Vector3dVector(verts)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    if tris.size:
        out.triangles = o3d.utility.Vector3iVector(
            np.ascontiguousarray(tris.reshape(-1, 3), dtype=np.int32)
        )
    colors = _mesh_rgb(mesh)
    if colors is not None:
        out.vertex_colors = o3d.utility.Vector3dVector(colors)
    return out


def surface_xyz(
    mesh: Any,
    max_points: int = DEFAULT_VIZ_POINTS,
    seed: int = 0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]:
    """Uniform surface samples; vertices only when the mesh has no faces."""
    o3d_mesh = _as_open3d_mesh(mesh)
    if len(o3d_mesh.triangles) == 0:
        points = _geometry_xyz(o3d_mesh)
        colors = _mesh_rgb(o3d_mesh)
        rng = np.random.default_rng(seed)
        if max_points >= 1 and len(points) > max_points:
            index = rng.choice(len(points), int(max_points), replace=False)
            points = points[index]
            if colors is not None:
                colors = colors[index]
        xyz = np.asarray(points, dtype=np.float32)
        rgb = None if colors is None else colors.astype(np.float32)
        return xyz, rgb
    n_points = max(int(max_points), 1)
    cloud = align.sample_mesh_to_point_cloud(o3d_mesh, n_points=n_points)
    xyz = np.asarray(cloud.points, dtype=np.float32)
    rgb = None
    if cloud.has_colors():
        rgb = np.asarray(cloud.colors, dtype=np.float32)
    return xyz, rgb


def _cropped_surface_xyz(
    mesh: Any,
    *,
    max_points: int,
    crop_x: tuple[float, float] | None,
    crop_y: tuple[float, float] | None,
    crop_z: tuple[float, float] | None,
    flip_axis: Sequence[bool] | None,
    crop_oversample: int,
    seed: int,
) -> PickerCloud:
    """Uniform surface samples, optionally flipped and XYZ-cropped."""
    n_points = int(max_points)
    work = flip_mesh(mesh, flip_axis)
    do_crop = crop_x is not None or crop_y is not None or crop_z is not None
    if do_crop:
        work = crop_mesh_xy(
            work,
            crop_x,
            crop_y,
            crop_z,
            require_all=False,
        )
        n_points = int(max_points) * max(int(crop_oversample), 1)
    xyz, rgb = surface_xyz(work, max_points=n_points, seed=seed)
    if do_crop:
        mask = _box_inside(xyz, crop_x, crop_y, crop_z)
        xyz = xyz[mask]
        if rgb is not None:
            rgb = rgb[mask]
        rng = np.random.default_rng(seed + 1)
        index = np.arange(len(xyz))
        if max_points >= 1 and len(xyz) > max_points:
            index = rng.choice(len(xyz), int(max_points), replace=False)
        xyz = xyz[index]
        if rgb is not None:
            rgb = rgb[index]
    return PickerCloud(
        np.asarray(xyz, dtype=np.float32),
        None if rgb is None else np.asarray(rgb, dtype=np.float32),
    )


def picker_points(
    bundle: dict[str, Any],
    *,
    max_points: int = 20_000,
    crop_x: tuple[float, float] | None = DEFAULT_GT_CROP_X,
    crop_y: tuple[float, float] | None = DEFAULT_GT_CROP_Y,
    crop_z: tuple[float, float] | None = DEFAULT_CROP_Z,
    source_crop_x: tuple[float, float] | None = DEFAULT_GT_CROP_X,
    source_crop_y: tuple[float, float] | None = DEFAULT_GT_CROP_Y,
    source_crop_z: tuple[float, float] | None = DEFAULT_CROP_Z,
    gt_flip_axis: Sequence[bool] | None = DEFAULT_FLIP_AXIS,
    source_flip_axis: Sequence[bool] | None = DEFAULT_FLIP_AXIS,
    crop_oversample: int = 8,
    seed: int = 0,
) -> tuple[PickerCloud, PickerCloud]:
    """Subsampled XYZ (+ RGB) for the Plotly picker.

    Uses uniform surface samples so CAD walls/floors are pickable, not
    only unique vertices. Optional axis flips run first, then XYZ crops
    keep straddling triangles, sample extra points, and keep those in
    the box. Flips are display-only; ``apply_manual_alignment`` does
    not bake them into the teleop command.
    """
    source = _cropped_surface_xyz(
        bundle["source_mesh"],
        max_points=max_points,
        crop_x=source_crop_x,
        crop_y=source_crop_y,
        crop_z=source_crop_z,
        flip_axis=source_flip_axis,
        crop_oversample=crop_oversample,
        seed=seed,
    )
    gt = _cropped_surface_xyz(
        bundle["gt_mesh"],
        max_points=max_points,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_z=crop_z,
        flip_axis=gt_flip_axis,
        crop_oversample=crop_oversample,
        seed=seed + 2,
    )
    return source, gt


def similarity_from_pairs(
    source_xyz: npt.ArrayLike,
    target_xyz: npt.ArrayLike,
) -> dict[str, Any]:
    """7-DoF similarity (Umeyama) from N corresponding points, N >= 3."""
    src = np.asarray(source_xyz, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target_xyz, dtype=np.float64).reshape(-1, 3)
    if src.shape != tgt.shape:
        raise ValueError(
            f"source/target pair counts differ: {src.shape} vs {tgt.shape}"
        )
    if len(src) < 3:
        raise ValueError("need at least 3 correspondences")
    o3d = align._require_open3d()
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(src)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(tgt)
    corr = o3d.utility.Vector2iVector([[i, i] for i in range(len(src))])
    estimation = (
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True
        )
    )
    matrix = np.asarray(
        estimation.compute_transformation(source, target, corr),
        dtype=np.float64,
    )
    scale, rotation, translation = align.decompose_similarity(matrix)
    aligned = (scale * (rotation @ src.T).T) + translation
    residual = aligned - tgt
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return {
        "transformation": matrix,
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "rmse": rmse,
        "n_pairs": int(len(src)),
        "source_xyz": src,
        "target_xyz": tgt,
    }


def apply_manual_alignment(
    bundle: dict[str, Any],
    transformation: npt.ArrayLike,
    *,
    source_flip_axis: Sequence[bool] | None = DEFAULT_FLIP_AXIS,
    gt_flip_axis: Sequence[bool] | None = DEFAULT_FLIP_AXIS,
) -> dict[str, Any]:
    """Copy *bundle* and attach the teleop command xform + transformed mesh.

    ``transformation`` is the Umeyama / ICP fit in the picker Z-up frame
    (after optional display flips). Picker X/Y/Z flips are not composed
    into the 4x4; they only change how the clouds are shown for clicking.
    The teleop command is
    ``T_fit @ native_to_aligned_frame`` (GLB Y-up→Z-up Rx(+90°)) so
    Isaac applies it to the original native cloud. Validation
    ``transformed_mesh`` is that command on ``source_mesh_native``.
    Scale is ``>= 0``.
    """
    t_fit = np.asarray(transformation, dtype=np.float64)
    scale, rotation, translation, command = (
        align.scene_similarity_from_alignment(t_fit, bundle["mesh"])
    )
    native = bundle.get("source_mesh_native", bundle["source_mesh"])
    out = dict(bundle)
    out["transformation"] = command
    out["align_transformation"] = t_fit
    out["fit_transformation"] = t_fit
    out["source_flip_axis"] = _as_flip_axis(source_flip_axis)
    out["gt_flip_axis"] = _as_flip_axis(gt_flip_axis)
    out["scale"] = scale
    out["rotation"] = rotation
    out["translation"] = translation
    out["transformed_mesh"] = transform_mesh(native, command)
    return out


def apply_command_alignment(
    bundle: dict[str, Any],
    flags: str | Sequence[str],
) -> dict[str, Any]:
    """Copy *bundle* and apply a teleop ``--xyz-deg`` / ``--xyz`` / ``--scale``.

    The flags are the Isaac command on the original native cloud (already
    including GLB Rx(+90°) when that is how they were produced). They are
    not composed with ``native_to_aligned_frame`` again.
    """
    scale, rotation, translation = align.parse_scene_room_flags(flags)
    command = align.compose_similarity(scale, rotation, translation)
    native = bundle.get("source_mesh_native", bundle["source_mesh"])
    out = dict(bundle)
    out["transformation"] = command
    out["scale"] = scale
    out["rotation"] = rotation
    out["translation"] = translation
    out["transformed_mesh"] = transform_mesh(native, command)
    return out


def _plotly_marker_color(
    color: str | npt.ArrayLike,
) -> str | list[str]:
    if isinstance(color, str):
        return color
    rgb = np.asarray(color, dtype=np.float64).reshape(-1, 3)
    u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.int32)
    return [f"rgb({r},{g},{b})" for r, g, b in u8]


def _hex_rgb(color: int) -> str:
    return f"rgb({(color >> 16) & 255},{(color >> 8) & 255},{color & 255})"


def _scatter3d(
    points: npt.NDArray[np.float32],
    *,
    color: str | npt.ArrayLike,
    name: str,
    size: float,
) -> Any:
    import plotly.graph_objects as go

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker={"size": size, "color": _plotly_marker_color(color)},
        name=name,
    )


def _require_plotly() -> Any:
    """Plotly 6 FigureWidget needs ``anywidget`` in the kernel."""
    try:
        import anywidget  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Plotly FigureWidget needs anywidget. In this kernel run: "
            "pip install anywidget"
        ) from exc
    import plotly.graph_objects as go

    if go.FigureWidget.__module__.endswith("missing_anywidget"):
        raise ImportError(
            "plotly was imported before anywidget was installed. "
            "Restart the kernel, then re-run from the import cell."
        )
    return go


def _plotly_layout(title: str) -> Any:
    go = _require_plotly()
    return go.Layout(
        title=title,
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        height=480,
        width=480,
    )


class CorrespondencePicker:
    """Click source then GT to store N point pairs."""

    def __init__(
        self,
        source_xyz: npt.ArrayLike,
        gt_xyz: npt.ArrayLike,
        *,
        source_rgb: npt.ArrayLike | None = None,
        gt_rgb: npt.ArrayLike | None = None,
    ) -> None:
        go = _require_plotly()
        from ipywidgets import HTML, Button, HBox, VBox

        self.source_pts = np.asarray(source_xyz, dtype=np.float32).reshape(
            -1, 3
        )
        self.gt_pts = np.asarray(gt_xyz, dtype=np.float32).reshape(-1, 3)
        self.pairs: list[tuple[int, int]] = []
        self.pending_src: int | None = None
        self.result: dict[str, Any] | None = None
        empty = np.zeros((0, 3), dtype=np.float32)
        src_color: str | npt.NDArray[np.float32] = "#e41a1c"
        gt_color: str | npt.NDArray[np.float32] = "#377eb8"
        if source_rgb is not None:
            src_color = np.asarray(source_rgb, dtype=np.float32).reshape(-1, 3)
        if gt_rgb is not None:
            gt_color = np.asarray(gt_rgb, dtype=np.float32).reshape(-1, 3)
        self.src_fig = go.FigureWidget(
            data=[
                _scatter3d(
                    self.source_pts, color=src_color, name="source", size=2
                ),
                _scatter3d(empty, color="#ff7f00", name="picked", size=6),
            ],
            layout=_plotly_layout("Source (click first)"),
        )
        self.gt_fig = go.FigureWidget(
            data=[
                _scatter3d(self.gt_pts, color=gt_color, name="gt", size=2),
                _scatter3d(empty, color="#ff7f00", name="picked", size=6),
            ],
            layout=_plotly_layout("GT (click second)"),
        )
        self.src_fig.data[0].on_click(self._on_source)
        self.gt_fig.data[0].on_click(self._on_gt)
        self._status = HTML(value=self._status_html())
        undo_btn = Button(description="Undo last")
        clear_btn = Button(description="Clear")
        solve_btn = Button(description="Solve")
        undo_btn.on_click(lambda _b: self.undo())
        clear_btn.on_click(lambda _b: self.clear())
        solve_btn.on_click(lambda _b: self.solve())
        self._box = VBox(
            [
                self._status,
                HBox([undo_btn, clear_btn, solve_btn]),
                HBox([self.src_fig, self.gt_fig]),
            ]
        )

    def _status_html(self) -> str:
        pending = (
            "pending SOURCE click"
            if self.pending_src is None
            else f"pending GT click (src #{self.pending_src})"
        )
        extra = ""
        if self.result is not None:
            extra = (
                f" | solved N={self.result['n_pairs']} "
                f"s={self.result['scale']:.4f} "
                f"rmse={self.result['rmse']:.4f}"
            )
        return (
            f"<b>{len(self.pairs)} pairs</b> — click SOURCE then GT "
            f"({pending}){extra}"
        )

    def _selected_xyz(
        self,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        empty = np.zeros((0, 3), dtype=np.float32)
        if not self.pairs:
            src_sel = empty
            gt_sel = empty
        else:
            src_sel = self.source_pts[[i for i, _j in self.pairs]]
            gt_sel = self.gt_pts[[j for _i, j in self.pairs]]
        if self.pending_src is not None:
            src_sel = np.vstack(
                [src_sel, self.source_pts[self.pending_src][None, :]]
            )
        return src_sel, gt_sel

    def _set_trace_xyz(self, trace: Any, pts: npt.NDArray[np.float32]) -> None:
        trace.x = pts[:, 0]
        trace.y = pts[:, 1]
        trace.z = pts[:, 2]

    def _refresh(self) -> None:
        src_sel, gt_sel = self._selected_xyz()
        with self.src_fig.batch_update():
            self._set_trace_xyz(self.src_fig.data[1], src_sel)
        with self.gt_fig.batch_update():
            self._set_trace_xyz(self.gt_fig.data[1], gt_sel)
        self._status.value = self._status_html()

    def _click_index(self, points: Any) -> int | None:
        inds = getattr(points, "point_inds", None)
        if not inds:
            return None
        return int(inds[0])

    def _on_source(self, _trace: Any, points: Any, _selector: Any) -> None:
        idx = self._click_index(points)
        if idx is None:
            return
        self.pending_src = idx
        self._refresh()

    def _on_gt(self, _trace: Any, points: Any, _selector: Any) -> None:
        if self.pending_src is None:
            return
        idx = self._click_index(points)
        if idx is None:
            return
        self.pairs.append((self.pending_src, idx))
        self.pending_src = None
        self.result = None
        self._refresh()

    def undo(self) -> None:
        if self.pending_src is not None:
            self.pending_src = None
        elif self.pairs:
            self.pairs.pop()
        self.result = None
        self._refresh()

    def clear(self) -> None:
        self.pairs.clear()
        self.pending_src = None
        self.result = None
        self._refresh()

    def pair_xyz(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        empty = np.zeros((0, 3), dtype=np.float64)
        if not self.pairs:
            return empty, empty
        src = np.stack([self.source_pts[i] for i, _j in self.pairs])
        tgt = np.stack([self.gt_pts[j] for _i, j in self.pairs])
        return src.astype(np.float64), tgt.astype(np.float64)

    def solve(self) -> dict[str, Any]:
        src, tgt = self.pair_xyz()
        self.result = similarity_from_pairs(src, tgt)
        self._status.value = self._status_html()
        print(
            f"similarity N={self.result['n_pairs']} "
            f"s={self.result['scale']:.4f} "
            f"rmse={self.result['rmse']:.4f} "
            f"t={self.result['translation']}"
        )
        return self.result

    def widget(self) -> Any:
        return self._box


def run_alignment(
    mesh: Path | str,
    target: Path | str | None = None,
    *,
    n_points: int = 80_000,
    voxel_sizes: Sequence[float] | None = None,
    max_correspondence_distances: Sequence[float] | None = None,
    estimate_scale: bool = True,
    max_iterations: int = 50,
) -> dict[str, Any]:
    """Load triangle meshes, sample for ICP, and transform the source mesh."""
    loaded = load_meshes(mesh, target)
    source_mesh = loaded["source_mesh"]
    gt_mesh = loaded["gt_mesh"]
    mesh_path = loaded["mesh"]
    target_path = loaded["target"]
    source = align.sample_mesh_to_point_cloud(source_mesh, n_points=n_points)
    gt = align.sample_mesh_to_point_cloud(gt_mesh, n_points=n_points)
    init = align.estimate_scale_init(source, gt) if estimate_scale else None
    result = align.multiscale_similarity_icp(
        source,
        gt,
        voxel_sizes=list(voxel_sizes or align.DEFAULT_VOXEL_SIZES),
        max_correspondence_distances=list(
            max_correspondence_distances or align.DEFAULT_MAX_CORR
        ),
        init=init,
        max_iterations=max_iterations,
    )
    scale, rotation, translation, command = (
        align.scene_similarity_from_alignment(
            result["transformation"], mesh_path
        )
    )
    native = loaded.get("source_mesh_native", source_mesh)
    transformed_mesh = transform_mesh(native, command)
    return {
        "mesh": mesh_path,
        "target": target_path,
        "source": source,
        "gt": gt,
        "source_mesh": source_mesh,
        "source_mesh_native": native,
        "gt_mesh": gt_mesh,
        "transformed_mesh": transformed_mesh,
        "result": result,
        "transformed": result["transformed_source"],
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "transformation": command,
        "align_transformation": result["transformation"],
    }


def _visible_scene_ranges(
    meshes: Sequence[Any],
) -> dict[str, dict[str, list[float]]] | None:
    """Axis ranges that enclose *meshes*, or ``None`` if they are empty."""
    mins: list[npt.NDArray[np.float64]] = []
    maxs: list[npt.NDArray[np.float64]] = []
    for mesh in meshes:
        verts = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
        if verts.size == 0:
            continue
        pts = verts.reshape(-1, 3)
        mins.append(pts.min(axis=0))
        maxs.append(pts.max(axis=0))
    if not mins:
        return None
    lo = np.min(np.stack(mins, axis=0), axis=0)
    hi = np.max(np.stack(maxs, axis=0), axis=0)
    pad = np.maximum((hi - lo) * 0.02, 1e-6)
    lo = lo - pad
    hi = hi + pad
    return {
        "xaxis": {"range": [float(lo[0]), float(hi[0])]},
        "yaxis": {"range": [float(lo[1]), float(hi[1])]},
        "zaxis": {"range": [float(lo[2]), float(hi[2])]},
    }


def _triangle_chunks(
    faces: npt.NDArray[np.uint32],
    n_vertices: int,
    max_vertices: int = MAX_MESH3D_VERTICES,
) -> list[npt.NDArray[np.int64]]:
    """Face index groups whose remapped vertex count fits WebGL Mesh3d."""
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if n_vertices <= max_vertices:
        return [faces]
    # Worst case: 3 unique verts per triangle.
    max_tris = max(1, int(max_vertices) // 3)
    return [faces[i : i + max_tris] for i in range(0, len(faces), max_tris)]


def _overlay_trace(
    mesh: Any,
    *,
    color: int,
    name: str,
    opacity: float,
    max_triangles: int | None,
    max_points: int,
    seed: int,
    visible: bool,
    point_size: float,
) -> list[Any]:
    import plotly.graph_objects as go

    vertices, faces, rgb = mesh_arrays(
        mesh, max_triangles=max_triangles, seed=seed
    )
    solid = _hex_rgb(color)
    marker_color = solid if rgb is None else _plotly_marker_color(rgb)
    if faces is None:
        if max_points >= 1 and len(vertices) > max_points:
            rng = np.random.default_rng(seed)
            keep = rng.choice(len(vertices), int(max_points), replace=False)
            vertices = vertices[keep]
            if isinstance(marker_color, list):
                marker_color = [marker_color[i] for i in keep]
        return [
            go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode="markers",
                marker={
                    "size": max(float(point_size) * 80.0, 1.5),
                    "color": marker_color,
                },
                name=name,
                visible=bool(visible),
                hoverinfo="skip",
                showlegend=True,
            )
        ]
    traces: list[Any] = []
    chunks = _triangle_chunks(faces, len(vertices))
    for i_chunk, chunk in enumerate(chunks):
        used = np.unique(chunk.ravel())
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used), dtype=np.int64)
        verts = vertices[used]
        faces_i = remap[chunk]
        if isinstance(marker_color, list):
            chunk_color: str | list[str] = [marker_color[int(j)] for j in used]
        else:
            chunk_color = marker_color
        kwargs: dict[str, Any] = {
            "x": verts[:, 0],
            "y": verts[:, 1],
            "z": verts[:, 2],
            "i": faces_i[:, 0],
            "j": faces_i[:, 1],
            "k": faces_i[:, 2],
            "opacity": float(opacity),
            "name": name,
            "legendgroup": name,
            "visible": bool(visible),
            "hoverinfo": "skip",
            "flatshading": True,
            "lighting": {"ambient": 0.65, "diffuse": 0.85},
            "showlegend": i_chunk == 0,
        }
        if rgb is not None:
            kwargs["vertexcolor"] = chunk_color
        else:
            kwargs["color"] = solid
        traces.append(go.Mesh3d(**kwargs))
    return traces


def overlay(
    gt: Any,
    source_before: Any,
    source_after: Any,
    *,
    point_size: float = 0.03,
    max_points: int = DEFAULT_VIZ_POINTS,
    max_triangles: int | None = None,
    opacity: float = 0.4,
    wireframe_gt: bool = True,
    show_gt: bool = False,
    show_previous: bool = False,
    show_transformed: bool = True,
    crop_gt: bool = True,
    crop_x: tuple[float, float] | None = DEFAULT_GT_CROP_X,
    crop_y: tuple[float, float] | None = DEFAULT_GT_CROP_Y,
    crop_z: tuple[float, float] | None = DEFAULT_CROP_Z,
    gt_flip_axis: Sequence[bool] | None = DEFAULT_FLIP_AXIS,
    source_flip_axis: Sequence[bool] | None = DEFAULT_FLIP_AXIS,
) -> Any:
    """Interactive overlay: mesh colors when present, else blue/red/green.

    Returns a Plotly ``Figure`` (not a Jupyter widget) so it renders in
    VS Code / Cursor. Prefers ``Mesh3d`` when objects have triangles.
    ``max_triangles=None`` keeps every face; pass an int to subsample.
    GT is XYZ-cropped by default. Camera bounds use visible layers so a
    hidden native cloud does not clip the transformed mesh. Final
    validation should pass the original native cloud as *source_before*
    / *source_after* without picker flips; the teleop command is already
    baked into *source_after*.
    """
    del wireframe_gt  # Plotly Mesh3d has no k3d-style wireframe.
    import plotly.graph_objects as go

    gt = flip_mesh(gt, gt_flip_axis)
    source_before = flip_mesh(source_before, source_flip_axis)
    do_crop = crop_x is not None or crop_y is not None or crop_z is not None
    if crop_gt and do_crop:
        gt = crop_mesh_xy(gt, crop_x, crop_y, crop_z)

    traces: list[Any] = []
    for mesh, color, name, shown, opac, seed in (
        (gt, K3D_BLUE, "gt", show_gt, min(opacity, 0.35), 0),
        (source_before, K3D_RED, "previous", show_previous, opacity, 1),
        (
            source_after,
            K3D_GREEN,
            "transformed",
            show_transformed,
            min(1.0, opacity + 0.15),
            2,
        ),
    ):
        traces.extend(
            _overlay_trace(
                mesh,
                color=color,
                name=name,
                opacity=opac,
                max_triangles=max_triangles,
                max_points=max_points,
                seed=seed,
                visible=shown,
                point_size=point_size,
            )
        )
    scene: dict[str, Any] = {"aspectmode": "data"}
    visible_meshes = [
        mesh
        for mesh, shown in (
            (gt, show_gt),
            (source_before, show_previous),
            (source_after, show_transformed),
        )
        if shown
    ]
    bounds = _visible_scene_ranges(visible_meshes)
    if bounds is not None:
        scene.update(bounds)
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Alignment overlay (legend toggles layers)",
        scene=scene,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        height=640,
        legend={"itemsizing": "constant"},
    )
    return fig


def k3d_overlay(*args: Any, **kwargs: Any) -> Any:
    """Deprecated alias for :func:`overlay` (Plotly, not k3d)."""
    return overlay(*args, **kwargs)


def teleop_command(bundle: dict[str, Any], room_usd: Path | str) -> str:
    """Pasteable ``run_isaacsim_teleop.sh`` line for this alignment."""
    return align.format_teleop_command(
        room_usd=Path(room_usd),
        mesh=Path(bundle["mesh"]),
        scale=float(bundle["scale"]),
        rotation=bundle["rotation"],
        translation=bundle["translation"],
    )
