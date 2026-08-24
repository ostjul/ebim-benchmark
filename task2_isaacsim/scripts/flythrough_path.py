# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Room flythrough poses: path-aligned ellipse plus inward/outward circles.

USD cameras look along local -Z. Horizontal look uses rotateXYZ Rx=90
(tilt from nadir to horizon) then Rz=atan2(-dx, dy) so -Z follows an XY
direction. Quaternion order matches USD R = Rz · Ry · Rx.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_EPS = 1e-9
_LOOK_RX_DEG = 90.0


@dataclass(frozen=True)
class CameraSpec:
    prim_path: str
    width: int
    height: int
    focal_length: float
    focus_distance: float
    projection: str


@dataclass(frozen=True)
class OrbitSpec:
    center_xy: tuple[float, float]
    diameter_x_m: float
    diameter_y_m: float
    height_m: float
    n_poses: int
    start_deg: float
    wave_amp_m: float
    wave_cycles: float

    @property
    def radius_x_m(self) -> float:
        return 0.5 * self.diameter_x_m

    @property
    def radius_y_m(self) -> float:
        return 0.5 * self.diameter_y_m


@dataclass(frozen=True)
class CircleSpec:
    look: str
    height_m: float
    radius_m: float
    n_poses: int
    start_deg: float


@dataclass(frozen=True)
class PanoramaSpec:
    prim_path: str
    translation: tuple[float, float, float]
    rotation_xyz_deg: tuple[float, float, float]
    width: int
    height: int
    projection: str


@dataclass(frozen=True)
class FlythroughConfig:
    camera: CameraSpec
    orbit: OrbitSpec
    circles: tuple[CircleSpec, ...]
    panorama: PanoramaSpec
    video_fps: float


@dataclass(frozen=True)
class FlythroughFrame:
    index: int
    xyz: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    rotation_xyz_deg: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    track: str = "orbit"
    look: str = "along_path"


def euler_xyz_to_quat(
    rotation_degrees: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """USD rotateXYZ quaternion (R = Rz · Ry · Rx), wxyz."""
    roll, pitch, yaw = (math.radians(angle) for angle in rotation_degrees)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def look_along_yaw_deg(dx: float, dy: float) -> float:
    """Yaw degrees so local -Z (after Rx=90) follows XY direction (dx, dy)."""
    if math.hypot(dx, dy) < _EPS:
        return 0.0
    return math.degrees(math.atan2(-dx, dy))


def look_along_rotation_xyz_deg(
    dx: float, dy: float, pitch_deg: float = 0.0
) -> tuple[float, float, float]:
    """Horizontal look (dx, dy) with extra pitch; +pitch looks world +Z."""
    return (_LOOK_RX_DEG + pitch_deg, 0.0, look_along_yaw_deg(dx, dy))


def rotate_vector_quat(
    quat_wxyz: tuple[float, float, float, float],
    vec: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a vector by a wxyz quaternion (q * v * q_conj)."""
    w, x, y, z = quat_wxyz
    vx, vy, vz = vec
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def frames_from_orbit(
    center_xy: tuple[float, float],
    radius_x_m: float,
    radius_y_m: float,
    height_m: float,
    n_poses: int,
    start_deg: float = 0.0,
    wave_amp_m: float = 0.0,
    wave_cycles: float = 4.0,
    track: str = "orbit",
    look: str = "along_path",
) -> list[FlythroughFrame]:
    """One CCW elliptical revolution, looking along the path tangent.

    XY is an axis-aligned ellipse (semi-axes radius_x_m, radius_y_m).
    Height follows a sine wave of amplitude wave_amp_m. Yaw and pitch
    follow the 3D tangent (true path slope), not the ellipse center.
    """
    if n_poses < 1:
        raise ValueError(f"n_poses must be >= 1, got {n_poses}")
    if radius_x_m <= 0.0:
        raise ValueError(f"radius_x_m must be positive, got {radius_x_m}")
    if radius_y_m <= 0.0:
        raise ValueError(f"radius_y_m must be positive, got {radius_y_m}")
    if wave_amp_m < 0.0:
        raise ValueError(f"wave_amp_m must be >= 0, got {wave_amp_m}")
    if wave_cycles < 0.0:
        raise ValueError(f"wave_cycles must be >= 0, got {wave_cycles}")
    cx, cy = (float(center_xy[0]), float(center_xy[1]))
    frames: list[FlythroughFrame] = []
    for index in range(int(n_poses)):
        theta = math.radians(start_deg) + 2.0 * math.pi * index / n_poses
        wave = wave_cycles * theta
        xyz = (
            cx + radius_x_m * math.cos(theta),
            cy + radius_y_m * math.sin(theta),
            height_m + wave_amp_m * math.sin(wave),
        )
        # Parametric tangent (CCW): dx/dθ, dy/dθ, dz/dθ.
        tx = -radius_x_m * math.sin(theta)
        ty = radius_y_m * math.cos(theta)
        tz = wave_amp_m * wave_cycles * math.cos(wave)
        xy_speed = math.hypot(tx, ty)
        pitch_deg = math.degrees(math.atan2(tz, max(xy_speed, _EPS)))
        rotation = look_along_rotation_xyz_deg(tx, ty, pitch_deg)
        frames.append(
            FlythroughFrame(
                index=index,
                xyz=xyz,
                yaw_deg=rotation[2],
                pitch_deg=pitch_deg,
                rotation_xyz_deg=rotation,
                quat_wxyz=euler_xyz_to_quat(rotation),
                track=track,
                look=look,
            )
        )
    return frames


def frames_from_circle(
    center_xy: tuple[float, float],
    radius_m: float,
    height_m: float,
    n_poses: int,
    look: str,
    start_deg: float = 0.0,
    track: str | None = None,
) -> list[FlythroughFrame]:
    """One CCW circle, looking horizontally inward or outward."""
    if look not in ("inward", "outward"):
        raise ValueError(f"look must be inward or outward, got {look!r}")
    if n_poses < 1:
        raise ValueError(f"n_poses must be >= 1, got {n_poses}")
    if radius_m <= 0.0:
        raise ValueError(f"radius_m must be positive, got {radius_m}")
    cx, cy = (float(center_xy[0]), float(center_xy[1]))
    name = track or f"circle_{look}_{height_m:.2f}m"
    frames: list[FlythroughFrame] = []
    sign = -1.0 if look == "inward" else 1.0
    for index in range(int(n_poses)):
        theta = math.radians(start_deg) + 2.0 * math.pi * index / n_poses
        xyz = (
            cx + radius_m * math.cos(theta),
            cy + radius_m * math.sin(theta),
            float(height_m),
        )
        dx = sign * (xyz[0] - cx)
        dy = sign * (xyz[1] - cy)
        rotation = look_along_rotation_xyz_deg(dx, dy, 0.0)
        frames.append(
            FlythroughFrame(
                index=index,
                xyz=xyz,
                yaw_deg=rotation[2],
                pitch_deg=0.0,
                rotation_xyz_deg=rotation,
                quat_wxyz=euler_xyz_to_quat(rotation),
                track=name,
                look=look,
            )
        )
    return frames


# Baked cubicle table in robot_room.usd sits at Task 2 XY
# (~1.25 x 0.74 x 0.75 m).
_TABLE_XY_DEFAULT = (2.05, 1.95)
_TABLE_XY_RADIUS_M = 0.30
_TABLE_HEIGHT_M = (0.50, 1.00)
_TABLE_FOOTPRINT_SHORT_M = (0.50, 1.00)
_TABLE_FOOTPRINT_LONG_M = (0.90, 1.70)


def is_baked_task2_table_bbox(
    mid_xy: tuple[float, float],
    size_xyz: tuple[float, float, float],
    table_xy: tuple[float, float] = _TABLE_XY_DEFAULT,
) -> bool:
    """True if a world AABB matches the baked Task 2 cubicle table."""
    dist = math.hypot(mid_xy[0] - table_xy[0], mid_xy[1] - table_xy[1])
    if dist > _TABLE_XY_RADIUS_M:
        return False
    short_xy, long_xy = sorted((abs(size_xyz[0]), abs(size_xyz[1])))
    height = abs(size_xyz[2])
    return (
        _TABLE_HEIGHT_M[0] < height < _TABLE_HEIGHT_M[1]
        and (
            _TABLE_FOOTPRINT_SHORT_M[0]
            < short_xy
            < _TABLE_FOOTPRINT_SHORT_M[1]
        )
        and _TABLE_FOOTPRINT_LONG_M[0] < long_xy < _TABLE_FOOTPRINT_LONG_M[1]
    )


def frames_from_config(config: FlythroughConfig) -> list[FlythroughFrame]:
    orbit = config.orbit
    chunks = [
        frames_from_orbit(
            orbit.center_xy,
            orbit.radius_x_m,
            orbit.radius_y_m,
            orbit.height_m,
            orbit.n_poses,
            orbit.start_deg,
            orbit.wave_amp_m,
            orbit.wave_cycles,
        )
    ]
    for circle in config.circles:
        chunks.append(
            frames_from_circle(
                orbit.center_xy,
                circle.radius_m,
                circle.height_m,
                circle.n_poses,
                circle.look,
                start_deg=circle.start_deg,
            )
        )
    frames: list[FlythroughFrame] = []
    for chunk in chunks:
        for frame in chunk:
            frames.append(replace(frame, index=len(frames)))
    return frames


_TRACK_COLORS = (
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#17becf",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#d62728",
)


def _group_tracks(
    frames: list[FlythroughFrame],
) -> list[tuple[str, list[FlythroughFrame]]]:
    groups: list[tuple[str, list[FlythroughFrame]]] = []
    for frame in frames:
        if not groups or groups[-1][0] != frame.track:
            groups.append((frame.track, [frame]))
        else:
            groups[-1][1].append(frame)
    return groups


def _track_legend(frames: list[FlythroughFrame]) -> str:
    look = frames[0].look
    if look == "along_path":
        return "orbit (along path)"
    height_m = frames[0].xyz[2]
    return f"{look} {height_m:.2f} m"


def plot_orbit_trajectory(
    frames: list[FlythroughFrame],
    out_path: Path,
    center_xy: tuple[float, float],
) -> Path:
    """Write a 3D trajectory figure (all tracks + look ticks)."""
    if not frames:
        raise ValueError("frames must not be empty")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to plot the orbit trajectory"
        ) from exc

    tracks = _group_tracks(frames)
    xs = [frame.xyz[0] for frame in frames]
    ys = [frame.xyz[1] for frame in frames]
    zs = [frame.xyz[2] for frame in frames]
    cx, cy = center_xy

    fig = plt.figure(figsize=(11.0, 5.5), dpi=120)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    look_len = 0.40
    for track_i, (_name, track_frames) in enumerate(tracks):
        color = _TRACK_COLORS[track_i % len(_TRACK_COLORS)]
        txs = [frame.xyz[0] for frame in track_frames]
        tys = [frame.xyz[1] for frame in track_frames]
        tzs = [frame.xyz[2] for frame in track_frames]
        ax3d.plot(
            txs + [txs[0]],
            tys + [tys[0]],
            tzs + [tzs[0]],
            color=color,
            linewidth=1.6,
            label=_track_legend(track_frames),
        )
        step = max(len(track_frames) // 12, 1)
        for frame in track_frames[::step]:
            look = rotate_vector_quat(frame.quat_wxyz, (0.0, 0.0, -1.0))
            ax3d.quiver(
                frame.xyz[0],
                frame.xyz[1],
                frame.xyz[2],
                look[0],
                look[1],
                look[2],
                length=look_len,
                normalize=True,
                color=color,
                arrow_length_ratio=0.25,
                linewidth=0.7,
            )
    ax3d.scatter(
        [frames[0].xyz[0]],
        [frames[0].xyz[1]],
        [frames[0].xyz[2]],
        color="#d62728",
        s=28,
        label="start",
        zorder=5,
    )
    ax3d.scatter(
        [cx],
        [cy],
        [sum(zs) / len(zs)],
        color="#111111",
        s=36,
        marker="x",
        label="center",
    )
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("Orbit trajectories (world m)")
    ax3d.legend(loc="upper left", fontsize=7)
    x_mid = 0.5 * (min(xs) + max(xs))
    y_mid = 0.5 * (min(ys) + max(ys))
    xy_span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-3)
    ax3d.set_xlim(x_mid - 0.55 * xy_span, x_mid + 0.55 * xy_span)
    ax3d.set_ylim(y_mid - 0.55 * xy_span, y_mid + 0.55 * xy_span)
    z_pad = max(0.15, 0.25 * (max(zs) - min(zs) + 1e-6))
    ax3d.set_zlim(min(zs) - z_pad, max(zs) + z_pad)

    axz = fig.add_subplot(2, 2, 2)
    axp = fig.add_subplot(2, 2, 4, sharex=axz)
    for track_i, (_name, track_frames) in enumerate(tracks):
        color = _TRACK_COLORS[track_i % len(_TRACK_COLORS)]
        n_poses = len(track_frames)
        thetas_deg = [360.0 * i / n_poses for i in range(n_poses)]
        axz.plot(
            thetas_deg,
            [frame.xyz[2] for frame in track_frames],
            color=color,
            label=_track_legend(track_frames),
        )
        axp.plot(
            thetas_deg,
            [frame.pitch_deg for frame in track_frames],
            color=color,
        )
    axz.set_ylabel("Z (m)")
    axz.set_title("Height and pitch vs orbit angle")
    axz.grid(True, alpha=0.3)
    axz.legend(fontsize=7, loc="best")
    axp.set_xlabel("Orbit angle (deg)")
    axp.set_ylabel("Pitch (deg)")
    axp.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _as_xy(value: Any, *, what: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{what} must be a list of 2 numbers")
    return (float(value[0]), float(value[1]))


def _as_xyz(value: Any, *, what: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{what} must be a list of 3 numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _as_hw(mapping: dict[str, Any], *, what: str) -> tuple[int, int]:
    try:
        width = int(mapping["width"])
        height = int(mapping["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{what} needs integer width and height") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"{what} width/height must be positive")
    return width, height


def load_flythrough_config(path: Path) -> FlythroughConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read flythrough config"
        ) from exc
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"flythrough config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")

    camera_raw = data.get("camera") or {}
    if not isinstance(camera_raw, dict):
        raise ValueError("camera must be a mapping")
    cam_w, cam_h = _as_hw(camera_raw, what="camera")
    camera = CameraSpec(
        prim_path=str(
            camera_raw.get("prim_path", "/World/Scene/flythrough_camera")
        ),
        width=cam_w,
        height=cam_h,
        focal_length=float(camera_raw.get("focal_length", 20.0)),
        focus_distance=float(camera_raw.get("focus_distance", 400.0)),
        projection=str(camera_raw.get("projection", "perspective")),
    )

    orbit_raw = data.get("orbit") or {}
    if not isinstance(orbit_raw, dict):
        raise ValueError("orbit must be a mapping")
    n_poses = int(orbit_raw.get("n_poses", 120))
    diameter_x_m = float(orbit_raw.get("diameter_x_m", 5.0))
    diameter_y_m = float(orbit_raw.get("diameter_y_m", 1.0))
    height_m = float(orbit_raw.get("height_m", 1.5))
    wave_amp_m = float(orbit_raw.get("wave_amp_m", 0.10))
    wave_cycles = float(orbit_raw.get("wave_cycles", 4.0))
    if n_poses < 1:
        raise ValueError("orbit.n_poses must be >= 1")
    if diameter_x_m <= 0.0:
        raise ValueError("orbit.diameter_x_m must be positive")
    if diameter_y_m <= 0.0:
        raise ValueError("orbit.diameter_y_m must be positive")
    if wave_amp_m < 0.0:
        raise ValueError("orbit.wave_amp_m must be >= 0")
    if wave_cycles < 0.0:
        raise ValueError("orbit.wave_cycles must be >= 0")
    orbit = OrbitSpec(
        center_xy=_as_xy(
            orbit_raw.get("center_xy", [2.05, 1.95]),
            what="orbit.center_xy",
        ),
        diameter_x_m=diameter_x_m,
        diameter_y_m=diameter_y_m,
        height_m=height_m,
        n_poses=n_poses,
        start_deg=float(orbit_raw.get("start_deg", 0.0)),
        wave_amp_m=wave_amp_m,
        wave_cycles=wave_cycles,
    )

    circles_raw = data.get("circles") or {}
    if not isinstance(circles_raw, dict):
        raise ValueError("circles must be a mapping")
    circle_radius_m = float(circles_raw.get("radius_m", 1.0))
    circle_n_poses = int(circles_raw.get("n_poses", n_poses))
    circle_start_deg = float(circles_raw.get("start_deg", 0.0))
    paths_raw = circles_raw.get("paths") or []
    if not isinstance(paths_raw, list):
        raise ValueError("circles.paths must be a list")
    if circle_radius_m <= 0.0:
        raise ValueError("circles.radius_m must be positive")
    if circle_n_poses < 1:
        raise ValueError("circles.n_poses must be >= 1")
    circles: list[CircleSpec] = []
    for i, path_raw in enumerate(paths_raw):
        if not isinstance(path_raw, dict):
            raise ValueError(f"circles.paths[{i}] must be a mapping")
        look = str(path_raw.get("look", "")).strip()
        if look not in ("inward", "outward"):
            raise ValueError(
                f"circles.paths[{i}].look must be inward or outward"
            )
        height_m = float(path_raw["height_m"])
        radius_m = float(path_raw.get("radius_m", circle_radius_m))
        path_n_poses = int(path_raw.get("n_poses", circle_n_poses))
        if radius_m <= 0.0:
            raise ValueError(
                f"circles.paths[{i}].radius_m must be positive"
            )
        if path_n_poses < 1:
            raise ValueError(
                f"circles.paths[{i}].n_poses must be >= 1"
            )
        circles.append(
            CircleSpec(
                look=look,
                height_m=height_m,
                radius_m=radius_m,
                n_poses=path_n_poses,
                start_deg=float(
                    path_raw.get("start_deg", circle_start_deg)
                ),
            )
        )

    video_fps = float(data.get("video_fps", 10.0))
    if video_fps <= 0.0:
        raise ValueError("video_fps must be positive")

    pano_raw = data.get("panorama") or {}
    if not isinstance(pano_raw, dict):
        raise ValueError("panorama must be a mapping")
    pano_w, pano_h = _as_hw(pano_raw, what="panorama")
    pano_rot = pano_raw.get("rotation_xyz_deg", [90.0, 0.0, 0.0])
    panorama = PanoramaSpec(
        prim_path=str(pano_raw.get("prim_path", "/World/Scene/pano_camera")),
        translation=_as_xyz(
            pano_raw.get("translation", [2.05, 1.95, orbit.height_m]),
            what="panorama.translation",
        ),
        rotation_xyz_deg=_as_xyz(pano_rot, what="panorama.rotation_xyz_deg"),
        width=pano_w,
        height=pano_h,
        projection=str(pano_raw.get("projection", "fisheyeSpherical")),
    )
    return FlythroughConfig(
        camera=camera,
        orbit=orbit,
        circles=tuple(circles),
        panorama=panorama,
        video_fps=video_fps,
    )
