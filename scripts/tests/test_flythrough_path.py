# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "task2_isaacsim" / "scripts"
CONFIG_PATH = REPO_ROOT / "task2_isaacsim" / "config" / "room_flythrough.yaml"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import flythrough_path  # noqa: E402

TABLE_XY = (2.05, 1.95)
RADIUS_X_M = 2.5
RADIUS_Y_M = 0.5
DIAMETER_X_M = 5.0
DIAMETER_Y_M = 1.0
HEIGHT_M = 1.5
N_POSES = 120
WAVE_AMP_M = 0.10
WAVE_CYCLES = 4.0


def _orbit_frames(**kwargs):
    args = dict(
        center_xy=TABLE_XY,
        radius_x_m=RADIUS_X_M,
        radius_y_m=RADIUS_Y_M,
        height_m=HEIGHT_M,
        n_poses=N_POSES,
        wave_amp_m=WAVE_AMP_M,
        wave_cycles=WAVE_CYCLES,
    )
    args.update(kwargs)
    return flythrough_path.frames_from_orbit(**args)


def _almost_unit(vec, expected, atol=1e-6):
    length = math.sqrt(sum(c * c for c in vec))
    assert length > 0.0
    normed = tuple(c / length for c in vec)
    assert normed == pytest.approx(expected, abs=atol)


def test_look_along_yaw_matches_usd_camera_minus_z():
    # After Rx=90, local -Z is world +Y at yaw 0. Rz then aims -Z at (dx, dy).
    cases = {
        (0.0, 1.0): 0.0,
        (1.0, 0.0): -90.0,
        (0.0, -1.0): 180.0,
        (-1.0, 0.0): 90.0,
    }
    look = (0.0, 0.0, -1.0)
    for direction, yaw in cases.items():
        got = flythrough_path.look_along_yaw_deg(*direction)
        assert math.cos(math.radians(got)) == pytest.approx(
            math.cos(math.radians(yaw))
        )
        assert math.sin(math.radians(got)) == pytest.approx(
            math.sin(math.radians(yaw))
        )
        rotation = flythrough_path.look_along_rotation_xyz_deg(*direction)
        assert rotation[0] == pytest.approx(90.0)
        assert rotation[1] == pytest.approx(0.0)
        quat = flythrough_path.euler_xyz_to_quat(rotation)
        world_look = flythrough_path.rotate_vector_quat(quat, look)
        _almost_unit(world_look, (direction[0], direction[1], 0.0))
        world_up = flythrough_path.rotate_vector_quat(quat, (0.0, 1.0, 0.0))
        _almost_unit(world_up, (0.0, 0.0, 1.0))


def test_rx90_identity_looks_positive_y():
    quat = flythrough_path.euler_xyz_to_quat((90.0, 0.0, 0.0))
    look = flythrough_path.rotate_vector_quat(quat, (0.0, 0.0, -1.0))
    _almost_unit(look, (0.0, 1.0, 0.0))


def test_orbit_looks_along_path_tangent():
    frames = _orbit_frames()
    assert len(frames) == N_POSES
    cx, cy = TABLE_XY
    look_local = (0.0, 0.0, -1.0)
    zs = [frame.xyz[2] for frame in frames]
    assert max(zs) == pytest.approx(HEIGHT_M + WAVE_AMP_M, abs=0.02)
    assert min(zs) == pytest.approx(HEIGHT_M - WAVE_AMP_M, abs=0.02)
    looks = []
    for frame in frames:
        theta = 2.0 * math.pi * frame.index / N_POSES
        wave = WAVE_CYCLES * theta
        tx = -RADIUS_X_M * math.sin(theta)
        ty = RADIUS_Y_M * math.cos(theta)
        tz = WAVE_AMP_M * WAVE_CYCLES * math.cos(wave)
        tlen = math.sqrt(tx * tx + ty * ty + tz * tz)
        dx = frame.xyz[0] - cx
        dy = frame.xyz[1] - cy
        ellipse = (dx / RADIUS_X_M) ** 2 + (dy / RADIUS_Y_M) ** 2
        assert ellipse == pytest.approx(1.0)
        world_look = flythrough_path.rotate_vector_quat(
            frame.quat_wxyz, look_local
        )
        looks.append(world_look)
        _almost_unit(world_look, (tx / tlen, ty / tlen, tz / tlen))
        xy_speed = math.hypot(tx, ty)
        assert frame.pitch_deg == pytest.approx(
            math.degrees(math.atan2(tz, xy_speed))
        )
    # Consecutive look vectors stay in the same hemisphere (no 180° switch).
    closed = looks + [looks[0]]
    for a, b in zip(closed, closed[1:]):
        assert a[0] * b[0] + a[1] * b[1] + a[2] * b[2] > 0.5


def test_orbit_starts_at_plus_x_and_is_ccw():
    frames = _orbit_frames()
    cx, cy = TABLE_XY
    assert frames[0].xyz[0] == pytest.approx(cx + RADIUS_X_M)
    assert frames[0].xyz[1] == pytest.approx(cy)
    assert frames[0].xyz[2] == pytest.approx(HEIGHT_M)
    # At θ=0, tangent is +Y; XY speed is the Y semi-axis.
    assert frames[0].yaw_deg == pytest.approx(0.0)
    assert frames[0].pitch_deg == pytest.approx(
        math.degrees(math.atan2(WAVE_AMP_M * WAVE_CYCLES, RADIUS_Y_M))
    )
    look = flythrough_path.rotate_vector_quat(
        frames[0].quat_wxyz, (0.0, 0.0, -1.0)
    )
    assert look[1] > 0.0
    assert frames[1].xyz[1] > cy
    assert frames[-1].xyz[:2] != pytest.approx(frames[0].xyz[:2])
    xs = [frame.xyz[0] for frame in frames]
    ys = [frame.xyz[1] for frame in frames]
    assert max(xs) - min(xs) == pytest.approx(DIAMETER_X_M)
    assert max(ys) - min(ys) == pytest.approx(DIAMETER_Y_M)


def test_zero_wave_is_flat_horizontal():
    frames = _orbit_frames(n_poses=12, wave_amp_m=0.0)
    for frame in frames:
        assert frame.xyz[2] == pytest.approx(HEIGHT_M)
        assert frame.pitch_deg == pytest.approx(0.0)
        assert frame.rotation_xyz_deg[0] == pytest.approx(90.0)


def test_orbit_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="n_poses"):
        _orbit_frames(n_poses=0)
    with pytest.raises(ValueError, match="radius_x_m"):
        _orbit_frames(radius_x_m=0.0)
    with pytest.raises(ValueError, match="radius_y_m"):
        _orbit_frames(radius_y_m=0.0)
    with pytest.raises(ValueError, match="wave_amp_m"):
        _orbit_frames(wave_amp_m=-0.1)


def test_default_yaml_config_loads():
    config = flythrough_path.load_flythrough_config(CONFIG_PATH)
    assert config.camera.width == 1280
    assert config.camera.height == 720
    assert config.camera.focal_length == pytest.approx(20.0)
    assert config.orbit.center_xy == pytest.approx(TABLE_XY)
    assert config.orbit.diameter_x_m == pytest.approx(DIAMETER_X_M)
    assert config.orbit.diameter_y_m == pytest.approx(DIAMETER_Y_M)
    assert config.orbit.height_m == pytest.approx(HEIGHT_M)
    assert config.orbit.n_poses == N_POSES
    assert config.orbit.wave_amp_m == pytest.approx(WAVE_AMP_M)
    assert config.orbit.wave_cycles == pytest.approx(WAVE_CYCLES)
    assert [(c.look, c.height_m, c.radius_m) for c in config.circles] == [
        ("inward", 1.00, 1.0),
        ("outward", 1.25, 1.0),
        ("inward", 1.50, 1.0),
        ("outward", 1.75, 1.0),
    ]
    frames = flythrough_path.frames_from_config(config)
    assert len(frames) == N_POSES * (1 + len(config.circles))
    assert frames[0].xyz[0] == pytest.approx(TABLE_XY[0] + RADIUS_X_M)
    assert frames[0].xyz[1] == pytest.approx(TABLE_XY[1])
    assert frames[0].yaw_deg == pytest.approx(0.0)
    assert frames[0].track == "orbit"
    assert frames[0].look == "along_path"
    assert frames[N_POSES].look == "inward"
    assert frames[N_POSES].xyz[2] == pytest.approx(1.00)
    assert frames[-1].look == "outward"
    assert frames[-1].xyz[2] == pytest.approx(1.75)


def test_circle_inward_and_outward_looks():
    inward = flythrough_path.frames_from_circle(
        TABLE_XY, radius_m=1.0, height_m=1.0, n_poses=12, look="inward"
    )
    outward = flythrough_path.frames_from_circle(
        TABLE_XY, radius_m=1.0, height_m=1.75, n_poses=12, look="outward"
    )
    look_local = (0.0, 0.0, -1.0)
    cx, cy = TABLE_XY
    assert inward[0].xyz[0] == pytest.approx(cx + 1.0)
    assert inward[0].xyz[1] == pytest.approx(cy)
    for frame in inward:
        radial = (frame.xyz[0] - cx, frame.xyz[1] - cy, 0.0)
        world_look = flythrough_path.rotate_vector_quat(
            frame.quat_wxyz, look_local
        )
        _almost_unit(world_look, (-radial[0], -radial[1], 0.0))
        assert frame.xyz[2] == pytest.approx(1.0)
        assert frame.pitch_deg == pytest.approx(0.0)
        assert math.hypot(radial[0], radial[1]) == pytest.approx(1.0)
    for frame in outward:
        radial = (frame.xyz[0] - cx, frame.xyz[1] - cy, 0.0)
        world_look = flythrough_path.rotate_vector_quat(
            frame.quat_wxyz, look_local
        )
        _almost_unit(world_look, (radial[0], radial[1], 0.0))
        assert frame.xyz[2] == pytest.approx(1.75)
    with pytest.raises(ValueError, match="look"):
        flythrough_path.frames_from_circle(
            TABLE_XY, 1.0, 1.0, 8, look="sideways"
        )


def test_orbit_trajectory_plot_writes_png(tmp_path):
    config = flythrough_path.load_flythrough_config(CONFIG_PATH)
    frames = flythrough_path.frames_from_orbit(
        TABLE_XY,
        RADIUS_X_M,
        RADIUS_Y_M,
        HEIGHT_M,
        n_poses=24,
        wave_amp_m=WAVE_AMP_M,
        wave_cycles=WAVE_CYCLES,
    )
    for circle in config.circles:
        frames.extend(
            flythrough_path.frames_from_circle(
                TABLE_XY,
                circle.radius_m,
                circle.height_m,
                n_poses=24,
                look=circle.look,
            )
        )
    out = tmp_path / "orbit_trajectory.png"
    written = flythrough_path.plot_orbit_trajectory(frames, out, TABLE_XY)
    assert written == out
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_baked_task2_table_bbox_matches_room_usd():
    # robot_room.usd /root/root/___008 at Task 2 XY.
    assert flythrough_path.is_baked_task2_table_bbox(
        (2.06, 1.971), (1.25, 0.74, 0.75)
    )
    # Other cubicle table in the room (letters A–F), not Task 2.
    assert not flythrough_path.is_baked_task2_table_bbox(
        (-2.05, 1.94), (1.70, 0.80, 0.75)
    )
    # Tabletop slab alone is too thin to match the full table group.
    assert not flythrough_path.is_baked_task2_table_bbox(
        (2.06, 1.971), (1.25, 0.74, 0.02)
    )
