# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "task2_isaacsim" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import flythrough_path  # noqa: E402
import pack_lichtfeld_dataset as pack  # noqa: E402


def test_pixel_focal_length_usd_defaults():
    fl_x = pack.pixel_focal_length(20.0, 1280)
    assert fl_x == pytest.approx((20.0 / 20.955) * 1280)


def test_camera_to_world_matches_look_minus_z():
    quat = flythrough_path.euler_xyz_to_quat((90.0, 0.0, 0.0))
    xyz = (4.55, 1.95, 1.5)
    matrix = pack.camera_to_world_matrix(xyz, quat)
    look = flythrough_path.rotate_vector_quat(quat, (0.0, 0.0, -1.0))
    # Column 2 is camera +Z, so look (-Z) is the negated third column.
    assert (-matrix[0][2], -matrix[1][2], -matrix[2][2]) == pytest.approx(look)
    assert (matrix[0][3], matrix[1][3], matrix[2][3]) == pytest.approx(xyz)
    assert matrix[3] == [0.0, 0.0, 0.0, 1.0]
    rotation = [
        [matrix[0][0], matrix[0][1], matrix[0][2]],
        [matrix[1][0], matrix[1][1], matrix[1][2]],
        [matrix[2][0], matrix[2][1], matrix[2][2]],
    ]
    det = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    assert det == pytest.approx(1.0, abs=1e-6)


def _write_capture(tmp_path: Path, n_frames: int = 2) -> Path:
    capture = tmp_path / "capture"
    rgb = capture / "rgb"
    rgb.mkdir(parents=True)
    frames = []
    for index in range(n_frames):
        (rgb / f"{index:06d}.png").write_bytes(b"\x89PNG\r\n")
        rotation = flythrough_path.look_along_rotation_xyz_deg(0.0, 1.0)
        quat = flythrough_path.euler_xyz_to_quat(rotation)
        frames.append(
            {
                "index": index,
                "track": "orbit",
                "look": "along_path",
                "xyz": [4.55, 1.95 + 0.1 * index, 1.5],
                "quat_wxyz": list(quat),
            }
        )
    (capture / "poses.json").write_text(
        json.dumps({"frames": frames}) + "\n", encoding="utf-8"
    )
    (capture / "camera.json").write_text(
        json.dumps(
            {
                "width": 1280,
                "height": 720,
                "focal_length": 20.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return capture


def test_pack_writes_transforms_and_relative_symlinks(tmp_path: Path):
    capture = _write_capture(tmp_path)
    dataset = pack.pack_lichtfeld_dataset(capture)
    transforms = json.loads((dataset / "transforms.json").read_text())
    assert transforms["camera_model"] == "PINHOLE"
    assert transforms["w"] == 1280
    assert transforms["h"] == 720
    assert transforms["fl_x"] == pytest.approx(transforms["fl_y"])
    assert transforms["cx"] == pytest.approx(640.0)
    assert transforms["cy"] == pytest.approx(360.0)
    assert len(transforms["frames"]) == 2
    assert transforms["frames"][0]["file_path"] == "images/000000.png"
    image = dataset / "images" / "000000.png"
    assert image.is_symlink()
    assert image.resolve() == (capture / "rgb" / "000000.png").resolve()
    rel = Path(image.readlink())
    assert not rel.is_absolute()


def test_pack_copy_and_overwrite(tmp_path: Path):
    capture = _write_capture(tmp_path)
    dataset = pack.pack_lichtfeld_dataset(capture, copy_images=True)
    image = dataset / "images" / "000000.png"
    assert image.is_file()
    assert not image.is_symlink()
    with pytest.raises(FileExistsError):
        pack.pack_lichtfeld_dataset(capture, copy_images=True)
    pack.pack_lichtfeld_dataset(capture, copy_images=True, overwrite=True)
    assert image.is_file()


def test_require_rgb_frames_reports_missing(tmp_path: Path):
    capture = _write_capture(tmp_path, n_frames=2)
    (capture / "rgb" / "000001.png").unlink()
    _, frames = pack.load_capture(capture)
    with pytest.raises(FileNotFoundError, match="000001.png"):
        pack.require_rgb_frames(capture, frames)


def test_pixel_focal_rejects_non_positive():
    with pytest.raises(ValueError, match="focal_length_mm"):
        pack.pixel_focal_length(0.0, 1280)
    with pytest.raises(ValueError, match="horizontal_aperture"):
        pack.pixel_focal_length(20.0, 1280, 0.0)
