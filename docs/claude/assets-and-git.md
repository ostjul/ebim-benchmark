# Assets, submodules, and git hygiene

## Large assets are not in git

The Task 1 robot USD and the cable board meshes are downloaded, not cloned:

```bash
task1_isaacsim/scripts/download_large_assets.sh
```

It unpacks a zip from OneDrive into `task1_isaacsim/`, placing:

- `assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd`
- `cable_world/assets/table_board_fixture/Assets/board_segment.usd`
- `cable_world/assets/table_board_fixture/Assets/board_segment_upper_right.usd`

All three are listed in `.gitignore`. Tasks 2 and 3 use the **same** robot USD, so this download is
a prerequisite for them too, not just Task 1. Override the source with
`LARGE_ASSETS_URL="…" task1_isaacsim/scripts/download_large_assets.sh` when OneDrive needs a manual
click.

If the download returns HTTP 403 and the `origin/Robotiq_DEMO` branch is available, the Task 3
README documents extracting just the robot USD with `git show`.

## Why not Git LFS

`check-added-large-files` rejects anything over 2 MB, and the fix is **not** to move it to LFS.
GitHub meters LFS storage and bandwidth against the repository owner: every clone that fetches an
LFS object draws on a 10 GiB/month allowance, while ordinary git objects draw on none of it. Small
binaries belong in git as ordinary blobs; assets in the hundreds of MB belong in external hosting.

Only `assets/robot_room.usd` remains LFS-tracked (see `.gitattributes`). If a checkout has pointer
files instead of real assets, `git lfs install` then `git lfs pull`.

`.gitattributes` also marks `task1_isaacsim/**/*.usd` as `-text` — those are binary USDC and must
never be EOL-converted — and pins `task1_mujoco/**/*.sh` plus the extensionless
`task1_mujoco/robotiq_duo_full_scene_minimal_core/release/ebim` launcher to LF, since CRLF breaks
their shebangs.

## Path resolution

Scripts resolve paths through `scripts/common/path_utils.py` (repository root, `assets/`,
`third_party/franka_description/urdfs/...`) rather than assuming a script's own location. This is
what allows runnable scripts to live outside the repository root.

Launchers additionally translate host paths to container paths before `docker exec` — a host path
under the repo root becomes `/workspace/EBiM_Challenge/<relative>`.

## Submodules

```
newton/
third_party/franka_description/
```

```bash
git clone --recurse-submodules <repository-url>
git submodule update --init --recursive   # for an existing clone
```

## Scenes and the base room

The active base scene is `assets/robot_room.usd` (and `assets/robot_room_v2/robot_room_v2.usdc`
for Task 1), composed through `scripts/scenes/scene_robot_room_keyboard.py`. New task work builds
on that room rather than generating new base scenes.

Everything under `scripts/deprecated/` — `create_wall_room.py`, `compose_scene_usd.py`, the older
tabletop scene generators and keyboard demos — is reference-only and does not define the current
competition scene. Their flags are documented in the root `README.md` behind a collapsed section.

Inspect any stage with:

```bash
python scripts/tools/inspect_usd.py assets/robot_room.usd
```

## Gitignored paths worth knowing

`.claude/` (agent-local notes), `**/recordings/*`, `**/output*`, `*.zip` except
`assets/Aloha_mobile_fr3_duo.zip`, `*.env` except `.env.example`, `docker/.env.base`, and
`task1_isaacsim/isaaclab_overlay/.env.ros2_jazzy`. There is also a `/dev/null` entry guarding
against a Git-LFS hook artifact that appears when `core.hooksPath` is set to the relative path
`dev/null` on Windows.
