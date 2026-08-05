# Containers and runtimes

Everything runs in containers; the host only needs Docker, the NVIDIA Container Toolkit, and (for
GUI work) X11. The repository is bind-mounted at `/workspace/EBiM_Challenge` in every container.

## Two container families

The repo's own compose stack is **not** what Task 1 uses. Keep them straight:

| Family | Started by | Used by |
|---|---|---|
| `docker/docker-compose.yaml` — Isaac Sim 5.1.0, Isaac Sim 6.0.0-dev2, Isaac Lab 2.3.2 | this repo | Tasks 2 and 3, shared scene tooling |
| `isaac-lab-ros2_jazzy` — Isaac Lab `release/3.0.0-beta2` + a `ros2_jazzy` overlay | a **separate IsaacLab checkout** at a pinned commit | Task 1 Isaac Sim only |

Task 1 needs the second because it runs the Newton / MJWarp backend. Setup is a clone of IsaacLab
next to this repo, `git checkout 0916ea3c0f126821ef1783c7119d248834fc8d0b`, then
`task1_isaacsim/isaaclab_overlay/apply_overlay.sh`, then
`./docker/container.py start ros2_jazzy` from the IsaacLab checkout. Override the location with
`ISAACLAB_ROOT`. Details in `task1_isaacsim/isaaclab_overlay/README.md`.

Tasks 2 and 3 run in plain **Isaac Sim 5.1.0 / PhysX** because the Task 2 thermal pad uses
`PhysxDeformableBodyAPI` and needs PhysX GPU deformables, which Isaac Lab + Newton cannot run.

## The compose stack

| Profile | Service | Container | Base image |
|---|---|---|---|
| `isaac-sim-5.1.0` | `isaac-sim-5-1-0` | `isaac-sim-5-1-0-workshop` | `nvcr.io/nvidia/isaac-sim:5.1.0` |
| `isaac-sim-6.0.0` | `isaac-sim-6-0-0` | `isaac-sim-6-0-0-workshop` | `nvcr.io/nvidia/isaac-sim:6.0.0-dev2` |
| `isaac-lab-2.3.2` | `isaac-lab-2-3-2` | `isaac-lab-2-3-2-workshop` | `nvcr.io/nvidia/isaac-lab:2.3.2` |

**Always pass `--env-file docker/.env.base` explicitly.** An `env_file:` entry only sets container
environment variables; it does not expand `${...}` in the compose model, so without the flag the
image tags and mounts resolve to empty strings.

```bash
docker compose --env-file docker/.env.base -f docker/docker-compose.yaml \
  --profile isaac-sim-5.1.0 up -d
docker exec -it isaac-sim-5-1-0-workshop bash
docker compose --env-file docker/.env.base -f docker/docker-compose.yaml down
```

## Build and validate everything

```bash
python3 scripts/tools/validate_docker_runtimes.py --prepare-dirs --build --up
python3 scripts/tools/validate_docker_runtimes.py --down --skip-script-check
```

Builds the three images in parallel, starts them, then checks the workspace mount, cache mounts,
X11, host networking, script syntax, and USD path resolution. `--prepare-dirs` alone bootstraps the
host cache layout. Add `--external-network-check` to also verify outbound HTTPS to `nvcr.io`.

## The `python` wrapper

`docker/ebim-python` is copied into the image as both `/usr/local/bin/ebim-python` and
`/usr/local/bin/python`. It delegates to `/workspace/isaaclab/isaaclab.sh -p` or
`/isaac-sim/python.sh`, whichever exists, and prepends the newest `omni.usd.libs-*` extension to
`PYTHONPATH`/`LD_LIBRARY_PATH` so plain USD tools get `pxr`. Inside a container, `python` is
therefore never the system interpreter.

## Persistent storage and UID/GID

Caches live under `${HOME}/docker/ebim-challenge/<runtime>/`, mirroring NVIDIA's documented layout
with a version suffix per image. The Isaac Sim services run as `${HOST_UID}:${HOST_GID}` with
`${ISAAC_SIM_GID}` as a supplemental group so they can still read `/isaac-sim`, and their `HOME`
and XDG paths are pinned under `/isaac-sim` so Omniverse does not write to `/`.

`HOST_UID`/`HOST_GID` must match the owner of this repository. The defaults in `docker/.env.base`
are `1001:1001`; export your own before building if they differ:

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)
```

The stack deliberately does **not** bind-mount `/isaac-sim/extscache` — those folders also contain
bundled shader resources, and an empty host directory there breaks RTX shader loading.

On `Permission denied` for `/isaac-sim/kit/logs` or `user.config.json`, re-run `--prepare-dirs`,
`chown -R` the cache root to your UID/GID, then recreate the container with `--force-recreate`.

## X11 and running without a local display

GUI containers mount `${DISPLAY}`, `${XAUTHORITY}`, and `/tmp/.X11-unix`. Once per graphical
session:

```bash
xhost +local:docker
export DISPLAY=${DISPLAY:-:0}
export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}
touch "$XAUTHORITY"
```

Isaac Sim's WebRTC livestreaming is **not** wired into these launchers — the scene scripts
construct `SimulationApp({"headless": ..., "width": 1280, "height": 720})` with no livestream
configuration, so enabling it would require a code change.

On **aarch64 hosts (GB10 / DGX Spark and friends) that change would be pointless**: Isaac Sim
5.1.0 refuses to livestream on that architecture at all. Its own `/isaac-sim/runheadless.sh` opens
with

```sh
if [ "$(uname -m)" = "aarch64" ]; then
    echo 'Livestreaming is not supported on aarch64 for 5.1.0.'
```

and the image ships no `isaac-sim.streaming.sh` wrapper on arm64 even though
`apps/isaacsim.exp.full.streaming.kit` is present and declares `execFile = "isaac-sim.streaming"`.
The experience file's `omni.services.livestream.nvcf` dependency has no arm64 build. Verified on a
GB10, 2026-08-04. Do not spend time wiring a `--livestream` flag for an arm64 box; on x86 the same
script takes the other branch and streaming is available.

Without a display, the supported paths are:

- `--headless` plus the **browser controller on port 8090** for arm/gripper control (documented for
  Task 3; reachable through an SSH tunnel with `ssh -L 8090:localhost:8090 <host>`).
- ROS-topic input for the mobile base, which never needs a window.

What you lose headless is the in-Kit-window keyboard teleop for arms and spine — it reads the carb
keyboard from a focused viewport and is explicitly disabled under `--headless`. See
[teleop-architecture.md](teleop-architecture.md) for which input channel needs what.

If the ROS 2 bridge fails at startup with a missing `libament_index_cpp.so`, launch with
`--ros2-bridge fastdds` or `--ros2-bridge cyclonedds`; the launcher re-execs itself once in ROS
mode so `LD_LIBRARY_PATH` is set before Isaac Sim starts.
