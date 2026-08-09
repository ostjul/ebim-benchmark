# Teleoperation architecture

All three Isaac tasks share one pipeline shape. Only the last stage — the simulator process —
differs between them.

```
host device publishers  →  teleop adapters      →  republisher + position controller  →  bridge in the sim container
(external repo)            keyboard_to_base.py     /bridge/* → /isaac/*, gripper          joint state + command node
                           gello_to_bridge.py      calibration; browser UI on :8090
```

The keyboard, GELLO, and pedal publishers live in the separate
[`EBiM-Benchmark/teleoperation`](https://github.com/EBiM-Benchmark/teleoperation) repository, not
here. Everything is `network_mode: host` with FastDDS over UDPv4, so topics flow between containers
and host with no broker configuration — and, by the same token, the design assumes a single machine
or a LAN.

When topics fail to cross a host/container or host/host boundary, the two settings that matter are
`RMW_IMPLEMENTATION=rmw_fastrtps_cpp` and a matching `ROS_DOMAIN_ID` on **both** sides; forcing
`FASTDDS_BUILTIN_TRANSPORTS=UDPv4` resolves the remainder. Reaching a simulator across the public
internet is out of scope for this design — it would need a VPN or a FastDDS discovery server, and
WAN latency inside a teleoperation loop is its own problem.

**Only one task's helper stack may run at a time.** They bind identical topic names on the host
network and all want browser port 8090.

## Input channels are not interchangeable

| Channel | Path into the simulator | Needs a displayed window? |
|---|---|---|
| Arms + spine, in-window keys | carb keyboard in the focused Kit viewport → RMPflow | **Yes** — disabled under `--headless` |
| Mobile base, keyboard | `keyboard_state_publisher` → `/keyboard/state` → adapter → `/pedal/state` | No (publisher terminal needs focus) |
| Arms + grippers, browser UI | HTTP on `:8090` → `/isaac/browser/*` | No |
| GELLO + foot pedal | `/dev/ttyACM*` on the publisher's machine → `/bridge/*` → republisher | No |

In-window keyboard control **bypasses ROS entirely**. While it is active the bridge ignores
incoming ROS arm and gripper commands and RMPflow owns those targets; base and spine control keep
working through `/pedal/state` and the Up/Down key handler. Conflicting bare-key viewport hotkeys
(`F` frame-selection, `Q/W/E/R` transform tools) are deregistered at startup.

A per-group watchdog (`--command-timeout`, default 1 s, negative disables) stops re-applying a
group's cached command once its topics go quiet, so the drives hold the last applied target and a
dead publisher cannot stomp later state such as a post-reset ready pose. `/pedal/state` has its own
`--pedal-timeout` that forces the base twist to zero instead.

## GELLO Duo specifics

`franka_gello_state_publisher`'s `main.launch.py` reads a duo YAML and spawns one `gello_publisher`
node per entry, namespaced `left`/`right`, publishing at **25 Hz**:

```
/{left,right}/gello/joint_states                                    -> sensor_msgs/JointState, names fr3_joint1..7, frame_id fr3_link0
/{left,right}/gripper/gripper_client/target_gripper_width_percent   -> std_msgs/Float32
```

`gello_to_bridge.py` remaps these verbatim — `msg.position[:7]`, no IK, no clutch, no scaling — onto
`/bridge/{left,right}_joint_commands` (renamed `{left,right}_fr3v2_joint1..7`) and
`/bridge/{left,right}_robotiq_joint_commands`.

**The GELLO path needs both helper containers, not one.** `position_controller` carries
`/bridge/*_joint_commands` → `/isaac/*_joint_commands` (arms); `ros_republisher` carries the gripper
topics and applies open/close calibration. The republisher sends arms to `/isaac/browser/*` by
default, so it alone is not enough — an easy way to end up with a working gripper and a frozen arm.

**Hardware is OpenRB-150, not U2D2.** The pre-assembled Franka GELLO Duo ships two ROBOTIS
OpenRB-150 controllers (USB VID:PID `2f5d:2202`), which enumerate as CDC-ACM — `/dev/ttyACM*`, the
device the input-channel table above already assumes. The DIY/U2D2 build is FTDI (`0403:6014`) and
shows up as `/dev/ttyUSB*` instead.

**`com_port` in the duo YAML is the bare by-id name** — e.g.
`usb-ROBOTIS_OpenRB-150_<serial>-if00` — with no `/dev/serial/by-id/` prefix; `main.launch.py`
prepends it. A full path there silently fails.

**Power-on init routine, required every time:** handles on the base pins, white marker dots on
joint 2 pointing up, cables untwisted, then press the reboot button on each OpenRB-150 (left of its
USB port) twice. Skipping it yields wrong joint angles with no error.

**Do not set `dynamixel_torque_enable` to 1** on an OpenRB-150 without an external 5 V supply on the
board's power terminal and the jumper on VIN(DXL) — Franka's own docs warn it can damage the host
USB port. Shipped configs keep it all-zero.

**Calibration** (`get_offsets.py`, in the `teleoperation` repo under
`src/franka_gello_state_publisher/scripts/`) derives `assembly_offsets` and `gripper_range_rad`. It
imports `franka_gello_state_publisher.*`, so it must run inside the built colcon workspace — in
practice inside the gello container, not on a bare host — and the running publisher must be stopped
first, since it holds the serial ports open. Dual-arm calibration poses:

```
left:  --start-joints -1.57 -0.80  1.80 -3.00  1.40 1.50 -2.10
right: --start-joints  1.57 -0.80 -1.80 -3.00 -1.40 1.50  2.10
both:  --joint-signs 1 -1 1 -1 1 1 1
```

Franka states pre-assembled units work on stock defaults, so treat this as a verification step
rather than a mandatory one.

Upstream references: https://franka.de/gello and
https://github.com/wuphilipp/gello_software/tree/main/ros2

## Code reuse — edit the source, not a copy

This is the part that is easy to get wrong. Task 2 and Task 3 do not have their own copies of the
helper nodes.

- **`task1_isaacsim/scripts/isaac_bridge_constants.py`** is the single source of joint names,
  Robotiq driver/coupled-joint constants, and topic layout. Task 2's bridge core imports it
  directly.
- **`task2_isaacsim/docker-compose.yml` and `task3_isaacsim/docker-compose.yml` mount
  `../task1_isaacsim` at `/workspace`**, so their helper containers execute the *Task 1* scripts
  unmodified. Changing `task1_isaacsim/scripts/{adapters,controllers}/*.py` or
  `task1_isaacsim/services/` changes all three tasks at once.
- **`task2_isaacsim/scripts/isaacsim_fr3duo_teleop_bridge_core.py`** (~1800 lines) is the PhysX
  reimplementation of the Task 1 bridge — same topics, joint names, and defaults, with Task 1's
  swerve-base math and spine control ported over. `scene_room.py` and `scene_barebone.py` are thin
  stage builders on top of it.
- **`task3_isaacsim/scripts/scene_room.py`** imports Task 2's bridge args plus the shared room
  builder, and adds `gripper_profiles.py` on top.
- **`scripts/scenes/scene_robot_room_keyboard.py`** is the shared room-stage builder used by Tasks
  2 and 3. It is an implementation module, never a participant entry point.

`task2_isaacsim/PIPELINE_REF.md` holds the full technical reference: container topology, per-topic
contract tables, configuration precedence, dataset schema, and the complete Task 1 ↔ Task 2
mapping table. Read it before changing anything topic-shaped.

## Topic contract

Task 2/3 topic names come from `task2_isaacsim/config/topics.yaml`, loaded through
`task2_isaacsim/scripts/topics.py`. The loader **fails hard** on a missing file or key — no process
falls back to baked-in names, so a rename is either picked up everywhere or rejected loudly at
startup. `topics.py` is stdlib + PyYAML only and safe to import outside Isaac Sim.

Renaming in `topics.yaml` is **not sufficient** for three groups of values:

1. **Command topics and `/pedal/state`** are rebuilt from `--bridge-prefix` / `--isaac-prefix`
   inside the Task 1 services reused via mount (`ros_joint_republisher.py`, the browser controller,
   the teleop adapters). The bridge logs its loaded contract at startup — compare with
   `ros2 topic list`.
2. **Robot camera namespaces and resolutions** are duplicated in
   `task2_isaacsim/assets/embodiments/fr3duo_mobile_task2/camera_sensors.yaml`;
   `camera_publishers.py` cross-checks the two at graph-build time and raises on mismatch.
3. **`/isaac/eval_camera/*`** must match `config/cameras_room.yaml` / `cameras_barebone.yaml` *and*
   the Task 2 evaluation stack. The room values mirror the hardcoded setup in
   `scripts/scenes/scene_robot_room_keyboard.py`, which stays authoritative there.

## Task 1 MuJoCo is a different animal

`task1_mujoco/` is a self-contained vendored import from an upstream repository. Everything runs in
a **single MuJoCo process** (`main.py`) — robot, cable, board, and physics in one simulation, no
separate worlds and no coupling layers. All five input modes
(`--input keyboard|gamepad|vr|gello|ros_teleop`) feed the same control stack: grasp-aware scaling →
smoothing → contact clamp → damped-least-squares IK → force-servo grasping.

`--input ros_teleop` is the mode built for splitting device from simulator: a publisher node reads
the physical device and emits device-agnostic Cartesian commands (`/cmd_vel`, `<side>/teleop_cmd`,
`<side>/gripper_cmd`), and the sim subscribes. The three publishers ship in
`task1_mujoco/teleop_ros2/` (keyboard, gamepad, VR), each with a `--pattern` synthetic self-test
for when no hardware is attached.

Both the local input modes and the ROS publishers **poll the device directly rather than reading
window events** — X11 `query_keymap` for the keyboard on Linux, `GetAsyncKeyState` on Windows, SDL
for the gamepad. Two consequences: held keys work without the viewer having focus, and the process
doing the polling needs a display connection (or the device) on *its own* machine, which is
precisely what `ros_teleop` lets you move. On Wayland, `main.py` forces `GLFW_PLATFORM=x11` so the
window lands on XWayland where `query_keymap` can see it.

It has its own launchers (`start.sh`, `start.bat`, `docker-run.sh`), its own `ruff.toml`, and is
excluded from the root pre-commit run.

## Evaluation

`scripts/evaluation/task2/` and `task3/` are **development facilitators**, not the official scorer
— official scoring follows the rules on the competition page. Task 2's evaluator runs in its own
`ros:jazzy-ros-base` container, fully isolated from the Isaac stack, and computes a bounding-box
IoU plus orientation check from the `/isaac/eval_camera/*` streams. Task 3's `grading.py` is pure
logic covering all four stages and is not yet connected to the live teleoperation loop.

## Simulation gotcha: Fabric and stale USD

With PhysX Fabric enabled (needed for scenes with hundreds of rigid bodies), USD remains the
authoring format but runtime body transforms travel through Fabric's simulation data path to the
renderer. USD xform attributes are then **stale during simulation** — read runtime body poses
through PhysX, Fabric-aware, or tensor APIs, never straight from USD.
