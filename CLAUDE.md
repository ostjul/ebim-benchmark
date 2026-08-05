# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Simulation environments for the EBiM robotics competition: three tasks, each shipped for one or
more physics engines. Every `task<N>_<engine>/` directory is a **self-contained participant
runtime** with its own README, `docker-compose.yml`, and launcher script.

| Task | Engine | Directory |
|---|---|---|
| 1 — Cable Routing & Plugging | Isaac Lab + Newton | `task1_isaacsim/` |
| 1 — Cable Routing & Plugging | MuJoCo | `task1_mujoco/` |
| 2 — Deformable Thermal Pad Placement | Isaac Sim 5.1.0 / PhysX | `task2_isaacsim/` |
| 3 — Assisted Living & Feeding | Isaac Sim 5.1.0 | `task3_isaacsim/` |

`STATUS.md` is the authoritative capability matrix — check it before assuming a task/engine
combination works end to end. Evaluation code here is a development facilitator; official scoring
lives on the competition page.

This is **not a pip-installable package**. `pyproject.toml` holds tool configuration only, and
scripts reach each other with `sys.path.insert(...)` of sibling directories rather than package
imports.

## Detailed guides

Read the relevant one before making changes in that area — each covers gotchas that are expensive
to rediscover.

| Guide | Covers |
|---|---|
| [docs/claude/lint-and-ci.md](docs/claude/lint-and-ci.md) | pre-commit as the only CI gate, the load-bearing excludes, ruff's two config zones, commit/PR conventions |
| [docs/claude/testing.md](docs/claude/testing.md) | which suites exist, the torch caveat, the Task 3 layout tripwire |
| [docs/claude/containers.md](docs/claude/containers.md) | the two container families, compose invariants, the `python` wrapper, X11 and headless operation |
| [docs/claude/teleop-architecture.md](docs/claude/teleop-architecture.md) | the shared pipeline, cross-task code reuse, the topic contract, MuJoCo's differences |
| [docs/claude/assets-and-git.md](docs/claude/assets-and-git.md) | downloaded assets, the no-LFS policy, submodules, path resolution |

Repo-side references: `task2_isaacsim/PIPELINE_REF.md` (the deepest technical reference for the
teleop and recording pipeline), `docs/developer_setup.md` (host setup), and each task's README.

### Where to record new findings

When you learn something worth keeping, route it by whether it generalizes:

| Kind of finding | Goes to | Shared? |
|---|---|---|
| True on **any** machine — setup steps, launcher flags, architecture, topic contracts, gotchas, fixes | `docs/claude/` — extend the matching guide, or add one and link it in the table above | Yes, tracked and committed |
| Specific to **one** person's hardware or environment — GPU and architecture, display or headless setup, attached devices, host paths, personal workflow | `.claude/memory/` — one file per fact, plus a pointer line in `.claude/memory/MEMORY.md` | No, gitignored and local |

Keep this split honest in both directions. A machine-specific detail written into `docs/claude/`
misleads every other contributor; a general repository fact buried in `.claude/memory/` is lost to
everyone else. When a finding has both parts, split it — the general mechanism goes in
`docs/claude/`, the local specifics in `.claude/memory/`, and the memory note points at the guide.

## Everyday commands

```bash
# Lint — the only CI gate. There is no test job in CI.
pre-commit run --all-files

# Tests (see docs/claude/testing.md for the torch caveat)
python3 -m pytest scripts/tests task3_isaacsim/tests task3_isaacsim/deprecated/tests

# Start the shared Isaac Sim container (--env-file is required, not optional)
docker compose --env-file docker/.env.base -f docker/docker-compose.yaml \
  --profile isaac-sim-5.1.0 up -d

# Task launchers, all from the repository root
EMBODIMENT=fr3duo_mobile bash task1_isaacsim/scripts/run_isaaclab_newton_teleop.sh \
  --with-keyboard-teleop --no-browser
bash task2_isaacsim/scripts/run_isaacsim_teleop.sh --scene barebone --with-keyboard-teleop
bash task3_isaacsim/scripts/run_isaacsim_teleop.sh --gripper robotiq
cd task1_mujoco && ./start.sh
```

Every Isaac launcher `docker exec`s into an **already-running** container, translates host paths to
container paths, and forwards everything after a bare `--` to the underlying Python scene or
bridge.

## Architecture in brief

**Two simulator runtimes, deliberately.** Task 1 runs Isaac Lab on the Newton / MJWarp backend in a
container built from a *separate* pinned IsaacLab checkout plus an overlay. Tasks 2 and 3 run plain
Isaac Sim 5.1.0 / PhysX, because the Task 2 thermal pad needs PhysX GPU deformables that Newton
cannot provide.

**One teleop pipeline, shared.** Host device publishers (from a separate `teleoperation` repo) →
teleop adapters → republisher and position controller → the bridge inside the simulator container.
Only the last stage differs per task.

**Tasks 2 and 3 do not copy Task 1's helper nodes — they mount them.** Their compose files bind
`../task1_isaacsim` at `/workspace`, and Task 2's bridge imports Task 1's
`isaac_bridge_constants.py` directly. A change under `task1_isaacsim/scripts/{adapters,controllers}`
or `task1_isaacsim/services/` changes all three tasks at once.

**Only one task's helper stack may run at a time** — they bind identical host-network topics and
all want browser port 8090.

**`task1_mujoco/` is a vendored, self-contained subtree**: one MuJoCo process, five input modes on
a single control stack, its own launchers and ruff config, excluded from the root pre-commit run.
