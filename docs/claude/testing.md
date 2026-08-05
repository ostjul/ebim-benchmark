# Testing

**CI does not run tests.** `.github/workflows/pre-commit.yaml` is the only workflow, so every test
below is opt-in and can rot silently. Run them before touching shared modules.

## pytest suites

`pyproject.toml` sets `pythonpath = ["scripts/common"]`, so `path_utils` and `tmr_base_control`
import without installation. There is no `conftest.py` anywhere; individual test files insert the
sibling directories they need.

```bash
python3 -m pytest scripts/tests task3_isaacsim/tests task3_isaacsim/deprecated/tests
```

| Suite | Covers |
|---|---|
| `scripts/tests/` | shared room builder, swerve-base math, Task 3 repository layout |
| `task3_isaacsim/tests/` | gripper-profile selection, launcher CLI smoke tests |
| `task3_isaacsim/deprecated/tests/` | the pre-ROS RMPflow stack, runnable in place |

### The torch caveat

Several tests import `torch`, which exists only inside the Isaac Lab container. On a bare host a
collection error aborts the entire run, so either run pytest inside the container or skip them:

```bash
python3 -m pytest scripts/tests task3_isaacsim/tests task3_isaacsim/deprecated/tests \
  --ignore=scripts/tests/test_tmr_base_control.py \
  --ignore=task3_isaacsim/deprecated/tests/test_teleop_targets.py \
  --ignore=task3_isaacsim/deprecated/tests/test_dual_arm_lula.py
```

That leaves 3 failures from in-body `import torch` inside
`scripts/tests/test_scene_robot_room_keyboard.py`. Everything else passes on a bare host — 71
passed at last check. Any *other* failure is a real one.

## Evaluation suites

Pure logic, no pytest, no ROS, no Isaac Sim. Task 3's is stdlib-only; Task 2's needs numpy and
PyYAML.

```bash
python3 -B scripts/evaluation/task3/tests/test_grading.py            # all four stages
python3 -B scripts/evaluation/task3/tests/test_grading.py stage2     # one stage
python3 -B scripts/evaluation/task2/tests/test_evaluation.py
```

Both files are executable scripts with an `if __name__ == "__main__"` runner that inserts the
parent directory on `sys.path` — that is why the flat `evaluation.py` / `grading.py` modules
import cleanly without a package.

The Task 3 Isaac Sim integration validator (`scripts/evaluation/task3/integration_test.py`) builds
its own scene and moves objects through deterministic test motions. It does **not** grade a live
teleoperation session; see `scripts/evaluation/task3/README.md`.

## The layout tripwire

`scripts/tests/test_task3_repository_layout.py` pins the Task 3 file layout in two directions: a
list of paths that must exist, and a list of superseded paths that must *not*. It exists because
Task 3 files have moved between `scripts/`, `scripts/common/`, and the task folder more than once.

It earned its keep. PR #41 landed the deprecated stack under `task3_isaacsim/deprecated/scripts/`
while the files' own path constants, both READMEs, and all five deprecated test modules assumed a
flat `task3_isaacsim/deprecated/`. Nothing in CI noticed. The consequences were real, not
cosmetic:

- `deprecated/scene_robot_room_rmpflow.py` computes `REPO_ROOT = parents[2]` and
  `TASK3_ROOT = parents[1]`; one level too deep, those resolved to `task3_isaacsim` and
  `deprecated`, breaking both `import scene_robot_room_keyboard` and the Lula config directory.
- `scripts/scenes/scene_robot_room_keyboard.py` puts `task3_isaacsim/deprecated/common` on
  `sys.path`; with that directory absent, `_run_keyboard_control_app`'s
  `from teleop_commands import safe_command` raised — and keyboard control defaults **on** for
  `--task task3`.

Restoring the flat layout fixed the tests and both runtime paths at once. If you move those files,
this test is what tells you.
