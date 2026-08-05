# Lint, CI, and contribution checks

## pre-commit is the only CI gate

```bash
pip install pre-commit   # once
pre-commit install       # once
pre-commit run --all-files
```

`.github/workflows/pre-commit.yaml` runs exactly `pre-commit run --all-files --show-diff-on-failure`
on every push and pull request, and it is the required status check on `main`. **There is no test
job in CI** — see [testing.md](testing.md) for what that leaves unguarded.

## The load-bearing excludes

The `exclude:` regex at the bottom of `.pre-commit-config.yaml` is a *global* file filter: it
applies to every hook, not just the last one. It currently skips `scripts/deprecated/`,
`scripts/manual_tests/`, `scripts/newton_examples/`, `assets/`, `DEMO/`, `docker/`, `.vscode/`,
`.github/`, `task1_isaacsim/`, and `task1_mujoco/`.

Three of those are **load-bearing, not noise reduction**:

| Path | What it protects |
|---|---|
| `.vscode/` | `tools/setup_vscode.py`, Isaac Lab BSD-3-Clause header |
| `scripts/newton_examples/` | 3 files, Newton Developers Apache-2.0 header |
| `task1_isaacsim/` | `cable_world/*.py`, Newton Developers Apache-2.0 header |

All are `.py`, so they match `insert-license`'s `files:` filter, and the exclusion is the only
thing stopping that hook from stamping an EBiM copyright on top of a third party's. This was
verified: at a non-excluded path the hook does stamp them, and pre-commit still reports success —
it fails silently in the direction that matters.

Before removing any of the three, add a per-hook `exclude:` to `insert-license` covering those
paths (tested: the hook then reports `Skipped`). `LICENSES/README.md` covers the Isaac Lab case in
detail; the Newton-headered files are Apache-2.0, same as this repo's root `LICENSE`.

## Hooks worth knowing

- **`insert-license`** auto-stamps `.github/LICENSE_HEADER.txt` (Apache-2.0, EBiM) onto every new
  `.py`/`.ya?ml` outside the excludes. New files get it for free; do not hand-write it.
- **`check-added-large-files`** rejects anything over 2 MB. Do **not** route around it with Git
  LFS — see [assets-and-git.md](assets-and-git.md) for why and what to do instead.
- **`codespell`** skips USD/image/bib/css files; `ignore-words-list` in `pyproject.toml` holds the
  domain words it would otherwise flag (`haa`, `reacher`, `thirdparty`, …).
- **`pyright`** is commented out in `.pre-commit-config.yaml` (it hung under VPN). Its config still
  lives in `pyproject.toml` under `[tool.pyright]` and covers `scripts` + `.vscode/tools`.

## Ruff configuration

Ruff discovers the closest config file up the tree, which produces two zones:

| Zone | Config | Line length |
|---|---|---|
| Repository default | `pyproject.toml` | 79 |
| `task1_mujoco/` | `task1_mujoco/ruff.toml` | 120 |

`task1_mujoco/` is a self-contained import from an upstream repo with its own established
convention; the override avoids a repo-wide reformat while keeping every other rule from the root
config. Note that `task1_mujoco/` is *also* in the global pre-commit exclude, so that ruff.toml
only takes effect when you invoke ruff directly.

`pyproject.toml` additionally sets `extend-exclude = ["task1_isaacsim", ...]` for ruff — that
directory is ported `franka_isaacSim` code with long lines and vendored sources.

The isort configuration defines custom sections so Omniverse runtime extensions (`isaacsim`,
`omni`, `pxr`, `carb`, `usdrt`, `curobo`) and each Isaac Lab extension group sort separately from
ordinary third-party imports. Import blocks in this repo will look unusual if you sort them by
hand — let ruff do it.

## Commit and PR conventions

- Conventional-commit subjects: `fix(task2): …`, `docs(licenses): …`, `feat(task3): …`.
- `.github/PULL_REQUEST_TEMPLATE.md` requires: pre-commit run clean, docs updated where paths or
  behavior changed, no large binaries added, contributor listed in `CONTRIBUTORS.md`, labels and
  milestone set.
- `.github/CODEOWNERS` puts `@Ju6276 @ShangQingLiu @ebim-benchmark/core` on everything and routes
  `/task1_isaacsim/` to `@QGSQ @2houyuhang`.
