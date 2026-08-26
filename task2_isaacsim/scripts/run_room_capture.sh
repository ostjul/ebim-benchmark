#!/usr/bin/env bash
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# --------------------------------------------------------------------------
# Task 2 launcher: room-only path-aligned orbit capture (Isaac Sim 5.1.0).
#
# Layout assumptions (see task2_isaacsim/README.md):
#   * The Isaac Sim 5.1.0 container (${ISAACSIM_CONTAINER}) is already running
#     with this repo bind-mounted at ${CONTAINER_REPO}
#     (default /workspace/EBiM_Challenge).
#   * Stop any scene_room.py / scene_barebone.py Kit process first — one
#     container can only run one Isaac Sim app.
# No teleop helper stack is started. No robot USD is required.
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASK2_DIRNAME="$(basename "${TASK2_ROOT}")"
REPO_ROOT="$(cd "${TASK2_ROOT}/.." && pwd)"
if [[ -f "${TASK2_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${TASK2_ROOT}/.env"
  set +a
fi

ISAACSIM_CONTAINER="${ISAACSIM_CONTAINER:-isaac-sim-5-1-0-workshop}"
CONTAINER_REPO="${CONTAINER_REPO:-/workspace/EBiM_Challenge}"
CONTAINER_TASK2="${CONTAINER_REPO}/${TASK2_DIRNAME}"

ROOM_USD_PATH="${ROOM_USD_PATH:-../assets/robot_room.usd}"
HEADLESS=false
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  task2_isaacsim/scripts/run_room_capture.sh [options] [-- capture-args]

Options:
  --room-usd-path PATH       Room USD path (default: ../assets/robot_room.usd)
  --headless                 Run Isaac Sim without a visible Kit window
  --                         Pass remaining args to capture_room_flythrough.py

Examples:
  bash task2_isaacsim/scripts/run_room_capture.sh --headless
  bash task2_isaacsim/scripts/run_room_capture.sh -- \
    --out /workspace/EBiM_Challenge/task2_isaacsim/captures/run1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --room-usd-path)
      ROOM_USD_PATH="$2"
      shift 2
      ;;
    --headless)
      HEADLESS=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

resolve_host_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    echo "${value}"
  else
    echo "$(cd "${TASK2_ROOT}" && cd "$(dirname "${value}")" && pwd)/$(basename "${value}")"
  fi
}

HOST_ROOM_USD="$(resolve_host_path "${ROOM_USD_PATH}")"

if [[ ! -f "${HOST_ROOM_USD}" ]]; then
  echo "Room USD file not found: ${HOST_ROOM_USD}" >&2
  exit 1
fi

if [[ "${HOST_ROOM_USD}" != "${REPO_ROOT}/"* ]]; then
  echo "Asset must be inside ${REPO_ROOT} so the Isaac Sim container can see it: ${HOST_ROOM_USD}" >&2
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx "${ISAACSIM_CONTAINER}"; then
  cat >&2 <<EOF
Isaac Sim container '${ISAACSIM_CONTAINER}' is not running.
Start it first (it must bind-mount this repo at ${CONTAINER_REPO}), or set
ISAACSIM_CONTAINER to the name of your running Isaac Sim 5.1.0 container.
EOF
  exit 1
fi

if ! docker exec "${ISAACSIM_CONTAINER}" test -d "${CONTAINER_REPO}"; then
  echo "The Isaac Sim container does not have this repository mounted at ${CONTAINER_REPO}." >&2
  exit 1
fi

CONTAINER_ROOM_USD="${CONTAINER_REPO}/${HOST_ROOM_USD#"${REPO_ROOT}/"}"

CAPTURE_ARGS=(
  "--room-usd" "${CONTAINER_ROOM_USD}"
  "--config" "${CONTAINER_TASK2}/config/room_flythrough.yaml"
)
if ${HEADLESS}; then
  CAPTURE_ARGS+=("--headless")
fi
CAPTURE_ARGS+=("${EXTRA_ARGS[@]}")

echo "Isaac Sim container: ${ISAACSIM_CONTAINER}"
echo "Repo mount:          ${REPO_ROOT} -> ${CONTAINER_REPO}"
echo "Room USD:            ${HOST_ROOM_USD}"

# Isaac Sim runs as uid 1001 (admin) in the container; the bind-mounted
# repo is owned by the host user, so create a world-writable captures
# directory on the host before Kit tries to mkdir inside it.
HOST_CAPTURES="${TASK2_ROOT}/captures"
mkdir -p "${HOST_CAPTURES}"
chmod a+rwx "${HOST_CAPTURES}" || true
echo "Captures dir:        ${HOST_CAPTURES}"
echo "Launching room-only orbit capture..."

DOCKER_EXEC_ENV=()
if [[ -t 1 ]]; then
  DOCKER_EXEC_ENV+=("-it")
else
  DOCKER_EXEC_ENV+=("-i")
fi
if [[ -n "${DISPLAY:-}" ]]; then
  DOCKER_EXEC_ENV+=("-e" "DISPLAY=${DISPLAY}")
fi
if [[ -n "${TERM:-}" ]]; then
  DOCKER_EXEC_ENV+=("-e" "TERM=${TERM}")
fi
DOCKER_EXEC_ENV+=("-e" "QT_X11_NO_MITSHM=1")
DOCKER_EXEC_ENV+=("-e" "PYTHONUNBUFFERED=1")

docker exec "${DOCKER_EXEC_ENV[@]}" "${ISAACSIM_CONTAINER}" \
  /isaac-sim/python.sh \
  "${CONTAINER_TASK2}/scripts/capture_room_flythrough.py" \
  "${CAPTURE_ARGS[@]}"
