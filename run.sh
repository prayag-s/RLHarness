#!/usr/bin/env bash
#
# Start the RL Harness backend and frontend together.
#
#   ./run.sh dev     Backend with auto-reload + Vite dev server (hot reload, source maps)
#   ./run.sh prod    Build the SPA once, then serve it from the API origin — no Vite
#
# In dev the browser talks to Vite on FRONTEND_PORT, which proxies /api and /ws to the
# backend. In prod the backend itself serves frontend/dist, so only one server runs.
#
# Overridable: HOST, BACKEND_PORT, FRONTEND_PORT, PYTHON, NPM.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
FRONTEND_DIST="$FRONTEND_DIR/dist"

HOST="${HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

if [[ -t 1 ]]; then
  C_INFO=$'\033[36m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_OFF=$'\033[0m'
else
  C_INFO=''; C_WARN=''; C_ERR=''; C_OFF=''
fi

log()  { printf '%s[run]%s %s\n' "$C_INFO" "$C_OFF" "$*"; }
warn() { printf '%s[run]%s %s\n' "$C_WARN" "$C_OFF" "$*" >&2; }
die()  { printf '%s[run]%s %s\n' "$C_ERR" "$C_OFF" "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: ./run.sh <dev|prod> [--no-build]

  dev    uvicorn --reload on :$BACKEND_PORT + Vite dev server on :$FRONTEND_PORT
  prod   npm run build (once), then uvicorn serving frontend/dist on :$BACKEND_PORT

Options:
  --no-build   prod only: reuse an existing frontend/dist instead of rebuilding
  -h, --help   show this help

Environment:
  HOST           bind address for both servers        (default 127.0.0.1)
  BACKEND_PORT   API port                             (default 8000)
  FRONTEND_PORT  Vite port, dev only                  (default 5173)
  PYTHON         Python interpreter with the backend deps (default .venv/bin/python)
  NPM            npm binary to use                    (default npm)
EOF
}

MODE="${1:-}"
case "$MODE" in
  dev|prod) shift ;;
  -h|--help) usage; exit 0 ;;
  '') usage >&2; exit 2 ;;
  *) die "unknown mode '$MODE' (expected 'dev' or 'prod')" ;;
esac

BUILD=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) BUILD=0 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1'" ;;
  esac
  shift
done

PIDS=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if ((${#PIDS[@]})); then
    log "stopping services..."
    for pid in "${PIDS[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

# --- toolchain checks --------------------------------------------------------------

resolve_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s' "$PYTHON"
  elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    printf '%s' "$ROOT_DIR/.venv/bin/python"
  else
    command -v python3 || true
  fi
}

PYTHON_BIN="$(resolve_python)"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || die "no Python interpreter found. Create one with:
  python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.lock"

if ! (cd "$BACKEND_DIR" && "$PYTHON_BIN" -c "
import importlib.util, sys
missing = [m for m in ('uvicorn', 'fastapi', 'rl_ide') if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
") ; then
  die "backend dependencies missing for $PYTHON_BIN. Install them with:
  $PYTHON_BIN -m pip install -r backend/requirements.lock"
fi

NPM_BIN="${NPM:-npm}"
command -v "$NPM_BIN" >/dev/null 2>&1 || die "npm not found; install Node.js 20+ first."
command -v curl >/dev/null 2>&1 || die "curl not found; it is used to wait for the services to come up."
VITE_BIN="$FRONTEND_DIR/node_modules/.bin/vite"

install_frontend_deps() {
  log "installing frontend dependencies..."
  if [[ -f "$FRONTEND_DIR/package-lock.json" ]]; then
    (cd "$FRONTEND_DIR" && "$NPM_BIN" ci)
  else
    (cd "$FRONTEND_DIR" && "$NPM_BIN" install)
  fi
}

require_frontend_deps() {
  [[ -x "$VITE_BIN" ]] || install_frontend_deps
}

# Waits for a port to accept HTTP connections (any status code counts as "up").
wait_for_http() {
  local url=$1 name=$2 pid=$3 i code
  for i in $(seq 1 120); do
    code="$(curl -s -o /dev/null -m 2 -w '%{http_code}' "$url" 2>/dev/null || true)"
    if [[ -n "$code" && "$code" != "000" ]]; then
      log "$name is up at $url"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    sleep 0.5
  done
  return 1
}

start_backend() {
  local -a args=(
    -m uvicorn rl_ide.api.server:app
    --host "$HOST" --port "$BACKEND_PORT"
  )
  # exec replaces the backgrounded subshell, so the recorded PID *is* uvicorn and
  # signals reach it directly instead of stopping a wrapper that orphans the server.
  cd "$BACKEND_DIR"
  exec "$PYTHON_BIN" "${args[@]}" "$@"
}

# --- dev ---------------------------------------------------------------------------

run_dev() {
  require_frontend_deps
  [[ -x "$VITE_BIN" ]] || die "Vite is not installed in $FRONTEND_DIR/node_modules."

  # A leftover production build must not shadow the dev server's proxy setup.
  unset RL_IDE_FRONTEND_DIST
  ((BUILD)) || warn "ignoring --no-build: dev always runs from source via Vite"

  log "dev: backend on http://$HOST:$BACKEND_PORT (auto-reload)"
  log "dev: frontend on http://$HOST:$FRONTEND_PORT (Vite, /api and /ws proxied)"
  start_backend --reload --reload-dir rl_ide --log-level info &
  local backend_pid=$!
  PIDS+=("$backend_pid")

  if ! wait_for_http "http://$HOST:$BACKEND_PORT/" "backend" "$backend_pid"; then
    die "backend failed to start on port $BACKEND_PORT"
  fi

  ( cd "$FRONTEND_DIR" && exec "$VITE_BIN" --host "$HOST" --port "$FRONTEND_PORT" --strictPort ) &
  local frontend_pid=$!
  PIDS+=("$frontend_pid")

  wait_for_http "http://$HOST:$FRONTEND_PORT/" "frontend" "$frontend_pid" \
    || die "Vite failed to start on port $FRONTEND_PORT"

  log "open http://$HOST:$FRONTEND_PORT"
}

# --- prod --------------------------------------------------------------------------

run_prod() {
  require_frontend_deps

  if ((BUILD)); then
    log "building frontend (production bundle)..."
    ( cd "$FRONTEND_DIR" && "$NPM_BIN" run build )
  else
    warn "--no-build: serving the existing $FRONTEND_DIST"
  fi
  [[ -f "$FRONTEND_DIST/index.html" ]] || die "no production build at $FRONTEND_DIST/index.html"

  log "prod: serving API + built SPA on http://$HOST:$BACKEND_PORT (no Vite)"
  # The API mounts this directory at "/" after every /api and /ws route.
  export RL_IDE_FRONTEND_DIST="$FRONTEND_DIST"
  start_backend --log-level info &
  local backend_pid=$!
  PIDS+=("$backend_pid")

  if ! wait_for_http "http://$HOST:$BACKEND_PORT/" "app" "$backend_pid"; then
    die "backend failed to start on port $BACKEND_PORT"
  fi

  log "open http://$HOST:$BACKEND_PORT"
}

if [[ "$MODE" == "dev" ]]; then
  run_dev
else
  run_prod
fi

set +e
wait -n
status=$?
set -e
log "a service exited (status $status); shutting down"
exit "$status"
