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
# Every invocation is a clean restart: an instance left running from a previous dev or
# prod launch (even on another port) is stopped first. Processes that are not part of
# this app are never killed — the script fails with the offending PID instead.
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

Both modes first stop any instance previously started from this directory, then start
fresh. Ports held by unrelated processes are reported instead of being killed.

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

kill_tree() {
  local pid=$1 child
  for child in $(ps -o pid= --ppid "$pid" 2>/dev/null); do
    kill_tree "$child"
  done
  kill "$pid" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if ((${#PIDS[@]})); then
    log "stopping services..."
    for pid in "${PIDS[@]}"; do
      kill_tree "$pid"
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

# --- stopping a previous instance --------------------------------------------------
#
# The API and Vite are recognised by what they run (the uvicorn module string, the
# frontend's local Vite binary) and by the directory they run from, so a leftover
# launcher shell, a reload worker, or an instance started on another port is still
# found — while an unrelated process that merely owns the port is left alone.

SERVICE_MARKER='rl_ide.api.server:app'

# Ports this run needs for itself, as "port:label" pairs.
PORTS_NEEDED=("$BACKEND_PORT:backend")
[[ "$MODE" == "dev" ]] && PORTS_NEEDED+=("$FRONTEND_PORT:frontend")

proc_cmd() { ps -o command= -p "$1" 2>/dev/null || true; }

proc_cwd() {
  if [[ -d "/proc/$1" ]]; then
    readlink "/proc/$1/cwd" 2>/dev/null || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1
  fi
}

port_listeners() {
  local port=$1
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnpH 2>/dev/null | awk -v suffix=":$port" '
      {
        n = split($4, parts, ":")
        if (parts[n] != substr(suffix, 2)) next
        while (match($0, /pid=[0-9]+/)) {
          print substr($0, RSTART + 4, RLENGTH - 4)
          $0 = substr($0, RSTART + RLENGTH)
        }
      }' || true
  fi
}

ancestors_of() {
  local cur=$1 depth=0 ppid
  while ((depth++ < 16)); do
    ppid="$(ps -o ppid= -p "$cur" 2>/dev/null | tr -d '[:space:]')"
    [[ -n "$ppid" && "$ppid" -gt 1 ]] || return 0
    printf '%s\n' "$ppid"
    cur=$ppid
  done
}

SELF_AND_ANCESTORS=" $$ $(ancestors_of $$ | tr '\n' ' ') "
is_self_pid() { [[ "$SELF_AND_ANCESTORS" == *" $1 "* ]]; }

is_within_dir() {
  [[ -n "$1" && ($1 == "${2%/}" || $1 == "${2%/}"/*) ]]
}

proc_exe() {
  local exe=${1%% *}
  printf '%s' "${exe##*/}"
}

is_backend_proc() {
  local pid=$1 cmd cwd
  cmd="$(proc_cmd "$pid")"
  # Only real servers: a launcher shell whose command *text* mentions uvicorn is not
  # one of ours to kill (it may be a supervisor/watchdog around someone else's work).
  case "$(proc_exe "$cmd")" in
    python|python3|python3.*|uvicorn) ;;
    *) return 1 ;;
  esac
  cwd="$(proc_cwd "$pid")"
  # Must be an interpreter started from this backend, not some other uvicorn elsewhere.
  is_within_dir "$cwd" "$BACKEND_DIR" || return 1
  if [[ "$cmd" == *"$SERVICE_MARKER"* || "$cmd" == *uvicorn* ]]; then
    return 0
  fi
  # `uvicorn --reload` serves from a spawn worker that carries no arguments of its own.
  [[ "$cmd" == *spawn_main* || "$cmd" == *multiprocessing-fork* ]]
}

is_frontend_proc() {
  local pid=$1 cmd cwd
  cmd="$(proc_cmd "$pid")"
  case "$(proc_exe "$cmd")" in
    node|vite|vite.js) ;;
    *) return 1 ;;
  esac
  [[ "$cmd" == *vite* ]] || return 1
  cwd="$(proc_cwd "$pid")"
  is_within_dir "$cwd" "$FRONTEND_DIR"
}

discover_existing_pids() {
  local pid
  for pid in $(port_listeners "$BACKEND_PORT") $(port_listeners "$FRONTEND_PORT"); do
    is_self_pid "$pid" && continue
    if is_backend_proc "$pid" || is_frontend_proc "$pid"; then printf '%s\n' "$pid"; fi
  done
  for pid in $(pgrep -f "$SERVICE_MARKER" 2>/dev/null || true); do
    is_self_pid "$pid" && continue
    is_backend_proc "$pid" && printf '%s\n' "$pid"
  done
  for pid in $(pgrep -f 'node_modules/(\.bin/)?vite' 2>/dev/null || true); do
    is_self_pid "$pid" && continue
    is_frontend_proc "$pid" && printf '%s\n' "$pid"
  done
  for pid in $(pgrep -f 'multiprocessing\.spawn|multiprocessing-fork' 2>/dev/null || true); do
    is_self_pid "$pid" && continue
    is_backend_proc "$pid" && printf '%s\n' "$pid"
  done
}

# Refuses to guess: a port owned by something that is not this app is reported, not killed.
assert_ports_ours() {
  local spec port label pid foreign=0
  for spec in "${PORTS_NEEDED[@]}"; do
    port=${spec%%:*}
    label=${spec##*:}
    for pid in $(port_listeners "$port"); do
      is_self_pid "$pid" && continue
      if ! is_backend_proc "$pid" && ! is_frontend_proc "$pid"; then
        warn "$label port $port is held by an unrelated process: pid $pid ($(proc_cmd "$pid"))"
        foreign=1
      fi
    done
  done
  ((foreign == 0)) || die "not killing unrelated processes; stop them and re-run."
}

wait_for_port_free() {
  local port=$1 label=$2 i pid holders
  for i in $(seq 1 60); do
    [[ -z "$(port_listeners "$port")" ]] && return 0
    sleep 0.25
  done
  holders=""
  for pid in $(port_listeners "$port"); do
    holders+="pid $pid ($(proc_cmd "$pid")); "
  done
  die "$label port $port is still in use by ${holders:-an unknown process}"
}

terminate_pids() {
  local -a pids=("$@")
  local pid i alive
  for pid in "${pids[@]}"; do
    log "  pid $pid: $(proc_cmd "$pid" | cut -c1-160)"
    kill -TERM "$pid" 2>/dev/null || true
  done
  for i in $(seq 1 40); do
    alive=0
    for pid in "${pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive=1
    done
    ((alive)) || break
    sleep 0.25
  done
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      warn "pid $pid ignored SIGTERM; sending SIGKILL"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

stop_existing_services() {
  local -a pids=()
  local pid stopped=0

  while read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(discover_existing_pids | sort -un)

  if ((${#pids[@]})); then
    log "stopping ${#pids[@]} existing process(es)..."
    terminate_pids "${pids[@]}"
    stopped=1
  fi

  # A --reload worker survives its reloader for a moment and can reclaim the port.
  pids=()
  while read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(discover_existing_pids | sort -un)

  if ((${#pids[@]})); then
    log "stopping ${#pids[@]} leftover reload process(es)..."
    terminate_pids "${pids[@]}"
    stopped=1
  fi

  if ((stopped)); then
    log "restarting cleanly"
  else
    log "no running instance found"
  fi

  local spec
  for spec in "${PORTS_NEEDED[@]}"; do
    wait_for_port_free "${spec%%:*}" "${spec##*:}"
  done
}

# --- dev ---------------------------------------------------------------------------

run_dev() {
  require_frontend_deps
  [[ -x "$VITE_BIN" ]] || die "Vite is not installed in $FRONTEND_DIR/node_modules."

  # A leftover production build must not shadow the dev server's proxy setup.
  unset RL_IDE_FRONTEND_DIST
  ((BUILD)) || warn "ignoring --no-build: dev always runs from source via Vite"

  stop_existing_services

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

  stop_existing_services

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

assert_ports_ours

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
