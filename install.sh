#!/usr/bin/env sh
set -eu

INSTALL_DIR=${JARVIS_CODE_INSTALL_DIR:-"$HOME/.local/share/jarvis-code"}
DEFAULT_REPO="jarvis-llm-codec/jlc"
REPO=${JARVIS_CODE_REPO:-$DEFAULT_REPO}
BRANCH=${JARVIS_CODE_BRANCH:-main}
ARCHIVE_URL=${JARVIS_CODE_ARCHIVE_URL:-}
NO_MODEL_PRELOAD=${JARVIS_CODE_NO_MODEL_PRELOAD:-0}
REQUIRE_MODEL_PRELOAD=${JARVIS_CODE_REQUIRE_MODEL_PRELOAD:-0}

log() {
  printf '[jarvis-install] %s\n' "$1"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'JARVIS Code requires %s.\n' "$1" >&2
    exit 1
  fi
}

install_git() {
  if command -v git >/dev/null 2>&1; then
    return 0
  fi
  log "Git not found; attempting automatic install"
  if command -v brew >/dev/null 2>&1; then
    brew install git
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y git
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y git
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y git
  elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -Sy --noconfirm git
  else
    printf 'JARVIS Code requires Git, and no supported package manager was found.\n' >&2
    exit 1
  fi
  if ! command -v git >/dev/null 2>&1; then
    printf 'Git installation did not make git available on PATH.\n' >&2
    exit 1
  fi
}

assert_node_version() {
  ver=$(node --version)
  major=$(printf '%s\n' "$ver" | sed 's/^v//' | cut -d. -f1)
  case "$major" in
    ''|*[!0-9]*)
      printf 'Could not parse Node.js version: %s\n' "$ver" >&2
      exit 1
      ;;
  esac
  if [ "$major" -lt 20 ]; then
    printf 'Node.js 20 or newer is required; found %s\n' "$ver" >&2
    exit 1
  fi
  printf '%s\n' "$ver"
}

find_python() {
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

ensure_python_venv() {
  # On Debian/Ubuntu the venv module (ensurepip) ships in a separate package;
  # without it `python -m venv` yields a pip-less, broken environment and the
  # rest of the install silently produces a non-working tree. Auto-install it,
  # the same way install_git bootstraps git.
  py=$1
  if "$py" -c 'import ensurepip' >/dev/null 2>&1; then
    return 0
  fi
  pyver=$("$py" -c 'import sys; print("python%d.%d" % sys.version_info[:2])' 2>/dev/null || printf 'python3')
  log "Python venv/ensurepip not available; attempting automatic install of ${pyver}-venv"
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y "${pyver}-venv" || sudo apt-get install -y python3-venv
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y python3 python3-pip
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y python3 python3-pip
  elif command -v pacman >/dev/null 2>&1; then
    : # Arch's python package already includes venv and ensurepip
  elif command -v brew >/dev/null 2>&1; then
    : # Homebrew's python includes venv and ensurepip
  fi
  if ! "$py" -c 'import ensurepip' >/dev/null 2>&1; then
    printf 'JARVIS Code requires the Python venv module with ensurepip. Install %s-venv (Debian/Ubuntu) or the equivalent and retry.\n' "$pyver" >&2
    exit 1
  fi
}

script_dir() {
  # This works for local script execution. When piped through sh, the installer
  # falls back to downloading the release archive.
  case "$0" in
    /*) dirname "$0" ;;
    */*) dirname "$(pwd)/$0" ;;
    *) pwd ;;
  esac
}

is_local_package() {
  d=$1
  [ -f "$d/jarvis.ps1" ] && [ -d "$d/sidecar" ] && [ -d "$d/pi" ]
}

copy_package() {
  src=$1
  dst=$2
  mkdir -p "$dst"
  if [ "$(cd "$src" && pwd)" = "$(cd "$dst" && pwd)" ]; then
    return
  fi
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete \
      --exclude '.git' \
      --exclude '_internal' \
      --exclude 'data' \
      --exclude 'pi-agent' \
      --exclude 'pi/node_modules' \
      --exclude 'sidecar/.venv' \
      --exclude '.pytest_cache' \
      --exclude '__pycache__' \
      "$src"/ "$dst"/
  else
    tmp_tar=$(mktemp)
    (cd "$src" && tar \
      --exclude './.git' \
      --exclude './_internal' \
      --exclude './data' \
      --exclude './pi-agent' \
      --exclude './pi/node_modules' \
      --exclude './sidecar/.venv' \
      --exclude './.pytest_cache' \
      --exclude './__pycache__' \
      -cf "$tmp_tar" .)
    (cd "$dst" && tar -xf "$tmp_tar")
    rm -f "$tmp_tar"
  fi
}

download_package() {
  dst=$1
  if [ -z "$ARCHIVE_URL" ]; then
    if [ -z "$REPO" ]; then
      printf 'Set JARVIS_CODE_REPO=jarvis-llm-codec/jlc or JARVIS_CODE_ARCHIVE_URL before running the remote installer.\n' >&2
      exit 1
    fi
    ARCHIVE_URL="https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz"
  fi

  tmp_dir=$(mktemp -d)
  archive="$tmp_dir/jarvis-code.tar.gz"
  extract="$tmp_dir/extract"
  mkdir -p "$extract"
  log "downloading $ARCHIVE_URL"
  curl -fsSL "$ARCHIVE_URL" -o "$archive"
  tar -xzf "$archive" -C "$extract"
  src=$(find "$extract" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  if [ -z "$src" ]; then
    printf 'Archive did not contain a source folder.\n' >&2
    exit 1
  fi
  copy_package "$src" "$dst"
}

install_node_deps() {
  root=$1
  need_cmd npm
  cd "$root/pi"
  if [ -f package-lock.json ]; then
    log "installing Node dependencies with npm ci"
    HUSKY=0 npm ci --include=dev
  else
    log "installing Node dependencies with npm install"
    HUSKY=0 npm install --include=dev
  fi
}

install_sidecar_venv() {
  root=$1
  if ! py=$(find_python); then
    printf 'JARVIS Code requires Python 3.10 or newer.\n' >&2
    exit 1
  fi
  sidecar="$root/sidecar"
  venv="$sidecar/.venv"
  venv_py="$venv/bin/python"
  if [ ! -x "$venv_py" ]; then
    ensure_python_venv "$py"
    log "creating sidecar venv"
    "$py" -m venv "$venv"
  fi
  # A venv built without ensurepip leaves no pip and the dependency install below
  # would fail confusingly; surface it as a clear, fatal error instead.
  if [ ! -x "$venv_py" ] || ! "$venv_py" -m pip --version >/dev/null 2>&1; then
    printf 'Sidecar venv has no working pip at %s. Install the Python venv package (e.g. python3-venv) and retry.\n' "$venv_py" >&2
    exit 1
  fi
  log "installing sidecar Python dependencies"
  "$venv_py" -m pip install --disable-pip-version-check --quiet --upgrade pip 'setuptools<82' wheel
  "$venv_py" -m pip install --disable-pip-version-check -r "$sidecar/requirements.txt"
}

preload_embedder_model() {
  root=$1
  if [ "$NO_MODEL_PRELOAD" = "1" ]; then
    log "skipping bge-m3 preload (disabled)"
    return
  fi
  venv_py="$root/sidecar/.venv/bin/python"
  doctor="$root/scripts/jarvis-doctor.py"
  if [ ! -x "$venv_py" ]; then
    printf 'Cannot preload bge-m3 because sidecar venv Python was not found at %s\n' "$venv_py" >&2
    exit 1
  fi
  if [ ! -f "$doctor" ]; then
    printf 'Cannot preload bge-m3 because JARVIS doctor was not found at %s\n' "$doctor" >&2
    exit 1
  fi
  log "preloading bge-m3 embedding model (first install may download about 2.3 GB)"
  if ! "$venv_py" "$doctor" --preload-embedder --require-embedder --skip-sidecar; then
    if [ "$REQUIRE_MODEL_PRELOAD" = "1" ]; then
      printf 'bge-m3 preload failed. Rerun with JARVIS_CODE_REQUIRE_MODEL_PRELOAD=0 to allow degraded install.\n' >&2
      exit 1
    fi
    printf 'Warning: bge-m3 preload failed. Install will continue; run `jarvis doctor --preload-embedder` after install to see details.\n' >&2
  fi
}

bootstrap_default_resources() {
  root=$1
  resources="$root/jarvis-resources"
  pi_agent="$root/pi-agent"
  mkdir -p "$pi_agent"
  if [ -d "$resources/skills" ]; then
    mkdir -p "$pi_agent/skills"
    cp -R "$resources/skills"/. "$pi_agent/skills"/
  fi
  if [ -d "$resources/themes" ]; then
    mkdir -p "$pi_agent/themes"
    cp -R "$resources/themes"/. "$pi_agent/themes"/
  fi

  settings="$pi_agent/settings.json"
  python_for_settings=$(find_python || true)
  if [ -n "$python_for_settings" ]; then
    "$python_for_settings" - "$settings" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise SystemExit(0)
    if not isinstance(data, dict):
        raise SystemExit(0)
else:
    data = {}
if isinstance(data.get("theme"), str) and data["theme"].strip():
    raise SystemExit(0)
data["theme"] = "orange-blue"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
  fi
}

install_command() {
  root=$1
  chmod +x "$root/jarvis.sh"
  bin_dir=${JARVIS_CODE_BIN_DIR:-"$HOME/.local/bin"}
  mkdir -p "$bin_dir"
  ln -sfn "$root/jarvis.sh" "$bin_dir/jarvis"
  log "installed command shim at $bin_dir/jarvis"
  case ":$PATH:" in
    *":$bin_dir:"*) ;;
    *) log "add $bin_dir to PATH if jarvis is not found" ;;
  esac
}

need_cmd node
node_version=$(assert_node_version)
log "using Node $node_version"
install_git
log "using $(git --version)"

src_dir=$(script_dir)
mkdir -p "$INSTALL_DIR"

if is_local_package "$src_dir"; then
  log "installing from local package $src_dir"
  copy_package "$src_dir" "$INSTALL_DIR"
else
  download_package "$INSTALL_DIR"
fi

mkdir -p "$INSTALL_DIR/data" "$INSTALL_DIR/pi-agent"
bootstrap_default_resources "$INSTALL_DIR"
install_node_deps "$INSTALL_DIR"
install_sidecar_venv "$INSTALL_DIR"
install_command "$INSTALL_DIR"
preload_embedder_model "$INSTALL_DIR"

log "installed JARVIS Code at $INSTALL_DIR"
log "run: jarvis"
log "diagnostics: jarvis doctor"
